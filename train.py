"""Trainining script for WaveNet vocoder

usage: train.py [options]

options:
    --run-name=<str>             Name the process to log the info.
    --device=<N>                 Select the device to run the model. -1: CPU, >0 GPU_id.
    --phase=<str>                Train or synthesis.
    --data-root=<dir>            Directory contains preprocessed features.
    --checkpoint-name=<path>     Select the chechpoint to load into the model.
    --hparams=<parmas>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --log-event-path=<name>      Log event path.
    --reset-optimizer            Reset optimizer.
    --speaker-id=<N>             Use specific speaker of data in case for multi-speaker datasets.
    --text-list-file=<path>      Use specific file to synthesis the melspectrum.
    -h, --help                   Show this help message and exit
"""

from docopt import docopt
import argparse
from utils import *
import os
import infolog
import shutil
import time
import warnings
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np

from model import builder
from model.loss import MaskedBCELoss, MaskedMSELoss

from hparams import hparams, hparams_debug_string


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets.dataloader import AudiobookDataset, AudioCollate, _ch_symbol_to_id, text_to_seq, get_item_list
from datasets.dataloader import SimilarTimeLengthSampler, DynamicalSimilarTimeLengthSampler, DynamicalBatchSampler
from tensorboardX import SummaryWriter

from os.path import dirname, join, expanduser
from utils import ValueWindow, time_string, plot_alignment


import infolog
log = infolog.log

global best_loss
global global_epoch
global global_step

torch.backends.cudnn.enabled = False

def train(train_loader, model, device, mels_criterion, stop_criterion, optimizer, scheduler, writer, train_dir):
    batch_time = ValueWindow()
    data_time = ValueWindow()
    losses = ValueWindow()

    # switch to train mode
    model.train()

    end = time.time()
    global global_epoch
    global global_step
    for i, (txts, mels, stop_tokens, txt_lengths, mels_lengths) in enumerate(train_loader):
        scheduler.adjust_learning_rate(optimizer, global_step)
        # measure data loading time
        data_time.update(time.time() - end)

        if device > -1:
            txts = txts.cuda(device)
            mels = mels.cuda(device)
            stop_tokens = stop_tokens.cuda(device)
            txt_lengths = txt_lengths.cuda(device)
            mels_lengths = mels_lengths.cuda(device)

        # compute output
        frames, decoder_frames, stop_tokens_predict, alignment = model(txts, txt_lengths, mels)
        decoder_frames_loss = mels_criterion(decoder_frames, mels, lengths=mels_lengths)
        frames_loss = mels_criterion(frames, mels, lengths=mels_lengths)
        stop_token_loss = stop_criterion(stop_tokens_predict, stop_tokens, lengths=mels_lengths)
        loss = decoder_frames_loss + frames_loss + stop_token_loss

        #print(frames_loss, decoder_frames_loss)
        losses.update(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if hparams.clip_thresh > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.get_trainable_parameters(), hparams.clip_thresh)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % hparams.print_freq == 0:
            log('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(global_epoch, i, len(train_loader),
                                                              batch_time=batch_time, data_time=data_time, loss=losses)
                )

        # Logs
        writer.add_scalar("loss", float(loss.item()), global_step)
        writer.add_scalar("avg_loss in {} window".format(losses.get_dinwow_size), float(losses.avg), global_step)
        writer.add_scalar("stop_token_loss", float(stop_token_loss.item()), global_step)
        writer.add_scalar("decoder_frames_loss", float(decoder_frames_loss.item()), global_step)
        writer.add_scalar("output_frames_loss", float(frames_loss.item()), global_step)

        if hparams.clip_thresh > 0:
            writer.add_scalar("gradient norm", grad_norm, global_step)
        writer.add_scalar("learning rate", optimizer.param_groups[0]['lr'], global_step)
        global_step += 1

    dst_alignment_path = join(train_dir, "{}_alignment.png".format(global_step))
    alignment = alignment.cpu().detach().numpy()
    plot_alignment(alignment[0, :txt_lengths[0], :mels_lengths[0]], dst_alignment_path, info="{}, {}".format(hparams.builder, global_step))


def validate(val_loader, model, device, mels_criterion, stop_criterion, writer, val_dir):
    batch_time = ValueWindow()
    losses = ValueWindow()

    # switch to evaluate mode
    model.eval()

    global global_epoch
    global global_step
    with torch.no_grad():
        end = time.time()
        for i, (txts, mels, stop_tokens, txt_lengths, mels_lengths) in enumerate(val_loader):
            # measure data loading time
            batch_time.update(time.time() - end)

            if device > -1:
                txts = txts.cuda(device)
                mels = mels.cuda(device)
                stop_tokens = stop_tokens.cuda(device)
                txt_lengths = txt_lengths.cuda(device)
                mels_lengths = mels_lengths.cuda(device)

            # compute output
            frames, decoder_frames, stop_tokens_predict, alignment = model(txts, txt_lengths, mels)
            decoder_frames_loss = mels_criterion(decoder_frames, mels, lengths=mels_lengths)
            frames_loss = mels_criterion(frames, mels, lengths=mels_lengths)
            stop_token_loss = stop_criterion(stop_tokens_predict, stop_tokens, lengths=mels_lengths)
            loss = decoder_frames_loss + frames_loss + stop_token_loss

            losses.update(loss.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % hparams.print_freq == 0:
                log('Epoch: [{0}]\t'
                    'Test: [{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(global_epoch, i, len(val_loader),
                                                                  batch_time=batch_time, loss=losses)
                    )
            # Logs
            writer.add_scalar("loss", float(loss.item()), global_step)
            writer.add_scalar("avg_loss in {} window".format(losses.get_dinwow_size), float(losses.avg),
                              global_step)
            writer.add_scalar("stop_token_loss", float(stop_token_loss.item()), global_step)
            writer.add_scalar("decoder_frames_loss", float(decoder_frames_loss.item()), global_step)
            writer.add_scalar("output_frames_loss", float(frames_loss.item()), global_step)


        dst_alignment_path = join(val_dir, "{}_alignment.png".format(global_step))
        alignment = alignment.cpu().detach().numpy()
        plot_alignment(alignment[0, :txt_lengths[0], :mels_lengths[0]], dst_alignment_path, info="{}, {}".format(hparams.builder, global_step))

    return losses.avg


def synthesis(test_lines, model, device, log_dir):
    global global_epoch
    global global_step
    synthesis_dir = os.path.join(log_dir, "synthesis_mels")
    os.makedirs(synthesis_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, line in enumerate(test_lines):
            txt = text_to_seq(line)
            if device > -1:
                txt = txt.cuda(device)
            frames, _, _, alignment = model(txt)

            dst_alignment_path = join(synthesis_dir, "{}_alignment_{}.png".format(global_step, idx))
            dst_mels_path = join(synthesis_dir, "{}_mels_{}.npy".format(global_step, idx))
            plot_alignment(alignment.T, dst_alignment_path, info="{}, {}".format(hparams.builder, global_step))
            np.save(dst_mels_path, frames)


class ExpLRDecay(object):
    def __init__(self, init_learning_rate, decay_rate, start_step, decay_steps):
        self.init_learning_rate = init_learning_rate
        self.decay_rate = decay_rate
        self.start_step = start_step
        self.decay_steps = decay_steps

    def adjust_learning_rate(self, optimizer, step):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.init_learning_rate * (1-self.decay_rate) ** ((step - self.start_step) / self.decay_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def save_checkpoint(model, optimizer, checkpoint_dir):
    global global_epoch
    global global_step
    checkpoint_path = join(checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": global_step,
        "global_epoch": global_epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path, device):
    if device > -1:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, device, optimizer, reset_optimizer):
    global global_epoch
    global global_step
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, device)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


def build_model():
    num_chars = len(_ch_symbol_to_id) + 1
    model = getattr(builder, hparams.builder)(
        num_chars=num_chars,
        max_decoder_steps=hparams.max_decoder_steps,
        frames_per_step=hparams.frames_per_step,
        dim_embedding=hparams.dim_embedding,
        dim_encoder=hparams.dim_encoder,
        enc_kernel_size=hparams.enc_kernel_size,
        num_mels=hparams.num_mels,
        dim_attention=hparams.dim_attention,
        dim_decoder=hparams.dim_decoder,
        dim_prenet=hparams.dim_prenet,
        num_layers=hparams.num_layers,
        num_location_features=hparams.num_location_features,
        gate_threshold=hparams.gate_threshold,
        dec_num_filters=hparams.dec_num_filters,
        dec_kernel_size=hparams.dec_kernel_size,
        batch_size=hparams.batch_size
    )
    return model


def prepare_run(run_name):
    log_dir = os.path.join(dirname(__file__), 'logs-{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name)
    return log_dir


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    run_name = args["--run-name"]  # dataset root
    device = args["--device"]
    phase = args["--phase"]        # train or synthesis
    data_root = args["--data-root"]                # dataset root
    checkpoint_name = args["--checkpoint-name"]
    speaker_id = args["--speaker-id"]
    log_event_path = args["--log-event-path"]
    reset_optimizer = args["--reset-optimizer"]
    text_list_file_path = args["--text-list-file"]

    preset = args["--preset"]

    speaker_id = int(speaker_id) if speaker_id is not None else None

    if run_name is None:
        run_name = "Tacotron2" + time_string()
    log_dir = prepare_run(run_name)

    if data_root is None:
        data_root = os.path.join(dirname(__file__), "data", "mandarin")

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])

    assert hparams.builder == "Tacotron2"

    if device is not None:
        hparams.device = device

    print(hparams_debug_string())

    train_path = os.path.join(log_dir, "train")
    val_path = os.path.join(log_dir, "val")
    checkpoint_path = os.path.join(log_dir, "pretrained")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    best_loss = 0
    global global_epoch
    global_epoch = 0
    global global_step
    global_step = 0

    if hparams.seed is not None:
        random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        cudnn.deterministic = hparams.cudnn_deterministic
        cudnn.benchmark = hparams.cudnn_benchmark
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        log("The system set the random number to:{}".format(hparams.seed))

    if hparams.device > -1:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    distributed = hparams.world_size > 1
    if distributed:
        dist.init_process_group(backend=hparams.dist_backend, init_method=hparams.dist_url,
                                world_size=hparams.world_size)
    model = build_model()
    print(model)

    if hparams.device > -1:
        model = model.cuda(hparams.device)
    elif distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    mels_criterion = MaskedMSELoss()
    stop_criterion = MaskedBCELoss()

    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=hparams.init_learning_rate,
                                 betas=(hparams.adam_beta1, hparams.adam_beta2), eps=hparams.adam_epsilon,
                                 weight_decay=hparams.weight_decay)
    scheduler = ExpLRDecay(init_learning_rate=hparams.init_learning_rate, decay_rate=hparams.decay_rate,
                           start_step=hparams.start_decay, decay_steps=hparams.decay_step)

    # optionally resume from a checkpoint
    if checkpoint_name is not None:
        if os.path.isfile(checkpoint_name):
            load_checkpoint(checkpoint_name, model, hparams.device, optimizer, reset_optimizer)
        else:
            file_full_path = os.path.join(checkpoint_path, checkpoint_name)
            if os.path.isfile(file_full_path):
                load_checkpoint(file_full_path, model, hparams.device, optimizer, reset_optimizer)
            else:
                log("=> no checkpoint found at '{}'".format(checkpoint_name))

    # synthesis
    if phase == "synthesis":
        if text_list_file_path is None:
            test_lines = [
                "yun2cong2ke1ji4cheng2li4yu2er4ling2yi1wu3nian2si4yue4",
                "shi4yi1jia1fu1hua4yu2zhong1guo2ke1xue2yuan4chong2qing4yan2jiu1yuan4de0gao1ke1ji4qi3ye4"
                "zhuan1zhu4yu2ji4suan4ji1shi4jue2yu3ren2gong1zhi4neng2",
                "yi2ge4hao3zheng4quan2zhi1de2yi3bao3chi2da4bu4fen4zai4yu2bu4tong2de0zheng4jian4",
                "he2li3de0fa1hui1qi2gong1yong4"
            ]
        else:
            test_lines = []
            with open(text_list_file_path, "rb") as f:
                lines = f.readlines()
                for line in lines:
                    text = line.decode("utf-8")[:-1]
                    test_lines.append(text)
        synthesis(test_lines, model, device, log_dir)
        return

    # Setup summary writer for tensorboard
    if log_event_path is None:
        log_event_path = os.path.join(log_dir, "log_event_path")
    print("Los event path: {}".format(log_event_path))
    writer = SummaryWriter(log_dir=log_event_path)

    # Prepare dataset
    dataset_dir = os.path.join(dirname(__file__), data_root)
    texts_list, mels_list, mels_length_list, speaker_ids_list = get_item_list(dataset_dir, "train.txt")

    #indices = np.arange(256*16)
    indices = np.arange(len(texts_list) - len(texts_list) % hparams.batch_size)
    test_size = hparams.test_batches * hparams.batch_size
    train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=hparams.seed)
    collate_fn = AudioCollate(padding_mels=hparams.padding_mels)

    # prepare train dataset
    train_dataset_text_ids = [texts_list[i] for i in train_indices]
    train_dataset_mels_ids = [mels_list[i] for i in train_indices]
    train_dataset_mels_length_ids = [mels_length_list[i] for i in train_indices]
    if speaker_ids_list is not None:
        train_dataset_speaker_ids = [speaker_ids_list[i] for i in train_indices]
    else:
        train_dataset_speaker_ids = None
    train_dataset = AudiobookDataset(train_dataset_text_ids, train_dataset_mels_ids, train_dataset_speaker_ids,
                                     dataset_dir)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=hparams.batch_size,
                                  num_workers=2, shuffle=True, pin_memory=hparams.pin_memory)
    else:
        if hparams.dynamical_batch_size:
            train_sampler = DynamicalSimilarTimeLengthSampler(train_dataset_mels_length_ids,
                                                              batch_size_min=hparams.batch_size,
                                                              batch_expand_level=hparams.batch_size_level,
                                                              batch_group=hparams.batch_group,
                                                              permutate=hparams.permutate)
            train_batch_sampler = DynamicalBatchSampler(train_sampler)
            train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=hparams.batch_size,
                                      batch_sampler=train_batch_sampler, num_workers=2, shuffle=False, pin_memory=True)
        else:
            train_sampler = SimilarTimeLengthSampler(train_dataset_mels_length_ids, descending=True,
                                                     batch_size=hparams.batch_size,
                                                     batch_group_size=hparams.batch_group_size,
                                                     permutate=hparams.permutate)
            train_sampler = None
            shuffle = (train_sampler == None)
            train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=hparams.batch_size,
                                      sampler=train_sampler, num_workers=2, shuffle=False, pin_memory=True)

    # prepare val dataset
    val_dataset_text_ids = [texts_list[i] for i in val_indices]
    val_dataset_mels_ids = [mels_list[i] for i in val_indices]
    val_dataset_mels_length_ids = [mels_length_list[i] for i in val_indices]
    if speaker_ids_list is not None:
        val_dataset_speaker_ids = [speaker_ids_list[i] for i in val_indices]
    else:
        val_dataset_speaker_ids = None

    val_dataset = AudiobookDataset(val_dataset_text_ids, val_dataset_mels_ids, val_dataset_speaker_ids, dataset_dir)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=hparams.batch_size, num_workers=2,
                            shuffle=True, pin_memory=True)

    for epoch in range(global_epoch, hparams.nepochs):
        # train for one epoch
        train(train_loader, model, hparams.device, mels_criterion, stop_criterion, optimizer, scheduler, writer,
              train_path)

        # evaluate on validation set
        loss = validate(val_loader, model, hparams.device, mels_criterion, stop_criterion, writer, val_path)

        # remember best prec@1 and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint(model, optimizer, checkpoint_path)


if __name__ == '__main__':
    main()
