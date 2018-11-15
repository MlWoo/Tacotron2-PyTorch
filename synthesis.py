import argparse
import os
import torch
from model import Model
import librosa
import numpy as np
import pickle
import random


sample_rate = 22050


def generate(model, data_path, output_path, bits, samples=3):
    ids = os.path.join(data_path, 'dataset_ids.pkl')
    with open(ids, 'rb') as f:
        dataset_ids = pickle.load(f)

    random.shuffle(dataset_ids)

    test_mels = []
    ground_truth = []
    for i in range(samples):
        name_id = dataset_ids[i]
        test_mels.append(np.load(f'{data_path}mels{name_id}.npy'))
        ground_truth.append(np.load(f'{data_path}quant{name_id}.npy'))

    for i, (gt, mel) in enumerate(zip(ground_truth, test_mels)):
        gt = 2 * gt.astype(np.float32) / (2**bits - 1.) - 1.
        librosa.output.write_wav(f'{output_path}_{i}_target.wav', gt, sr=sample_rate)
        output = model.generate(mel)
        librosa.output.write_wav(f'{output_path}_{i}_generated.wav', output, sample_rate)


def main():
    parser = argparse.ArgumentParser(description='WaveRNN-PyTorch Training')
    parser.add_argument('--data_path', metavar='DIR', default='/home/lynn/workspace/wumenglin/WaveRNN_pytorch/dataset/',
                        help='path to data')
    parser.add_argument('--model_path', metavar='DIR', default='/home/lynn/workspace/wumenglin/WaveRNN_pytorch/',
                        help='path to model')
    parser.add_argument('--output_dir', metavar='DIR', default='/home/lynn/workspace/wumenglin/WaveRNN_pytorch/outputwavs/',
                        help='path to output wavs')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('-q', '--quant_bits', default=9, type=int,
                        metavar='N', help='quantilization bits (default: 9)')

    args = parser.parse_args()

    # create model
    model = Model(rnn_dims=512, fc_dims=512, bits=args.quant_bits, pad=2,
                  upsample_factors=(5, 5, 11), feat_dims=80,
                  compute_dims=128, res_out_dims=128, res_blocks=10)

    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path + "checkpoint.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])

    if args.gpu is not None:
        model = model.cuda(args.gpu)

    generate(model, data_path=args.data_path, output_path=args.output_dir, bits=args.quant_bits)


if __name__ == '__main__':
    main()