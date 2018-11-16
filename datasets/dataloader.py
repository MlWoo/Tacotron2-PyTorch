import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from utils import *
import os
import random


_eos = '~'
_characters_ch = 'abcdefghijklmnopqrstuvwxyz 01234'
symbols_ch = [_eos] + list(_characters_ch)


_ch_symbol_to_id = {s: i for i, s in enumerate(symbols_ch)}


def _pad_1d(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0, constant_values=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)], mode="constant", constant_values=constant_values)
    return x


def text_to_seq(txt):
    seq = []
    for char in txt:
        seq.append(_ch_symbol_to_id[char])
    seq.append(_ch_symbol_to_id['~'])
    return seq


def get_item_list(data_root, file_name="train.txt"):
    print(data_root)
    meta = os.path.join(data_root, file_name)
    with open(meta, "rb") as f:
        lines = f.readlines()
    l_ = lines[0].decode("utf-8").split("|")
    assert len(l_) == 6 or len(l_) == 7
    multi_speaker = len(l_) == 7
    texts_list = list(map(lambda l: l.decode("utf-8").split("|")[5][:-1], lines))
    mels_list = list(map(lambda l: l.decode("utf-8").split("|")[1], lines))
    mels_length_list = list(map(lambda l: int(l.decode("utf-8").split("|")[4]), lines))
    if multi_speaker:
        speaker_ids_list = list(map(lambda l: int(l.decode("utf-8").split("|")[-1]), lines))
    else:
        speaker_ids_list = None

    return texts_list, mels_list, mels_length_list, speaker_ids_list


class AudiobookDataset(Dataset):
    def __init__(self, text_ids, mels_ids, speaker_ids, path):
        self.path = path
        self.text_ids = text_ids
        self.mels_ids = mels_ids
        self.speaker_ids = speaker_ids

    def __getitem__(self, index):
        text = self.text_ids[index]
        mels = np.load(os.path.join(self.path, 'mels', self.mels_ids[index]))
        if self.speaker_ids is not None:
            speaker = self.speaker_ids[index]
        else:
            speaker = None

        return text, mels, speaker

    def __len__(self):
        return len(self.text_ids)


class AudioCollate:
    def __init__(self, padding_mels):
        padding_idx = len(_ch_symbol_to_id)
        self.padding_idx = padding_idx
        self.padding_mels = padding_mels

    def __call__(self, batch):
        #print("========>", text_to_seq(batch[0][0]))
        txt_enc = [text_to_seq(x[0]) for x in batch]
        txt_lengths = [len(x) for x in txt_enc]
        max_txt_length = max(txt_lengths)

        mels_lengths = [len(x[1]) for x in batch]
        max_mels_length = max(mels_lengths)

        stop_tokens = [np.asarray([0.] * (mels_length - 1)) for mels_length in mels_lengths]

        txt_batch = np.stack(_pad_1d(x, max_txt_length, constant_values=self.padding_idx) for x in txt_enc)
        mels_batch = np.stack(_pad_2d(x[1], max_mels_length, b_pad=0, constant_values=self.padding_mels) for x in batch)
        stop_token_batch = np.stack(_pad_1d(x, max_mels_length, constant_values=1) for x in stop_tokens)
        stop_token_batch = np.expand_dims(stop_token_batch, axis=2)

        txt_batch = torch.LongTensor(txt_batch)
        mels_batch = torch.FloatTensor(mels_batch)
        mels_batch = mels_batch.permute(0, 2, 1)
        txt_lengths = torch.LongTensor(txt_lengths)
        mels_lengths = torch.LongTensor(mels_lengths)
        stop_token_batch = torch.FloatTensor(stop_token_batch)
        stop_token_batch = stop_token_batch.permute(0, 2, 1)

        return txt_batch, mels_batch, stop_token_batch, txt_lengths, mels_lengths


class SimilarTimeLengthSampler(Sampler):
    """Partially randomized sampler

    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batches
    """

    def __init__(self, lengths, descending=False, batch_size=16, batch_group_size=None, permutate=True):
        self.lengths, self.sorted_indices = torch.sort(torch.LongTensor(lengths), descending=descending)

        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.clone()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].view(-1, self.batch_size)[perm, :].view(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            random.shuffle(indices[s:])

        print(indices)
        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)


class DynamicalSimilarTimeLengthSampler(Sampler):
    """Partially randomized sampler

    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batches
    """

    def __init__(self, lengths, batch_size_min=16, batch_expand_level=4, batch_group=32, permutate=True):
        sorted_lengths, sorted_indices = torch.sort(torch.LongTensor(lengths))
        last_idx = len(lengths) - len(lengths) % batch_size_min
        batch_acc_size = sorted_lengths[last_idx] * batch_size_min
        s_list = []
        e_list = []
        s = 0
        e = 0
        batch_expand_level_ = batch_expand_level
        while e < last_idx:
            s_tmp = e + 1
            while batch_expand_level_ > 0:
                e_tmp = s + batch_size_min*batch_expand_level_ - 1
                length = sorted_lengths(e_tmp)
                if batch_expand_level_ * length > batch_acc_size:
                    batch_expand_level_ -= 1
                else:
                    e = e_tmp
                    s_list.append(s_tmp)
                    e_list.append(e_tmp)
                    break
        assert batch_expand_level_ > 0

        self.batch_group = batch_group
        self.sorted_indices = sorted_indices
        self.s_list = s_list
        self.e_list = e_list
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.clone()
        batch_group = self.batch_group
        for i in range(batch_group):
            s = self.s_list[i]
            e = self.e_list[i]
            random.shuffle(indices[s:e])

        self.shuffle_indices = indices

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(self.s_list))
            random.shuffle(perm)
            perm_s = [list(self.s_list[i], self.e_list[i]) for i in perm]

        return iter(perm_s)

    def __len__(self):
        return len(self.sorted_indices)

    def shuffle_indices(self):
        return self.shuffle_indices.clone()


class DynamicalBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
    """

    def __init__(self, sampler):
        if not isinstance(sampler, DynamicalSimilarTimeLengthSampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        self.sampler = sampler

    def __iter__(self):
        self.shuffle_indices = self.sampler.shuffle_indices
        batch = []
        for start, end in self.sampler:
            for idx in self.shuffle_indices[start, end]:
                batch.append(idx)
            yield batch
            batch = []

    def __len__(self):
        return len(self.sampler.s_list)

