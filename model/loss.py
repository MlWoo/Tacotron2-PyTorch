import torch
import torch.nn as nn


def sequence_mask(sequence_length, pos=True, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand)


class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()

    def forward(self, inputs, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, 1, T)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(dim=1).float()

        # (B, C, T)
        mask_ = mask.expand_as(target)
        mask_.requires_grad = False
        losses = nn.functional.binary_cross_entropy_with_logits(inputs, target, reduce=False, pos_weight=20)
        return ((losses * mask_).sum()) / mask_.sum()


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduce=False)

    def forward(self, inputs, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, 1, T)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(dim=1).float()

        # (B, C, T)
        mask_ = mask.expand_as(target)
        mask_.requires_grad = False
        losses = self.criterion(inputs, target)
        return ((losses * mask_).sum()) / mask_.sum()
