import torch.nn.functional as F
from torch import nn
import torch

import pdb

class LocationAttention(nn.Module):
    def __init__(self, dim_key, dim_query, dim_attention, num_location_features, attention_kernel_size=31,
                 smoothing=True, cumulate=True):
        super(LocationAttention, self).__init__()

        self.W = nn.Linear(in_features=dim_query, out_features=dim_attention)
        self.V = nn.Linear(in_features=dim_key, out_features=dim_attention, bias=False)
        self.U = nn.Linear(in_features=num_location_features, out_features=dim_attention, bias=False)
        self.conv_f = nn.Conv1d(in_channels=1, out_channels=num_location_features, kernel_size=attention_kernel_size,
                                padding=attention_kernel_size//2)
        self.w = nn.Linear(in_features=dim_attention, out_features=1, bias=False)
        self.score_mask_value = -float('inf')
        self.smoothing = smoothing
        self.cumulate = cumulate

    def init_state(self, memory, mask):
        """
        memory: T x B x Ck
        """
        self.memory = memory.permute(1, 0, 2).contiguous()  # [B x T x Ck]
        self.key_attention = self.V(self.memory) # [B x T x Ca]
        self.mask = mask.unsqueeze(dim=2)

    def score(self, query, probability):
        """
        :param key: encoder_output [T x B x Ck]
        :param query: decoder_input [1 x B x Cq]
        :param probability: probability [B x 1 x T]
        :return:
        """
        query_attention = self.W(query).permute(1, 0, 2) # [B x 1 x Ca]
        location_value = self.conv_f(probability)  # [B x Cl x T]
        location_value = location_value.permute(0, 2, 1)  # [B x T x Cl]
        location_attention = self.U(location_value)  # [B x T x Ca]
        return self.w(torch.tanh(self.key_attention + location_attention + query_attention))  # [B x T x 1]

    def softmax_smoothing(self, energies, dim=0):
        logistic_sigmoid = torch.sigmoid(energies)
        return logistic_sigmoid / torch.sum(logistic_sigmoid, dim=dim, keepdim=True)

    def forward(self, query_input, state):
        """
        :param query_input: 1 x B x Cq
        :param state:       B x 1 x T
        :return:
        """
        energies = self.score(query_input, state)  # [B x T x 1]
        if self.mask is not None:
            energies.data.masked_fill_(self.mask, self.score_mask_value)

        if self.smoothing:
            probability = self.softmax_smoothing(energies, dim=1)  # [B x T x 1]
        else:
            probability = torch.nn.functional.softmax(energies, dim=1)  # [B x T x 1]

        probability = probability.permute(0, 2, 1) # [B x 1 x T]

        if self.cumulate:
            next_state = probability + state  # [B x 1 x T]
        else:
            next_state = probability

        context = torch.bmm(probability, self.memory)  # [B x 1 x Ck]
        context_tbc = context.permute(1, 0, 2)         # [1 x B x Ck]

        return context_tbc, probability, next_state

