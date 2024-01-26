# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.linear1 = nn.Linear(dim, 2*dim)
        self.linear2 = nn.Linear(dim*2, dim)
        self.linear3 = nn.Linear(dim, 1)

    def forward(self, data):
        out1 = F.relu(self.linear1(data))
        out2 = self.linear2(out1)
        out3 = self.linear3(out2)
        out = F.softplus(out3)

        return out


class SpatiotemporalModel(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, event_times, spatial_events, input_mask, t0, t1):
        pass


class CombinedSpatiotemporalModel(SpatiotemporalModel):

    def __init__(self, temporal_model, encoder_model=None):
        super().__init__()
        self.encoder = encoder_model
        self.temporal_model = temporal_model
        self.pred = Predictor(dim=1+self.temporal_model.hdim)

    def forward(self, event_times, spatial_events, input_mask, t0, t1):
        if self.encoder:
            spatial_events = self.encoder(spatial_events, event_times, input_mask)
        time_emb, lamda, loglik = self._temporal_logprob(event_times, spatial_events, input_mask, t0, t1)
        pre_label = self.pred(torch.cat([time_emb, lamda.unsqueeze(-1)], dim=1))
        return pre_label, loglik

    def _temporal_logprob(self, event_times, spatial_events, input_mask, t0, t1):
        return self.temporal_model.logprob(event_times, spatial_events, input_mask, t0, t1)


class RNN_layers(nn.Module):

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.GRU(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1)

        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        order_seq_lengths = torch.index_select(lengths, dim=0, index=idx_sort).cpu()
        order_tensor_in = torch.index_select(data, dim=0, index=idx_sort).cpu()

        x_packed = nn.utils.rnn.pack_padded_sequence(order_tensor_in, order_seq_lengths, batch_first=True).to(data)
        temp = self.rnn(x_packed)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        # unsort output to original order
        y = torch.index_select(out, dim=0, index=idx_unsort).to(data)

        y = y[-1].unsqueeze(1)
        out = self.projection(y)
        return out


def zero_diffeq(t, h):
    return torch.zeros_like(h)


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)

