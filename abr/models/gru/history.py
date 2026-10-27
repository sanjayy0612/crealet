"""Finite-window recurrent baseline, with no state retained between decisions."""
import torch
from torch import nn


class MaskedGRUEncoder(nn.GRUCell):
    """Keep GRUCell parameter names for the existing baseline interface."""

    def forward(self, values, valid):
        encoded = values.new_zeros((len(values), self.hidden_size))
        for t in range(values.shape[1]):
            updated = super().forward(values[:, t], encoded)
            encoded = torch.where(valid[:, t].bool(), updated, encoded)
        return encoded
