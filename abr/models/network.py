"""Small candidate scorers. Outputs are uncalibrated scores, not trained Q-values.

The GRU re-encodes the finite history window per decision. No hidden state lives
across calls or episodes. A masked step cannot update its history representation.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .gru import MaskedGRUEncoder
from .features import register_scaling, scale_player, scale_candidates

FEATURE_VERSION = 1


@dataclass(frozen=True)
class ModelConfig:
    kind: str = 'gru'
    history_length: int = 8
    history_width: int = 48
    state_width: int = 32
    scorer_width: int = 32

    def __post_init__(self):
        if self.kind not in {'dense', 'gru'}:
            raise ValueError('kind must be dense or gru')
        for value in (self.history_length, self.history_width, self.state_width, self.scorer_width):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Model dimensions must be positive integers')


class CandidateScorer(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.kind == 'gru':
            self.history_encoder = MaskedGRUEncoder(4, c.history_width)
        else:
            self.history_encoder = nn.Sequential(nn.Linear(c.history_length * 5, c.history_width),
                nn.SiLU(), nn.Linear(c.history_width, c.history_width), nn.SiLU())
        self.state_encoder = nn.Sequential(nn.Linear(15, c.state_width), nn.SiLU())
        # Candidate-set mean, min, max lets future value depend on the available ladder.
        self.fusion = nn.Sequential(nn.Linear(c.history_width + c.state_width + 9, c.history_width), nn.SiLU())
        self.scorer = nn.Sequential(nn.Linear(c.history_width + 3, c.scorer_width),
                                    nn.SiLU(), nn.Linear(c.scorer_width, 1))
        # Fixed physical unit scaling. Nothing is fit to validation or test data.
        register_scaling(self)

    def forward(self, batch):
        history, player, candidates, mask = (batch[k] for k in ('history', 'player', 'candidates', 'candidate_mask'))
        c = self.config
        if (history.ndim != 3 or history.shape[1:] != (c.history_length, 5)
                or player.shape != (history.shape[0], 15)
                or candidates.ndim != 3 or candidates.shape[0] != history.shape[0] or candidates.shape[-1] != 3
                or mask.shape != candidates.shape[:2] or mask.dtype != torch.bool or candidates.shape[1] < 1):
            raise ValueError('Invalid model batch dimensions or candidate mask')
        if not bool(mask.any(dim=1).all()):
            raise ValueError('Each observation needs at least one available candidate')
        valid = history[..., 4:5]
        values = torch.log1p(history[..., :4] * valid)
        if c.kind == 'gru':
            encoded = self.history_encoder(values, valid)
        else:
            encoded = self.history_encoder(torch.cat((values, valid), dim=-1).flatten(1))
        state = self.state_encoder(scale_player(self, player))
        features = scale_candidates(self, candidates)
        valid_features = features.masked_fill(~mask[..., None], 0.)
        mean = valid_features.sum(1) / mask.sum(1, keepdim=True)
        minimum = features.masked_fill(~mask[..., None], float('inf')).amin(1)
        maximum = features.masked_fill(~mask[..., None], -float('inf')).amax(1)
        fused = self.fusion(torch.cat((encoded, state, mean, minimum, maximum), dim=-1))
        shared_state = fused[:, None, :].expand(-1, candidates.shape[1], -1)
        scores = self.scorer(torch.cat((shared_state, valid_features), dim=-1)).squeeze(-1)
        return scores.masked_fill(~mask, -float('inf'))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
