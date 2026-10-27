"""ABR-CandidateTransformer v1: joint numeric attention, policy and value readouts.

History has temporal positions; candidates deliberately have no position or slot
embeddings. The contextualized state token supplies the actor-critic value head.
No hidden state is retained between calls. Outputs start untrained.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .features import register_scaling, scale_player, scale_candidates


@dataclass(frozen=True)
class CandidateTransformerConfig:
    kind: str = 'candidate_transformer'
    history_length: int = 8
    hidden_size: int = 64
    layers: int = 2
    attention_heads: int = 4
    feedforward_size: int = 128
    dropout: float = 0.0

    def __post_init__(self):
        if self.kind != 'candidate_transformer':
            raise ValueError('Expected candidate_transformer kind')
        for value in (self.history_length, self.hidden_size, self.layers,
                      self.attention_heads, self.feedforward_size):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Transformer dimensions must be positive integers')
        if self.hidden_size % self.attention_heads:
            raise ValueError('Hidden size must be divisible by attention heads')
        if isinstance(self.dropout, bool) or not 0 <= self.dropout < 1:
            raise ValueError('Dropout must be in [0, 1)')


class CandidateTransformer(nn.Module):
    def __init__(self, config: CandidateTransformerConfig | None = None):
        super().__init__()
        self.config = config or CandidateTransformerConfig()
        c = self.config
        self.history_projection = nn.Linear(5, c.hidden_size)
        self.state_projection = nn.Linear(15, c.hidden_size)
        self.candidate_projection = nn.Linear(3, c.hidden_size)
        self.history_positions = nn.Parameter(torch.empty(c.history_length, c.hidden_size))
        nn.init.normal_(self.history_positions, std=0.02)
        # Construct independently: nn.TransformerEncoder clones an identical
        # initial layer, whereas each layer here gets its own random initialization.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(c.hidden_size, c.attention_heads,
                dim_feedforward=c.feedforward_size, dropout=c.dropout,
                activation='gelu', batch_first=True, norm_first=True)
            for _ in range(c.layers)
        ])
        def head():
            return nn.Sequential(nn.LayerNorm(c.hidden_size),
                nn.Linear(c.hidden_size, c.hidden_size), nn.GELU(), nn.Linear(c.hidden_size, 1))
        self.policy_head = head()
        self.value_head = head()
        register_scaling(self)

    def forward(self, batch):
        history, player, candidates, mask, history_mask = (
            batch[k] for k in ('history', 'player', 'candidates', 'candidate_mask', 'history_mask'))
        c = self.config
        if (history.ndim != 3 or history.shape[1:] != (c.history_length, 5)
                or player.shape != (history.shape[0], 15)
                or candidates.ndim != 3 or candidates.shape[0] != history.shape[0]
                or candidates.shape[-1] != 3 or candidates.shape[1] < 1
                or mask.shape != candidates.shape[:2] or mask.dtype != torch.bool
                or history_mask.shape != history.shape[:2] or history_mask.dtype != torch.bool):
            raise ValueError('Invalid transformer batch dimensions or masks')
        if not bool(mask.any(dim=1).all()):
            raise ValueError('Each observation needs at least one available candidate')
        # Sanitize excluded rows before projection, not just after attention.
        h = history.masked_fill(~history_mask[..., None], 0.)
        h = torch.cat((torch.log1p(h[..., :4]), h[..., 4:5]), dim=-1)
        h = self.history_projection(h) + self.history_positions
        state = self.state_projection(scale_player(self, player))[:, None, :]
        candidate_values = candidates.masked_fill(~mask[..., None], 0.)
        candidate_tokens = self.candidate_projection(scale_candidates(self, candidate_values))
        tokens = torch.cat((h, state, candidate_tokens), dim=1)
        valid = torch.cat((history_mask, torch.ones_like(mask[:, :1]), mask), dim=1)
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=~valid)
        logits = self.policy_head(tokens[:, c.history_length + 1:]).squeeze(-1)
        value = self.value_head(tokens[:, c.history_length]).squeeze(-1)
        return logits.masked_fill(~mask, -float('inf')), value

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
