"""Dueling Q-value head over the numeric CandidateTransformer encoder."""
from dataclasses import dataclass

import torch

from .candidate_transformer import CandidateTransformer, CandidateTransformerConfig
from .features import scale_candidates, scale_player


@dataclass(frozen=True)
class DuelingDoubleDQNConfig:
    kind: str = 'dueling_double_dqn'
    history_length: int = 8
    hidden_size: int = 64
    layers: int = 2
    attention_heads: int = 4
    feedforward_size: int = 128
    dropout: float = 0.0

    def __post_init__(self):
        if self.kind != 'dueling_double_dqn':
            raise ValueError('Expected dueling_double_dqn kind')
        CandidateTransformerConfig(
            history_length=self.history_length, hidden_size=self.hidden_size,
            layers=self.layers, attention_heads=self.attention_heads,
            feedforward_size=self.feedforward_size, dropout=self.dropout)


class DuelingDoubleDQN(CandidateTransformer):
    """Q(s,a)=V(s)+A(s,a)-mean_valid(A); returns one Q-value per bitrate.

    Double-DQN target-network behavior is implemented by the trainer. This module
    supplies the dueling action-value model and can be loaded with the normal
    checkpoint/policy path.
    """

    def __init__(self, config: DuelingDoubleDQNConfig | None = None):
        config = config or DuelingDoubleDQNConfig()
        base_config = CandidateTransformerConfig(
            history_length=config.history_length,
            hidden_size=config.hidden_size,
            layers=config.layers,
            attention_heads=config.attention_heads,
            feedforward_size=config.feedforward_size,
            dropout=config.dropout)
        super().__init__(base_config)
        # The former policy head is the per-action advantage stream.
        self.advantage_head = self.policy_head
        del self.policy_head
        self.config = config

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
            raise ValueError('Invalid dueling DQN batch dimensions or masks')
        if not bool(mask.any(dim=1).all()):
            raise ValueError('Each observation needs at least one available candidate')

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

        candidate_tokens = tokens[:, c.history_length + 1:]
        state_token = tokens[:, c.history_length]
        advantages = self.advantage_head(candidate_tokens).squeeze(-1)
        value = self.value_head(state_token).squeeze(-1)
        valid_advantages = advantages.masked_fill(~mask, 0.)
        mean_advantage = valid_advantages.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
        q_values = value[:, None] + advantages - mean_advantage
        return q_values.masked_fill(~mask, -float('inf'))
