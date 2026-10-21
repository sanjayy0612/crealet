"""Single-pass, deterministic selection. Score magnitude is not confidence."""
import numpy as np
import torch
from .inputs import make_batch


class DecisionPolicy:
    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def scores(self, observation, stream_config):
        device = next(self.model.parameters()).device
        batch = make_batch([observation], [stream_config], self.model.config, device=device)
        scores = self.model(batch)[0].cpu().numpy().copy()
        if not np.isfinite(scores).all():
            raise ValueError('Model produced non-finite candidate scores')
        return scores

    def __call__(self, observation, stream_config):
        return int(self.scores(observation, stream_config).argmax())
