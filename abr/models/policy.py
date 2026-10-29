"""Single-pass, deterministic selection. Score magnitude is not confidence."""
import numpy as np
import torch
from .inputs import make_batch
from .safety import allowed_actions


class DecisionPolicy:
    def __init__(self, model, *, safety=None):
        self.model = model.eval()
        self.safety = safety

    @torch.no_grad()
    def scores(self, observation, stream_config):
        device = next(self.model.parameters()).device
        batch = make_batch([observation], [stream_config], self.model.config, device=device)
        output = self.model(batch)
        logits = output[0] if isinstance(output, tuple) else output
        scores = logits[0].cpu().numpy().copy()
        if not np.isfinite(scores).all():
            raise ValueError('Model produced non-finite candidate scores')
        return scores

    def __call__(self, observation, stream_config):
        scores = self.scores(observation, stream_config)
        return int(np.where(allowed_actions(observation, self.safety), scores, -np.inf).argmax())
