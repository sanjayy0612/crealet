"""Fixed physical-unit scaling shared by all decision architectures."""
import torch


def register_scaling(module):
    module.register_buffer('player_scale', torch.tensor([10.,10.,1.,1.,1., 1.,10.,10.,10.,10.,1.,1.,1.,1.,1.]))
    module.register_buffer('player_log', torch.tensor([1,1,0,0,1, 0,1,1,1,1,0,0,0,1,1], dtype=torch.bool))
    module.register_buffer('candidate_scale', torch.tensor([1.,1.,10.]))


def scale_player(module, player):
    scaled = player / module.player_scale
    return torch.where(module.player_log, torch.log1p(scaled), scaled)


def scale_candidates(module, candidates):
    return torch.log1p(candidates / module.candidate_scale)
