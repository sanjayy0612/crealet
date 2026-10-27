"""Architecture registration shared by checkpoints and CLI construction."""
from .network import CandidateScorer, ModelConfig
from .candidate_transformer import CandidateTransformer, CandidateTransformerConfig

MODEL_KINDS = ('dense', 'gru', 'candidate_transformer')


def build_model(config):
    if config.get('kind') == 'candidate_transformer':
        return CandidateTransformer(CandidateTransformerConfig(**config))
    return CandidateScorer(ModelConfig(**config))
