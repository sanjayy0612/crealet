"""Architecture registration shared by checkpoints and CLI construction."""
from .network import CandidateScorer, ModelConfig
from .candidate_transformer import CandidateTransformer, CandidateTransformerConfig
from .dueling_dqn import DuelingDoubleDQN, DuelingDoubleDQNConfig

MODEL_KINDS = ('dense', 'gru', 'candidate_transformer', 'dueling_double_dqn')


def build_model(config):
    if config.get('kind') == 'candidate_transformer':
        return CandidateTransformer(CandidateTransformerConfig(**config))
    if config.get('kind') == 'dueling_double_dqn':
        return DuelingDoubleDQN(DuelingDoubleDQNConfig(**config))
    return CandidateScorer(ModelConfig(**config))
