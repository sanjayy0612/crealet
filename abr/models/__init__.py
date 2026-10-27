"""Optional PyTorch decision-model components. Environment-only imports stay light."""
from .network import CandidateScorer, ModelConfig
from .candidate_transformer import CandidateTransformer, CandidateTransformerConfig
from .inputs import make_batch
from .checkpoint import load_model, save_initial_model

__all__ = ['CandidateScorer', 'ModelConfig', 'CandidateTransformer', 'CandidateTransformerConfig',
           'make_batch', 'load_model', 'save_initial_model']
