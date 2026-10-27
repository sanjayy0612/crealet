"""Legacy GRU history encoder; shared scoring remains in models.network."""
from .history import MaskedGRUEncoder

__all__ = ['MaskedGRUEncoder']
