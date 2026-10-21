"""Versioned NPZ snapshots, with no pickle deserialization or generated confidence."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .network import CandidateScorer, ModelConfig, FEATURE_VERSION


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()


def save_initial_model(model, path, *, seed):
    """This construction milestone only produces explicitly UNTRAINED artifacts."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    metadata = {'format_version': 1, 'observation_version': 2, 'feature_version': FEATURE_VERSION,
                'model_config': asdict(model.config), 'training_status': 'untrained',
                'initialization_seed': seed, 'parameter_count': model.parameter_count,
                'model_source_sha256': source_hash(), 'torch_version': str(torch.__version__)}
    arrays = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    arrays['__metadata__'] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Open explicitly so NumPy does not silently change the requested file name.
    with target.open('wb') as file:
        np.savez_compressed(file, **arrays)
    return metadata


def load_model(path, *, device='cpu'):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive['__metadata__'].tobytes().decode())
        if (metadata.get('format_version') != 1 or metadata.get('observation_version') != 2
                or metadata.get('feature_version') != FEATURE_VERSION
                or metadata.get('training_status') != 'untrained'):
            raise ValueError('Unsupported checkpoint or feature contract')
        if metadata.get('model_source_sha256') != source_hash():
            raise ValueError('Model source differs from checkpoint; regenerate or explicitly migrate it')
        # Building a model must not consume the caller's CPU random state.
        with torch.random.fork_rng(devices=[]):
            model = CandidateScorer(ModelConfig(**metadata['model_config']))
        state = model.state_dict()
        if set(archive.files) != set(state) | {'__metadata__'}:
            raise ValueError('Checkpoint tensor keys do not match architecture')
        loaded = {}
        for name, expected in state.items():
            value = archive[name]
            if value.shape != tuple(expected.shape) or value.dtype != expected.numpy().dtype or not np.isfinite(value).all():
                raise ValueError(f'Invalid checkpoint tensor: {name}')
            loaded[name] = torch.from_numpy(value.copy())
        if metadata.get('parameter_count') != model.parameter_count:
            raise ValueError('Checkpoint parameter count mismatch')
        # Scaling constants are part of the feature contract, not trainable weights.
        for name in ('player_scale', 'player_log', 'candidate_scale'):
            if not torch.equal(loaded[name], state[name]):
                raise ValueError('Checkpoint changed fixed feature scaling')
        model.load_state_dict(loaded, strict=True)
    return model.to(device).eval(), metadata
