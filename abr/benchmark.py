"""Environment-only replay. No learned policy or training dependencies.

python -m abr.benchmark --dataset ... --media ... --mode live --output ...
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
from .data import Dataset
from .media import Media
from .env import StreamingEnv, StreamConfig


def evaluate(env, *, baseline='lowest', seed=0, policy=None, policy_metadata=None):
    if baseline not in {'lowest', 'highest', 'throughput', 'model'}:
        raise ValueError("Unknown diagnostic baseline")
    if baseline == 'model':
        if not callable(policy) or not isinstance(policy_metadata, dict):
            raise ValueError("Model replay requires a policy and explicit artifact metadata")
    elif policy is not None or policy_metadata is not None:
        raise ValueError("Policy supplied for a diagnostic baseline")
    episodes = []
    for index, trace in enumerate(env.dataset.splits[env.split]):
        obs, _ = env.reset(seed=seed, options={'trace_index': index})
        steps, total_reward = [], 0.
        while True:
            action = env.action_space.n - 1 if baseline == 'highest' else 0
            if baseline == 'throughput':
                samples = obs['history'][obs['history'][:, -1] > 0, 0]
                if len(samples):
                    # Diagnostic harmonic goodput heuristic, not BOLA/MPC/LoL+.
                    estimate = 0.85 * len(samples) / np.sum(1 / np.maximum(samples, 1e-12))
                    eligible = np.flatnonzero(obs['candidates'][:, 0] <= estimate)
                    action = int(eligible[-1]) if len(eligible) else 0
            if baseline == 'model':
                action = policy(obs, env.config)
                if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
                    raise ValueError("Policy must return an integer candidate index")
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps.append({'action': action, **info})
            if terminated or truncated:
                break
        episodes.append({'trace': trace.name, 'trace_sha256': trace.fingerprint,
                         'source': trace.source, 'source_group': trace.source_group,
                         'trace_semantics': trace.semantics, 'completed': terminated,
                         'termination_reason': info['termination_reason'], 'reward': total_reward,
                         'metrics': env.metrics(), 'steps': steps})
    digest = hashlib.sha256()
    for name in ('data.py', 'media.py', 'transport.py', 'playback.py', 'env.py', 'trace_import.py', 'benchmark.py'):
        path = Path(__file__).with_name(name)
        digest.update(name.encode() + path.read_bytes())
    return {'provenance': {**env.provenance(), 'source_sha256': digest.hexdigest(),
                           'python': platform.python_version(), 'seed': seed, 'split': env.split,
                           'baseline': baseline, 'history_length': env.history_length,
                           'policy_metadata': policy_metadata,
                           'versions': {x: importlib.metadata.version(x) for x in ('numpy', 'gymnasium')}},
            'episodes': episodes,
            'completion_rate': float(np.mean([x['completed'] for x in episodes]))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--media', help='Explicit part-size/timing JSON; required for live mode')
    p.add_argument('--mode', choices=['vod', 'live'], default='vod')
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    p.add_argument('--baseline', choices=['lowest', 'highest', 'throughput'], default='lowest')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--config', help='JSON StreamConfig overrides; mode comes from --mode')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text()) if args.config else {}
    cfg['mode'] = args.mode
    env = StreamingEnv(Dataset.load(args.dataset), media=Media.load(args.media) if args.media else None,
                       split=args.split, config=StreamConfig(**cfg))
    result = evaluate(env, baseline=args.baseline, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'episodes': len(result['episodes']), 'completion_rate': result['completion_rate'],
                      'output': str(out)}))


if __name__ == '__main__':
    main()
