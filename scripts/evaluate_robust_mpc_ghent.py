"""Compare frozen RobustMPC, DQN and highest-allowed on Ghent test traces.

Run with ``.venv/bin/python -m scripts.evaluate_robust_mpc_ghent`` from the
repository root. This is diagnostic because these two test traces were used in
the earlier BOLA screen; no RobustMPC settings are selected from their results.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from abr.data import Dataset
from abr.env import QoEWeights, StreamConfig, StreamingEnv
from abr.models.checkpoint import load_model
from abr.models.policy import DecisionPolicy
from abr.models.safety import ActionSafety, allowed_actions
from abr.robust_mpc import RobustMPCConfig, RobustMPCPolicy


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'experiments/configs/robust_mpc_ghent_v1.json'
OUTPUT = ROOT / 'runs/robust-mpc-ghent-v1/report.json'


def evaluate_one(dataset, protocol, name, model_policy, mpc_policy):
    config = StreamConfig(**protocol['stream_config'])
    weights = QoEWeights(**protocol['qoe_weights'])
    safety = protocol['safety_filter']
    env = StreamingEnv(dataset, split=protocol['split'], config=config, weights=weights)
    reports = []
    for trace_index, trace in enumerate(dataset.splits[protocol['split']]):
        observation, _ = env.reset(seed=0, options={'trace_index': trace_index})
        total_reward = 0.0
        latencies_ms, actions, planner_diagnostics = [], [], []
        while True:
            started = time.perf_counter_ns()
            if name == 'robust_mpc':
                action, diagnostic = mpc_policy.plan(
                    observation, env.config, env.weights, segment_index=env.segment)
                planner_diagnostics.append(diagnostic)
            elif name == 'dqn_frozen':
                action = model_policy(observation, env.config)
            elif name == 'highest_allowed':
                mask = allowed_actions(observation, ActionSafety(**safety))
                action = int(np.flatnonzero(mask)[-1])
            else:
                raise ValueError(f'Unknown policy {name}')
            latencies_ms.append((time.perf_counter_ns() - started) / 1e6)
            actions.append(int(action))
            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            if terminated or truncated:
                break
        metrics = env.metrics()
        reports.append({
            'trace': trace.name,
            'trace_sha256': trace.fingerprint,
            'source_group': trace.source_group,
            'completed': bool(terminated),
            'termination_reason': info['termination_reason'],
            'reward': float(total_reward),
            'metrics': metrics,
            'actions': actions,
            'decision_latency_ms': {
                'median': float(np.median(latencies_ms)),
                'p95': float(np.percentile(latencies_ms, 95)),
                'max': float(np.max(latencies_ms)),
                'decisions': len(latencies_ms),
            },
            'planner_diagnostics': planner_diagnostics if name == 'robust_mpc' else None,
        })
    return reports


def summarize(episodes):
    return {
        'sessions': len(episodes),
        'completed': sum(episode['completed'] for episode in episodes),
        'mean_reward': float(np.mean([episode['reward'] for episode in episodes])),
        'mean_played_bitrate_kbps': float(np.mean([
            episode['metrics']['mean_played_bitrate_kbps'] for episode in episodes])),
        'mean_startup_s': float(np.mean([episode['metrics']['startup_s'] for episode in episodes])),
        'mean_rebuffer_s': float(np.mean([episode['metrics']['rebuffer_s'] for episode in episodes])),
        'decision_latency_p50_ms': float(np.median([
            episode['decision_latency_ms']['median'] for episode in episodes])),
        'decision_latency_p95_ms': float(np.max([
            episode['decision_latency_ms']['p95'] for episode in episodes])),
    }


def main():
    torch.set_num_threads(1)
    protocol = json.loads(PROTOCOL_PATH.read_text())
    dataset = Dataset.load(ROOT / protocol['dataset'])
    checkpoint = ROOT / protocol['frozen_model']['checkpoint']
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != protocol['frozen_model']['checkpoint_sha256']:
        raise ValueError('DQN checkpoint hash does not match the frozen protocol')
    model, _ = load_model(checkpoint)
    safety = ActionSafety(**protocol['safety_filter'])
    dqn = DecisionPolicy(model, safety=safety)
    mpc_config = RobustMPCConfig(**protocol['robust_mpc'])
    mpc = RobustMPCPolicy(dataset.video.durations_s, config=mpc_config,
                          safety=protocol['safety_filter'])

    policies = {'robust_mpc': (None, mpc), 'dqn_frozen': (dqn, None),
                'highest_allowed': (None, None)}
    results = []
    for name, (model_policy, mpc_policy) in policies.items():
        episodes = evaluate_one(dataset, protocol, name, model_policy, mpc_policy)
        result = {'policy': name, 'episodes': episodes, 'summary': summarize(episodes)}
        results.append(result)
        print(json.dumps({'policy': name, **result['summary'],
                          'traces': [{'name': episode['trace'],
                                      'reward': round(episode['reward'], 3),
                                      'completed': episode['completed'],
                                      'rebuffer_s': round(episode['metrics']['rebuffer_s'], 3),
                                      'bitrate_kbps': round(episode['metrics']['mean_played_bitrate_kbps'], 1),
                                      'latency_p95_ms': round(episode['decision_latency_ms']['p95'], 3)}
                                     for episode in episodes]}), flush=True)

    payload = {
        'protocol_id': protocol['protocol_id'],
        'protocol_sha256': hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        'dataset': protocol['dataset'],
        'dataset_sha256': dataset.fingerprint,
        'split': protocol['split'],
        'frozen_model': {**protocol['frozen_model'], 'checkpoint_sha256_verified': checkpoint_hash},
        'robust_mpc': protocol['robust_mpc'],
        'implementation_sha256': {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in ('abr/robust_mpc.py', 'abr/env.py', 'abr/transport.py',
                         'abr/playback.py', 'scripts/evaluate_robust_mpc_ghent.py')
        },
        'stream_config': protocol['stream_config'],
        'qoe_weights': protocol['qoe_weights'],
        'safety_filter': protocol['safety_filter'],
        'policies': results,
        'limits': protocol['limits'],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(OUTPUT), 'policies': len(results)}))


if __name__ == '__main__':
    main()
