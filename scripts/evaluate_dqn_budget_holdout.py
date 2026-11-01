"""Evaluate frozen 120/240-episode DQNs on the frozen Ghent session holdout."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from abr.data import Dataset
from abr.env import QoEWeights
from abr.models.checkpoint import load_model
from abr.models.policy import DecisionPolicy
from abr.models.safety import ActionSafety, allowed_actions
from abr.robust_mpc import RobustMPCConfig, RobustMPCPolicy
from abr.startup_training import make_env


ROOT = Path(__file__).resolve().parents[1]
TRAINING_PROTOCOL = ROOT / 'experiments/configs/startup_ladder_240.json'
BUDGET_PROTOCOL = ROOT / 'experiments/configs/dqn_training_budget_v1.json'
OUTPUT = ROOT / 'runs/startup-ladder-budget-240/holdout.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_policy(dataset, protocol, name, model_policy=None, mpc_policy=None):
    safety = ActionSafety(**protocol['safety'])
    env = make_env(dataset, 'test', protocol)
    episodes = []
    for trace_index, trace in enumerate(dataset.splits['test']):
        obs, _ = env.reset(seed=0, options={'trace_index': trace_index})
        total_reward = 0.0
        policy_actions, effective_actions, latency_ms = [], [], []
        planner_diagnostics = []
        while True:
            before = time.perf_counter_ns()
            if name == 'robust_mpc':
                action, details = mpc_policy.plan(obs, env.config, env.weights,
                                                  segment_index=env.segment)
                planner_diagnostics.append(details)
            elif name == 'highest_allowed_ghent':
                mask = allowed_actions(obs, safety)
                action = int(np.flatnonzero(mask)[-1])
            else:
                action = int(model_policy(obs, env.config))
            latency_ms.append((time.perf_counter_ns() - before) / 1e6)
            policy_actions.append(int(action))
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            if info['segment_complete']:
                effective_actions.append(int(info.get('effective_action', action)))
            if terminated or truncated:
                break
        metrics = env.metrics()
        episodes.append({
            'trace': trace.name,
            'trace_sha256': trace.fingerprint,
            'source_group': trace.source_group,
            'source': trace.source,
            'completed': bool(terminated),
            'termination_reason': info['termination_reason'],
            'reward': float(total_reward),
            'metrics': metrics,
            'policy_actions': policy_actions,
            'effective_actions': effective_actions,
            'decision_latency_ms': {
                'p50': float(np.percentile(latency_ms, 50)),
                'p95': float(np.percentile(latency_ms, 95)),
                'max': float(np.max(latency_ms)),
                'decisions': len(latency_ms),
            },
            'planner_diagnostics': planner_diagnostics if name == 'robust_mpc' else None,
        })
    return {'content': 'envivio', 'policy': name, 'episodes': episodes,
            'summary': {
                'sessions': len(episodes),
                'completed': sum(ep['completed'] for ep in episodes),
                'mean_reward': float(np.mean([ep['reward'] for ep in episodes])),
                'mean_bitrate_kbps': float(np.mean([
                    ep['metrics']['mean_played_bitrate_kbps'] for ep in episodes])),
                'mean_startup_s': float(np.mean([ep['metrics']['startup_s'] for ep in episodes])),
                'mean_rebuffer_s': float(np.mean([ep['metrics']['rebuffer_s'] for ep in episodes])),
                'decision_latency_p50_ms': float(np.median([
                    ep['decision_latency_ms']['p50'] for ep in episodes])),
                'decision_latency_p95_ms': float(np.max([
                    ep['decision_latency_ms']['p95'] for ep in episodes])),
            }}


def main():
    torch.set_num_threads(1)
    protocol = json.loads(TRAINING_PROTOCOL.read_text())
    budget_protocol = json.loads(BUDGET_PROTOCOL.read_text())
    dataset_path = ROOT / budget_protocol['fresh_test_requirement']['dataset']
    dataset = Dataset.load(dataset_path)
    expected = budget_protocol['fresh_test_requirement']['dataset_sha256']
    if dataset.fingerprint != expected:
        raise ValueError('Frozen fresh-test dataset fingerprint mismatch')
    if len(dataset.splits['test']) != budget_protocol['fresh_test_requirement']['sessions']:
        raise ValueError('Frozen test-session count mismatch')

    safety = ActionSafety(**protocol['safety'])
    reports = []
    checkpoint_records = []
    frozen = protocol['frozen_checkpoints']
    for episodes in (120, 240):
        root = ROOT / ('runs/startup-ladder' if episodes == 120 else 'runs/startup-ladder-budget-240')
        for seed in (7, 11, 13):
            path = root / f'seed-{seed}/accepted.npz'
            if not path.is_file():
                raise FileNotFoundError(f'Missing frozen seed checkpoint: {path}')
            checkpoint_hash = sha256(path)
            expected_hash = (frozen['120_episode_reference'][f'seed_{seed}'] if episodes == 120
                             else frozen['240_episode_selected'][f'seed_{seed}']['sha256'])
            if checkpoint_hash != expected_hash:
                raise ValueError(f'Frozen checkpoint hash mismatch for {episodes}-episode seed {seed}')
            model, metadata = load_model(path)
            if episodes == 240 and metadata['training']['episode'] != frozen['240_episode_selected'][f'seed_{seed}']['selected_episode']:
                raise ValueError(f'Frozen checkpoint selection episode mismatch for seed {seed}')
            policy = DecisionPolicy(model, safety=safety)
            name = f'dqn_{episodes}_seed_{seed}'
            report = evaluate_policy(dataset, protocol, name, model_policy=policy)
            report['training_episodes'] = episodes
            report['seed'] = seed
            reports.append(report)
            checkpoint_records.append({'episodes': episodes, 'seed': seed,
                                       'checkpoint': str(path.relative_to(ROOT)),
                                       'checkpoint_sha256': checkpoint_hash,
                                       'selected_training_episode': metadata['training']['episode']})
            print(json.dumps({'policy': name, **report['summary']}), flush=True)

    env = make_env(dataset, 'test', protocol)
    rule = evaluate_policy(dataset, protocol, 'highest_allowed_ghent')
    reports.append(rule)
    print(json.dumps({'policy': rule['policy'], **rule['summary']}), flush=True)

    mpc = RobustMPCPolicy(dataset.video.durations_s,
                          config=RobustMPCConfig(horizon_chunks=5,
                                                throughput_window=5,
                                                error_window=5,
                                                rtt_s=0.08),
                          safety=protocol['safety'])
    robust = evaluate_policy(dataset, protocol, 'robust_mpc', mpc_policy=mpc)
    reports.append(robust)
    print(json.dumps({'policy': robust['policy'], **robust['summary']}), flush=True)

    payload = {
        'protocol_id': budget_protocol['protocol_id'],
        'training_protocol_sha256': sha256(TRAINING_PROTOCOL),
        'budget_protocol_sha256': sha256(BUDGET_PROTOCOL),
        'test_dataset': str(dataset_path.relative_to(ROOT)),
        'test_dataset_sha256': dataset.fingerprint,
        'test_trace_count': len(dataset.splits['test']),
        'checkpoint_records': checkpoint_records,
        'stream_config': protocol['config'],
        'recovery': protocol['recovery'],
        'qoe_weights': {'rebuffer': 12.0, 'startup': 1.0, 'switch': 1.0,
                        'latency': 0.1, 'rate_deviation': 1.0},
        'safety': protocol['safety'],
        'implementation_sha256': {
            name: sha256(ROOT / name)
            for name in ('abr/robust_mpc.py', 'abr/startup_training.py',
                         'scripts/evaluate_dqn_budget_holdout.py')
        },
        'policies': reports,
        'limitations': [
            'The 32 test sessions are new sessions from the independent Ghent collection; route metadata only identifies transport-mode groups, so do not claim every geographic path is unique.',
            'Ghent throughput is historical observed HTTP goodput and the dataset assumes 80 ms RTT.',
            'The DQN was trained on Norway HSDPA traces and the Envivio seven-rung media ladder; test actions use this exact seven-rung ladder.',
            'The primary log-bitrate reward is a proxy. Run VMAF over completed-segment effective_actions using the aligned Envivio media before making a quality claim.',
            'These results are a trace-driven application-level replay, not a browser or current-network result.'
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(OUTPUT), 'policies': len(reports),
                      'traces_per_policy': len(dataset.splits['test'])}), flush=True)


if __name__ == '__main__':
    main()
