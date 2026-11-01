"""Evaluate finite-video BOLA on frozen Ghent test routes.

Run from the repository root with ``.venv/bin/python -m scripts.evaluate_bola_finite_ghent``.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from abr.bola import BolaBasicConfig, BolaBasicPolicy, BolaFinitePolicy
from abr.data import Dataset
from abr.env import QoEWeights, StreamConfig, StreamingEnv
from abr.models.checkpoint import load_model
from abr.models.policy import DecisionPolicy
from abr.models.safety import ActionSafety


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'experiments/configs/bola_finite_ghent_v1.json'
OUTPUT = ROOT / 'runs/bola-finite-ghent-v1/report.json'


def allowed_actions(observation, safety):
    candidates = observation['candidates']
    allowed = np.ones(len(candidates), dtype=bool)
    if safety is None:
        return allowed
    lowest = int(np.argmin(candidates[:, 0]))
    samples = observation['history'][observation['history'][:, -1] > 0, 0]
    allowed[:] = False
    if len(samples) and observation['state'][0] >= safety['min_buffer_s']:
        estimate = safety['throughput_factor'] * len(samples) / np.sum(1 / np.maximum(samples, 1e-12))
        allowed = candidates[:, 0] <= estimate
    allowed[lowest] = True
    return allowed


def reward_for_stats_delta(before, after, weights):
    terms = {
        'quality': after['quality_integral'] - before['quality_integral'],
        'startup': -weights.startup * (after['startup_s'] - before['startup_s']),
        'rebuffer': -weights.rebuffer * (after['rebuffer_s'] - before['rebuffer_s']),
        'switch': -weights.switch * (after['switch_log'] - before['switch_log']),
        'latency': -weights.latency * (after['latency_excess_integral'] - before['latency_excess_integral']),
        'rate_deviation': -weights.rate_deviation * (after['rate_deviation_s'] - before['rate_deviation_s']),
    }
    return float(sum(terms.values())), terms


def evaluate_policy(dataset, protocol, policy_name, basic, finite, dqn):
    config = StreamConfig(**protocol['stream_config'])
    weights = QoEWeights(**protocol['qoe_weights'])
    env = StreamingEnv(dataset, split=protocol['split'], config=config, weights=weights)
    safety = protocol['safety_filter']
    episodes = []
    for trace_index, trace in enumerate(dataset.splits[protocol['split']]):
        obs, _ = env.reset(seed=0, options={'trace_index': trace_index})
        total_reward = 0.0
        steps = []
        while True:
            wait_s = 0.0
            finite_details = None
            if policy_name in {'bola_basic', 'bola_basic_filtered'}:
                action = basic(obs, env.config)
                if policy_name.endswith('_filtered'):
                    scores = basic.scores(obs, env.config)
                    scores[~allowed_actions(obs, safety)] = -np.inf
                    action = int(np.argmax(scores))
            elif policy_name in {'bola_finite', 'bola_finite_filtered'}:
                elapsed_s = float(env.segment_starts[env.segment])
                remaining_s = env.total_s - elapsed_s
                action, wait_s, finite_details = finite.decision(
                    obs, env.config, elapsed_s=elapsed_s, remaining_s=remaining_s)
                if policy_name.endswith('_filtered'):
                    scores = np.asarray(finite_details['scores'])
                    scores[~allowed_actions(obs, safety)] = -np.inf
                    action = int(np.argmax(scores))
            elif policy_name == 'highest_allowed':
                allowed = allowed_actions(obs, safety)
                action = int(np.flatnonzero(allowed)[-1])
            elif policy_name == 'dqn_frozen':
                action = dqn(obs, env.config)
            else:
                raise ValueError(f'Unknown policy: {policy_name}')

            pause_terms = {}
            if wait_s > 0:
                before_pause = dict(env.player.stats)
                elapsed, wait_complete = env._wait(wait_s)
                pause_reward, pause_terms = reward_for_stats_delta(
                    before_pause, env.player.stats, env.weights)
                total_reward += pause_reward
                wait_s = elapsed
                if not wait_complete:
                    # The next request records the finite trace ending as an
                    # incomplete episode, using the normal environment path.
                    wait_s = elapsed

            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps.append({
                'action': int(action), 'wait_s': wait_s,
                'finite_state': finite_details,
                'pause_reward_terms': pause_terms,
                **info,
            })
            if terminated or truncated:
                break
        episodes.append({
            'trace': trace.name,
            'trace_sha256': trace.fingerprint,
            'source': trace.source,
            'source_group': trace.source_group,
            'trace_semantics': trace.semantics,
            'completed': bool(terminated),
            'termination_reason': info['termination_reason'],
            'reward': total_reward,
            'metrics': env.metrics(),
            'steps': steps,
        })
    for episode in episodes:
        episode['explicit_wait_s'] = float(sum(step['wait_s'] for step in episode['steps']))
        episode['policy_actions'] = [step['action'] for step in episode.pop('steps')]
    return episodes


def summarize(episodes):
    return {
        'sessions': len(episodes),
        'completed': sum(ep['completed'] for ep in episodes),
        'mean_reward': float(np.mean([ep['reward'] for ep in episodes])),
        'mean_bitrate_kbps_all_episodes': float(np.mean([
            ep['metrics']['mean_played_bitrate_kbps'] for ep in episodes])),
        'mean_startup_s': float(np.mean([ep['metrics']['startup_s'] for ep in episodes])),
        'mean_rebuffer_s': float(np.mean([ep['metrics']['rebuffer_s'] for ep in episodes])),
        'mean_explicit_wait_s': float(np.mean([ep['explicit_wait_s'] for ep in episodes])),
    }


def main():
    torch.set_num_threads(1)
    protocol = json.loads(PROTOCOL_PATH.read_text())
    dataset_path = ROOT / protocol['dataset']
    dataset = Dataset.load(dataset_path)
    basic_config = BolaBasicConfig(**protocol['bola'])
    basic = BolaBasicPolicy(config=basic_config)
    finite = BolaFinitePolicy(config=basic_config)
    checkpoint_path = ROOT / protocol['frozen_model']['checkpoint']
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_sha256 != protocol['frozen_model']['checkpoint_sha256']:
        raise ValueError('Frozen model checkpoint hash does not match protocol')
    model, _ = load_model(checkpoint_path)
    dqn = DecisionPolicy(model, safety=ActionSafety(**protocol['safety_filter']))
    reports = []
    for policy_name in protocol['policies']:
        episodes = evaluate_policy(dataset, protocol, policy_name, basic, finite, dqn)
        report = {
            'policy': policy_name,
            'episodes': episodes,
            'summary': summarize(episodes),
        }
        reports.append(report)
        print(json.dumps({'policy': policy_name, **report['summary'],
                          'traces': [{'name': ep['trace'], 'completed': ep['completed'],
                                      'startup_s': round(ep['metrics']['startup_s'], 3),
                                      'rebuffer_s': round(ep['metrics']['rebuffer_s'], 3),
                                      'mean_bitrate_kbps': round(ep['metrics']['mean_played_bitrate_kbps'], 1)}
                                     for ep in episodes]}), flush=True)

    payload = {
        'protocol_id': protocol['protocol_id'],
        'protocol_sha256': hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        'dataset': protocol['dataset'],
        'dataset_sha256': dataset.fingerprint,
        'implementation_sha256': {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('abr/bola.py', 'abr/env.py', 'abr/transport.py',
                         'abr/playback.py', 'scripts/evaluate_bola_finite_ghent.py')
        },
        'environment': {'stream_config': protocol['stream_config'],
                        'qoe_weights': protocol['qoe_weights'],
                        'safety_filter': protocol['safety_filter']},
        'frozen_model': {**protocol['frozen_model'], 'checkpoint_sha256_verified': checkpoint_sha256},
        'split': protocol['split'],
        'video_duration_s': float(dataset.video.durations_s.sum()),
        'policies': reports,
        'measurement_limits': protocol['measurement_limits'],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(OUTPUT), 'policies': len(reports)}))


if __name__ == '__main__':
    main()
