"""Compare BOLA-Basic with and without the existing safety mask on frozen VOD cases.

Run from the repository root with ``.venv/bin/python -m scripts.evaluate_bola_transfer``.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from abr.bola import BolaBasicConfig, BolaBasicPolicy
from abr.data import Dataset
from abr.benchmark import evaluate
from abr.env import QoEWeights, StreamConfig, StreamingEnv
from abr.recovery import RecoveryConfig


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'experiments/configs/single_task_vod_v1.json'
OUTPUT = ROOT / 'runs/single-task-vod-v1/bola-transfer.json'


def allowed_actions(observation, safety):
    """Pure NumPy copy of abr.models.safety.allowed_actions for this baseline."""
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


def make_env(dataset, protocol):
    return StreamingEnv(dataset, split=protocol['split'],
                        config=StreamConfig(**protocol['stream_config']),
                        recovery=RecoveryConfig(**protocol['request_recovery']),
                        weights=QoEWeights(rebuffer=12))


def compact(report):
    for episode in report['episodes']:
        steps = episode.pop('steps')
        episode['policy_actions'] = [step['action'] for step in steps]
        episode['effective_actions'] = [step.get('effective_action', step['action']) for step in steps]
    return report


def main():
    protocol = json.loads(PROTOCOL.read_text())
    bola_config = BolaBasicConfig(segment_duration_s=4.0, gamma_p=5.0)
    safety = protocol['safety_filter']
    bola = BolaBasicPolicy(config=bola_config)

    def filtered_bola(observation, stream_config):
        scores = bola.scores(observation, stream_config)
        allowed = allowed_actions(observation, safety)
        scores[~allowed] = -np.inf
        return int(np.argmax(scores))

    policies = {
        'bola_basic': (bola, None),
        'bola_basic_filtered': (filtered_bola, safety),
    }
    reports = []
    for media in protocol['media']:
        dataset = Dataset.load(ROOT / media['dataset'])
        for name, (policy, action_safety) in policies.items():
            env = make_env(dataset, protocol)
            result = compact(evaluate(
                env, baseline='model', policy=policy,
                policy_metadata={
                    'kind': name,
                    'algorithm': 'BOLA-Basic score with logarithmic bitrate utility',
                    'config': {'segment_duration_s': bola_config.segment_duration_s,
                               'gamma_p': bola_config.gamma_p,
                               'safety_filter': action_safety},
                }))
            result.update(content=media['name'], policy=name)
            reports.append(result)
            print(json.dumps({
                'content': media['name'], 'policy': name,
                'completed': sum(episode['completed'] for episode in result['episodes']),
                'sessions': len(result['episodes']),
                'mean_reward': float(np.mean([episode['reward'] for episode in result['episodes']])),
                'mean_bitrate_kbps': float(np.mean([
                    episode['metrics']['mean_played_bitrate_kbps'] for episode in result['episodes']])),
                'stalls_s': [round(episode['metrics']['rebuffer_s'], 3) for episode in result['episodes']],
            }), flush=True)

    payload = {
        'protocol_id': protocol['protocol_id'],
        'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        'algorithm': 'BOLA-Basic',
        'algorithm_parameters': {'segment_duration_s': bola_config.segment_duration_s,
                                 'gamma_p': bola_config.gamma_p,
                                 'utility': 'ln(candidate bitrate / minimum ladder bitrate)',
                                 'V': '(max_buffer_s / segment_duration_s - 1) / (max utility + gamma_p)',
                                 'segment_size': 'observation-v2 nominal candidate size estimate',
                                 'no_download_action': 'not representable; choose the highest-scoring available rung'},
        'reports': reports,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(OUTPUT), 'reports': len(reports)}))


if __name__ == '__main__':
    main()
