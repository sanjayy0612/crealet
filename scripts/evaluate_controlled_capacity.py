"""Compare frozen DQNs, the safety-filtered rule, and RobustMPC on controlled traces."""
import hashlib
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from abr.data import Dataset
from abr.env import StreamConfig, QoEWeights, StreamingEnv
from abr.models.checkpoint import load_model
from abr.models.policy import DecisionPolicy
from abr.models.safety import ActionSafety, allowed_actions
from abr.quality import load_quality_utility
from abr.recovery import RecoveryConfig
from abr.robust_mpc import RobustMPCConfig, RobustMPCPolicy


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'experiments/configs/controlled_capacity_transfer_v1.json'
OUTPUT = ROOT / 'runs/controlled-capacity-transfer-v1/results.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(dataset, content, protocol, policy_name, *, model_policy=None,
             mpc_policy=None, policy_kind=None, split='test'):
    utility_spec = protocol.get('quality_utility_by_content', {}).get(content)
    utility = (None if utility_spec is None else
               load_quality_utility(ROOT / utility_spec['path'], dataset,
                                    expected_sha256=utility_spec['sha256']))
    env = StreamingEnv(
        dataset,
        split=split,
        config=StreamConfig(**protocol['stream_config']),
        recovery=RecoveryConfig(**protocol['recovery']),
        weights=QoEWeights(**protocol['qoe_weights']),
        quality_utility=utility,
    )
    safety = ActionSafety(**protocol['safety'])
    episodes = []
    for trace_index, trace in enumerate(dataset.splits[split]):
        obs, _ = env.reset(seed=0, options={'trace_index': trace_index})
        total_reward = 0.0
        policy_actions, effective_actions, latency_ms = [], [], []
        planner_diagnostics = []
        while True:
            start_ns = time.perf_counter_ns()
            if policy_kind == 'highest_allowed' or policy_name == 'controlled_highest_allowed':
                mask = allowed_actions(obs, safety)
                action = int(np.flatnonzero(mask)[-1])
            elif policy_kind == 'robust_mpc' or policy_name == 'controlled_robust_mpc':
                action, details = mpc_policy.plan(
                    obs, env.config, env.weights, segment_index=env.segment)
                planner_diagnostics.append(details)
            else:
                action = int(model_policy(obs, env.config))
            latency_ms.append((time.perf_counter_ns() - start_ns) / 1e6)
            policy_actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info['segment_complete']:
                effective_actions.append(int(info.get('effective_action', action)))
            if terminated or truncated:
                break
        metrics = env.metrics()
        family = trace.name.removeprefix('test-').rsplit('-', 1)[0]
        episodes.append({
            'trace': trace.name,
            'trace_sha256': trace.fingerprint,
            'scenario_family': family,
            'source_group': trace.source_group,
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
            'planner_diagnostics': planner_diagnostics if mpc_policy else None,
        })
    return {
        'content': content,
        'policy': policy_name,
        'episodes': episodes,
        'summary': {
            'sessions': len(episodes),
            'completed': sum(item['completed'] for item in episodes),
            'mean_reward': float(np.mean([item['reward'] for item in episodes])),
            'mean_bitrate_kbps': float(np.mean([
                item['metrics']['mean_played_bitrate_kbps'] for item in episodes])),
            'mean_startup_s': float(np.mean([item['metrics']['startup_s'] for item in episodes])),
            'mean_rebuffer_s': float(np.mean([item['metrics']['rebuffer_s'] for item in episodes])),
            'rebuffer_events': int(sum(item['metrics']['stall_events'] for item in episodes)),
            'decision_latency_p50_ms': float(np.median([
                item['decision_latency_ms']['p50'] for item in episodes])),
            'decision_latency_p95_ms': float(np.max([
                item['decision_latency_ms']['p95'] for item in episodes])),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL_PATH)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if 'scenario_spec_sha256' in protocol:
        if sha256(ROOT / protocol['scenario_spec']) != protocol['scenario_spec_sha256']:
            raise ValueError('Frozen scenario specification hash mismatch')
    if 'training_protocol_sha256' in protocol:
        if sha256(ROOT / protocol['training_protocol']) != protocol['training_protocol_sha256']:
            raise ValueError('Frozen training protocol hash mismatch')
    for arm, training_protocol in protocol.get('training_protocols', {}).items():
        if sha256(ROOT / training_protocol['path']) != training_protocol['sha256']:
            raise ValueError(f'Frozen {arm} training protocol hash mismatch')
    safety = ActionSafety(**protocol['safety'])
    mpc_config = RobustMPCConfig(**protocol['robust_mpc'])
    reports, checkpoint_records, data_records = [], [], {}

    for content, content_protocol in protocol['contents'].items():
        dataset_path = ROOT / content_protocol['dataset']
        dataset = Dataset.load(dataset_path)
        expected = protocol['dataset_fingerprints'][content]
        if dataset.fingerprint != expected:
            raise ValueError(f'{content}: frozen dataset fingerprint mismatch')
        if len(dataset.splits['test']) != protocol['test_set']['profiles']:
            raise ValueError(f'{content}: frozen profile count mismatch')
        data_records[content] = {
            'dataset': str(dataset_path.relative_to(ROOT)),
            'fingerprint': dataset.fingerprint,
            'trace_count': len(dataset.splits['test']),
            'video_segments': len(dataset.video.durations_s),
            'video_duration_s': float(dataset.video.durations_s.sum()),
        }

        for policy_record in protocol['policies']:
            name = policy_record['name']
            if 'checkpoint' in policy_record:
                checkpoint = ROOT / policy_record['checkpoint']
                checkpoint_hash = sha256(checkpoint)
                if checkpoint_hash != policy_record['sha256']:
                    raise ValueError(f'Frozen checkpoint hash mismatch for {name}')
                model, metadata = load_model(checkpoint)
                if ('selected_episode' in policy_record
                        and metadata['training']['episode'] != policy_record['selected_episode']):
                    raise ValueError(f'Frozen checkpoint selection episode mismatch for {name}')
                policy = DecisionPolicy(model, safety=safety)
                report = evaluate(dataset, content, protocol, name,
                                  model_policy=policy, policy_kind='dqn')
                checkpoint_records.append({
                    'policy': name,
                    'checkpoint': str(checkpoint.relative_to(ROOT)),
                    'sha256': checkpoint_hash,
                    'selected_episode': metadata['training']['episode'],
                })
            elif policy_record.get('kind') == 'highest_allowed' or name == 'controlled_highest_allowed':
                report = evaluate(dataset, content, protocol, name,
                                  policy_kind='highest_allowed')
            elif policy_record.get('kind') == 'robust_mpc' or name == 'controlled_robust_mpc':
                mpc = RobustMPCPolicy(dataset.video.durations_s,
                                      config=mpc_config, safety=protocol['safety'])
                report = evaluate(dataset, content, protocol, name,
                                  mpc_policy=mpc, policy_kind='robust_mpc')
            else:
                raise ValueError(f'Unknown policy in frozen protocol: {name}')
            reports.append(report)
            print(json.dumps({'content': content, 'policy': name,
                              **report['summary']}), flush=True)

    payload = {
        'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(protocol_path),
        'scenario_spec_sha256': sha256(ROOT / protocol['scenario_spec']),
        'training_protocols': protocol.get('training_protocols'),
        'evaluator_sha256': sha256(Path(__file__)),
        'datasets': data_records,
        'checkpoint_records': checkpoint_records,
        'scenario_semantics': 'controlled_capacity; synthetic application-goodput replay',
        'stream_config': protocol['stream_config'],
        'recovery': protocol['recovery'],
        'qoe_weights': protocol['qoe_weights'],
        'safety': protocol['safety'],
        'robust_mpc': protocol['robust_mpc'],
        'policies': reports,
        'limitations': protocol['interpretation_rule'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'reports': len(reports),
                      'test_profiles_per_content': protocol['test_set']['profiles']}), flush=True)


if __name__ == '__main__':
    main()
