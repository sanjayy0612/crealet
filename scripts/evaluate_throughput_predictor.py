"""Final frozen comparison of learned and harmonic throughput estimates on v4 test traces."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from abr.data import Dataset
from abr.env import QoEWeights, StreamConfig, StreamingEnv
from abr.models.safety import ActionSafety, allowed_actions
from abr.quality import load_quality_utility
from abr.recovery import RecoveryConfig
from abr.robust_mpc import RobustMPCConfig, RobustMPCPolicy
from abr.throughput_predictor import (LearnedThroughputPredictor, features,
                                      future_mean_capacity, harmonic_robust_prediction)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'experiments/configs/throughput_predictor_v1.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forecast_samples(dataset, protocol, predictor, split='test'):
    cfg = protocol['evaluation']
    env = StreamingEnv(dataset, split=split, history=protocol['training']['history_length'],
                       config=StreamConfig(**cfg['stream_config']),
                       recovery=RecoveryConfig(**cfg['recovery']),
                       weights=QoEWeights(**cfg['qoe_weights']))
    safety = ActionSafety(**cfg['safety'])
    targets, learned, harmonic, families = [], [], [], []
    for trace_index, trace in enumerate(dataset.splits[split]):
        obs, _ = env.reset(seed=0, options={'trace_index': trace_index})
        while True:
            duration = float(obs['candidates'][0, 2])
            target = future_mean_capacity(trace, env.network.time_s + trace.rtt_s, duration)
            if target is not None:
                targets.append(target)
                learned.append(predictor.predict(obs)[0])
                harmonic.append(harmonic_robust_prediction(
                    obs, RobustMPCConfig(**cfg['robust_mpc'])))
                if trace.semantics == 'observed_goodput' and trace.source_group:
                    families.append(trace.source_group)
                else:
                    families.append(trace.name.removeprefix('test-').rsplit('-', 1)[0])
            mask = allowed_actions(obs, safety)
            action = int(np.flatnonzero(mask)[-1])
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    target = np.asarray(targets, dtype=np.float64)
    learned = np.asarray(learned, dtype=np.float64)
    harmonic = np.asarray(harmonic, dtype=np.float64)
    q = float(protocol['training']['quantile'])

    def metrics(prediction):
        error = target - prediction
        return {'samples': len(target),
                'mae_mbps': float(np.mean(np.abs(error))),
                'rmse_mbps': float(np.sqrt(np.mean(error ** 2))),
                'underprediction_fraction': float(np.mean(error > 0)),
                'overprediction_fraction': float(np.mean(error < 0)),
                'mean_prediction_mbps': float(np.mean(prediction))}

    # Quantile coverage is the key calibration check for the learned p20 output.
    report = {'target': 'mean trace capacity during next segment, after RTT; Mbps',
              'observation_policy': 'highest_allowed', 'split': split,
              'target_mean_mbps': float(np.mean(target)),
              'learned_p20': metrics(learned), 'harmonic_robust_point': metrics(harmonic),
              'learned_p20_pinball_mbps': float(np.mean(np.maximum(q * (target - learned),
                                                                     (q - 1) * (target - learned)))),
              'learned_p20_coverage': float(np.mean(target <= learned)),
              'by_family': {}}
    for family in sorted(set(families)):
        mask = np.asarray(families) == family
        report['by_family'][family] = {
            'samples': int(mask.sum()),
            'learned_mae_mbps': float(np.mean(np.abs(target[mask] - learned[mask]))),
            'harmonic_mae_mbps': float(np.mean(np.abs(target[mask] - harmonic[mask]))),
            'learned_p20_coverage': float(np.mean(target[mask] <= learned[mask]))}
    return report


def evaluate_policy(dataset, content, protocol, name, kind, predictor=None):
    cfg = protocol['evaluation']
    quality_spec = cfg['quality_utility_by_content'][content]
    quality = load_quality_utility(ROOT / quality_spec['path'], dataset,
                                   expected_sha256=quality_spec['sha256'])
    env = StreamingEnv(dataset, split='test', config=StreamConfig(**cfg['stream_config']),
                       recovery=RecoveryConfig(**cfg['recovery']),
                       weights=QoEWeights(**cfg['qoe_weights']), quality_utility=quality)
    safety = ActionSafety(**cfg['safety'])
    mpc = None
    if kind != 'highest_allowed':
        mpc = RobustMPCPolicy(dataset.video.durations_s,
                              config=RobustMPCConfig(**cfg['robust_mpc']),
                              safety=cfg['safety'], throughput_predictor=predictor)
    episodes = []
    for trace_index, trace in enumerate(dataset.splits['test']):
        obs, _ = env.reset(seed=0, options={'trace_index': trace_index})
        total_reward, latencies = 0.0, []
        planner_diagnostics = []
        while True:
            start = time.perf_counter_ns()
            if kind == 'highest_allowed':
                action = int(np.flatnonzero(allowed_actions(obs, safety))[-1])
            else:
                action, details = mpc.plan(obs, env.config, env.weights,
                                           segment_index=env.segment)
                planner_diagnostics.append(details)
            latencies.append((time.perf_counter_ns() - start) / 1e6)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        metrics = env.metrics()
        episodes.append({'trace': trace.name, 'trace_sha256': trace.fingerprint,
                         'scenario_family': trace.name.removeprefix('test-').rsplit('-', 1)[0],
                         'completed': bool(terminated), 'termination_reason': info['termination_reason'],
                         'reward': float(total_reward), 'metrics': metrics,
                         'decision_latency_ms': {'p50': float(np.percentile(latencies, 50)),
                                                 'p95': float(np.percentile(latencies, 95)),
                                                 'max': float(np.max(latencies))},
                         'planner_diagnostics': planner_diagnostics if mpc else None})
    return {'content': content, 'policy': name, 'episodes': episodes,
            'summary': {'sessions': len(episodes),
                        'completed': int(sum(e['completed'] for e in episodes)),
                        'mean_reward': float(np.mean([e['reward'] for e in episodes])),
                        'mean_quality_utility': float(np.mean([
                            e['metrics']['quality_integral'] / max(e['metrics']['played_s'], 1e-12)
                            for e in episodes])),
                        'mean_played_bitrate_kbps': float(np.mean([
                            e['metrics']['mean_played_bitrate_kbps'] for e in episodes])),
                        'mean_startup_s': float(np.mean([e['metrics']['startup_s'] for e in episodes])),
                        'mean_rebuffer_s': float(np.mean([e['metrics']['rebuffer_s'] for e in episodes])),
                        'rebuffer_events': int(sum(e['metrics']['stall_events'] for e in episodes)),
                        'decision_latency_p50_ms': float(np.median([
                            e['decision_latency_ms']['p50'] for e in episodes])),
                        'decision_latency_p95_ms': float(np.max([
                            e['decision_latency_ms']['p95'] for e in episodes]))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs/throughput-predictor-v1/final-results.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = json.loads(PROTOCOL.read_text())
    if sha256(ROOT / protocol['scenario_spec']) != protocol['scenario_spec_sha256']:
        raise ValueError('Frozen scenario spec hash mismatch')
    expected = protocol['dataset_fingerprints']
    datasets = {}
    for content in protocol['evaluation']['contents']:
        dataset = Dataset.load(ROOT / protocol['dataset_root'] / f'{content}.json')
        if dataset.fingerprint != expected[content]:
            raise ValueError(f'{content} dataset fingerprint mismatch')
        if len(dataset.splits['test']) != protocol['evaluation']['profile_count']:
            raise ValueError(f'{content} test profile count mismatch')
        datasets[content] = dataset
    checkpoint_path = ROOT / protocol['artifact']
    training_report = json.loads((checkpoint_path.parent / 'training.json').read_text())
    if training_report['protocol_sha256'] != sha256(PROTOCOL):
        raise ValueError('Predictor training protocol hash mismatch')
    if training_report['artifact_sha256'] != sha256(checkpoint_path):
        raise ValueError('Predictor checkpoint hash mismatch')
    predictor = LearnedThroughputPredictor.load(
        checkpoint_path, expected_dataset_fingerprint=expected['envivio'])

    forecast = forecast_samples(datasets['envivio'], protocol, predictor)
    reports = []
    for content, dataset in datasets.items():
        reports.append(evaluate_policy(dataset, content, protocol, 'highest_allowed',
                                       'highest_allowed'))
        reports.append(evaluate_policy(dataset, content, protocol,
                                       'harmonic_mean_robust_mpc', 'robust_mpc'))
        reports.append(evaluate_policy(dataset, content, protocol,
                                       'learned_forecast_robust_mpc', 'robust_mpc', predictor))
        for report in reports[-3:]:
            print(json.dumps({'content': content, 'policy': report['policy'],
                              **report['summary']}), flush=True)
    payload = {'protocol_id': protocol['protocol_id'],
               'protocol_sha256': sha256(PROTOCOL),
               'scenario_spec_sha256': sha256(ROOT / protocol['scenario_spec']),
               'datasets': {key: {'fingerprint': value.fingerprint,
                                  'test_profiles': len(value.splits['test'])}
                            for key, value in datasets.items()},
               'predictor': {'path': str(checkpoint_path.relative_to(ROOT)),
                             'sha256': sha256(checkpoint_path),
                             'selected_epoch': training_report['selected_epoch'],
                             'training_protocol_sha256': training_report['protocol_sha256']},
               'forecast_accuracy': forecast, 'policies': reports,
               'scenario_semantics': 'synthetic controlled application-goodput; not measured network data',
               'limitations': protocol['interpretation_rule']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'policy_reports': len(reports),
                      'forecast_samples': forecast['learned_p20']['samples']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
