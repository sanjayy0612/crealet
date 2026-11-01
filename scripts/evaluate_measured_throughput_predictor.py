"""Evaluate the frozen predictor on the measured Ghent route-holdout sessions."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from abr.data import Dataset
from abr.throughput_predictor import LearnedThroughputPredictor
from scripts.evaluate_throughput_predictor import evaluate_policy, forecast_samples


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'experiments/configs/throughput_predictor_ghent_diagnostic_v1.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    for relative_path, expected in protocol['evaluation_code_sha256'].items():
        if sha256(ROOT / relative_path) != expected:
            raise ValueError(f'Frozen evaluation code hash mismatch: {relative_path}')
    model_protocol_path = ROOT / protocol['predictor_protocol']
    if sha256(model_protocol_path) != protocol['predictor_protocol_sha256']:
        raise ValueError('Frozen synthetic predictor protocol hash mismatch')
    base = json.loads(model_protocol_path.read_text())
    dataset_path = ROOT / protocol['dataset']
    if sha256(dataset_path) != protocol['dataset_file_sha256']:
        raise ValueError('Measured dataset file hash mismatch')
    dataset = Dataset.load(dataset_path)
    if dataset.fingerprint != protocol['dataset_fingerprint']:
        raise ValueError('Measured dataset fingerprint mismatch')
    traces = dataset.splits[protocol['test_split']]
    if len(traces) != protocol['test_sessions']:
        raise ValueError('Measured test session count mismatch')
    checkpoint_path = ROOT / protocol['predictor_checkpoint']
    if sha256(checkpoint_path) != protocol['predictor_checkpoint_sha256']:
        raise ValueError('Frozen predictor checkpoint hash mismatch')
    training_report = json.loads((checkpoint_path.parent / 'training.json').read_text())
    if training_report['protocol_sha256'] != protocol['predictor_protocol_sha256']:
        raise ValueError('Predictor was not trained under the pinned synthetic protocol')
    if training_report['dataset_fingerprint'] != base['dataset_fingerprints']['envivio']:
        raise ValueError('Predictor training data differs from frozen synthetic dataset')
    predictor = LearnedThroughputPredictor.load(
        checkpoint_path, expected_dataset_fingerprint=base['dataset_fingerprints']['envivio'])

    torch.set_num_threads(1)
    forecast = forecast_samples(dataset, base, predictor, split=protocol['test_split'])
    reports = []
    for name, kind, model in (
        ('highest_allowed', 'highest_allowed', None),
        ('harmonic_mean_robust_mpc', 'robust_mpc', None),
        ('learned_forecast_robust_mpc', 'robust_mpc', predictor),
    ):
        report = evaluate_policy(dataset, 'envivio', base, name, kind, model)
        for episode, trace in zip(report['episodes'], traces):
            episode['transport_mode'] = trace.source_group.removeprefix('ghent-')
            episode.pop('scenario_family', None)
        report['summary']['by_transport_mode'] = summarize_modes(report['episodes'])
        reports.append(report)
        print(json.dumps({'policy': name, **report['summary']}), flush=True)

    output = args.output or ROOT / protocol['output']
    payload = {
        'protocol_id': protocol['protocol_id'],
        'protocol_sha256': sha256(PROTOCOL_PATH),
        'source': protocol['source'], 'source_url': protocol['source_url'],
        'dataset': str(dataset_path.relative_to(ROOT)),
        'dataset_file_sha256': sha256(dataset_path),
        'dataset_fingerprint': dataset.fingerprint,
        'split': protocol['test_split'], 'sessions': len(traces),
        'measured_data_semantics': protocol['measurement_semantics'],
        'assumed_rtt_s': protocol['rtt_s'],
        'reuse_disclosure': protocol['reuse_disclosure'],
        'predictor': {'checkpoint': str(checkpoint_path.relative_to(ROOT)),
                      'checkpoint_sha256': sha256(checkpoint_path),
                      'trained_only_on_synthetic_dataset': training_report['dataset_fingerprint'],
                      'training_protocol_sha256': training_report['protocol_sha256']},
        'forecast_accuracy': forecast, 'policies': reports,
        'limitations': [
            'Historical 2010-2011 Ghent 4G sessions, not current-network measurements.',
            'The source provides interval HTTP goodput, not physical link capacity; RTT is assumed to be 80 ms.',
            'These test sessions have appeared in earlier project studies, although the current predictor was fit only on synthetic data and was not tuned on them.',
            'Trace replay does not model TCP/QUIC, radio scheduling, packet loss, or live adaptation.',
            'Results on one historical region and content ladder do not establish broad generalization.'
        ]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'policy_reports': len(reports),
                      'sessions_per_policy': len(traces),
                      'forecast_samples': forecast['learned_p20']['samples']}, indent=2), flush=True)


def summarize_modes(episodes):
    summary = {}
    for mode in sorted({episode['transport_mode'] for episode in episodes}):
        rows = [episode for episode in episodes if episode['transport_mode'] == mode]
        summary[mode] = {
            'sessions': len(rows),
            'completed': int(sum(row['completed'] for row in rows)),
            'mean_reward': float(sum(row['reward'] for row in rows) / len(rows)),
            'mean_quality_utility': float(sum(
                row['metrics']['quality_integral'] / max(row['metrics']['played_s'], 1e-12)
                for row in rows) / len(rows)),
            'mean_rebuffer_s': float(sum(row['metrics']['rebuffer_s'] for row in rows) / len(rows)),
        }
    return summary


if __name__ == '__main__':
    main()
