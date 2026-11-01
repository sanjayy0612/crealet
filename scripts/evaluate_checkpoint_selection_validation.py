"""Replay every saved eligible checkpoint on validation only for VMAF selection."""
import hashlib
import json
from pathlib import Path

import torch

from abr.data import Dataset
from abr.models.checkpoint import load_model
from abr.models.policy import DecisionPolicy
from scripts.evaluate_controlled_capacity import evaluate
from abr.models.safety import ActionSafety


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'experiments/configs/controlled_capacity_checkpoint_selection_v1.json'
EVALUATION_PROTOCOL_PATH = ROOT / 'experiments/configs/controlled_capacity_stress_transfer_v2.json'
OUTPUT = ROOT / 'runs/controlled-capacity-selection-v1/validation_decisions.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    protocol = json.loads(PROTOCOL_PATH.read_text())
    evaluation_protocol = json.loads(EVALUATION_PROTOCOL_PATH.read_text())
    dataset_path = ROOT / protocol['training_dataset']
    dataset = Dataset.load(dataset_path)
    if dataset.fingerprint != protocol['scenario_training']['training_dataset_sha256']:
        raise ValueError('Training/validation dataset fingerprint mismatch')
    safety = ActionSafety(**protocol['safety'])
    reports, checkpoint_records = [], []

    for seed in protocol['fixed_training']['seeds']:
        run_dir = ROOT / 'runs/controlled-capacity-selection-v1' / f'seed-{seed}'
        training = json.loads((run_dir / 'metrics.json').read_text())
        for record in training['history']:
            if not record.get('eligible'):
                continue
            checkpoint = ROOT / record['checkpoint']
            model, metadata = load_model(checkpoint)
            if metadata['training']['episode'] != record['episode']:
                raise ValueError(f'Seed {seed}: checkpoint episode mismatch')
            if not metadata['training']['validation_eligible']:
                raise ValueError(f'Seed {seed}: ineligible checkpoint was marked for selection')
            name = f'selection_v1_seed_{seed}_episode_{record["episode"]:03d}'
            report = evaluate(
                dataset, 'envivio', evaluation_protocol, name,
                model_policy=DecisionPolicy(model, safety=safety),
                policy_kind='dqn', split='validation')
            report['seed'] = seed
            report['training_episode'] = record['episode']
            report['validation_eligible'] = True
            report['checkpoint_sha256'] = sha256(checkpoint)
            reports.append(report)
            checkpoint_records.append({
                'policy': name,
                'seed': seed,
                'training_episode': record['episode'],
                'checkpoint': str(checkpoint.relative_to(ROOT)),
                'sha256': sha256(checkpoint),
                'validation_mean_reward': record['validation']['mean_reward'],
            })
            print(json.dumps({'policy': name, **report['summary']}), flush=True)

    baseline_report = evaluate(
        dataset, 'envivio', evaluation_protocol, 'selection_v1_highest_allowed',
        policy_kind='highest_allowed', split='validation')
    reports.append(baseline_report)

    payload = {
        'protocol_id': 'abr-controlled-capacity-checkpoint-selection-v1',
        'protocol_sha256': sha256(PROTOCOL_PATH),
        'evaluation_protocol_sha256': sha256(EVALUATION_PROTOCOL_PATH),
        'training_dataset_sha256': dataset.fingerprint,
        'split': 'validation',
        'checkpoint_records': checkpoint_records,
        'policies': reports,
        'selection_scope': 'Only the validation split was replayed or used to compare checkpoint criteria; the test split was not replayed or used for selection.',
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(OUTPUT), 'eligible_checkpoints': len(checkpoint_records),
                      'validation_traces_per_checkpoint': len(dataset.splits['validation'])}), flush=True)


if __name__ == '__main__':
    main()
