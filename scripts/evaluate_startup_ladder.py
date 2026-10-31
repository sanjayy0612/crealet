"""Frozen screen of real startup ladder; fixed protocol precedes scoring."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from abr.benchmark import evaluate
from abr.data import Dataset
from abr.env import StreamingEnv, StreamConfig, QoEWeights
from abr.models.checkpoint import load_model
from abr.models.policy import DecisionPolicy
from abr.models.safety import ActionSafety, allowed_actions
from abr.recovery import RecoveryConfig
from scripts.evaluate_request_recovery import compact


def main():
    torch.set_num_threads(1)
    out = Path('runs/startup-ladder')
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path('runs/learning-ablation/baseline-seed-7/accepted.npz')
    config, recovery, safety = StreamConfig(max_buffer_s=60.), RecoveryConfig(), ActionSafety()
    protocol = {'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'config': asdict(config), 'recovery': asdict(recovery), 'safety': asdict(safety),
                'media_index_sha256': hashlib.sha256(Path('artifacts/startup-ladder-v1/index.json').read_bytes()).hexdigest(),
                'training_dataset': 'data/startup-ladder/training.json', 'development_dataset': 'data/startup-ladder/development.json',
                'targets': {'startup_s': 3., 'rebuffer_s': 8., 'complete': True},
                'fixed_training': {'seeds': [7, 11, 13], 'episodes': 120, 'validation_every': 20,
                                   'warmup': 500, 'learn_every': 4, 'batch_size': 64, 'target_every': 250,
                                   'replay_capacity': 50000, 'learning_rate': .0001, 'gamma': 1.,
                                   'epsilon_steps': 10000, 'baseline': 'highest_allowed'},
                'training_gate': 'All frozen validation/development and six native playback cases must meet timing/completion limits',
                'quality_warning': '8 kbps / 96x54 is a survival rung; startup gain trades substantial visual quality',
                'test_status': 'Known development sessions and the same video; no claim of untouched-test generalization'}
    path = out / 'protocol.json'
    serialized = json.dumps(protocol, indent=2) + '\n'
    if path.exists() and path.read_text() != serialized:
        raise ValueError('Existing protocol differs')
    path.write_text(serialized)
    cases = [{'name': f'{label}-{policy.replace("_", "-")}', 'policy': policy, 'split': 'test',
              'trace': index, 'port': 8861, 'wall_limit_s': 240}
             for label, index in [('slow-first', 0), ('slow-start', 3), ('hard', 4)]
             for policy in ('dqn', 'highest_allowed')]
    (out / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    model, _ = load_model(checkpoint)
    dqn = DecisionPolicy(model, safety=safety)
    results = []
    for label, filename, split in [('validation', protocol['training_dataset'], 'validation'),
                                    ('development', protocol['development_dataset'], 'test')]:
        data = Dataset.load(filename)
        for name in ('lowest', 'highest_allowed', 'dqn'):
            env = StreamingEnv(data, split=split, config=config, weights=QoEWeights(rebuffer=12), recovery=recovery)
            kwargs = {'baseline': 'lowest'} if name == 'lowest' else {
                'baseline': 'model', 'policy': dqn if name == 'dqn' else
                lambda obs, cfg: int(np.flatnonzero(allowed_actions(obs, safety))[-1]),
                'policy_metadata': {'kind': name, 'changed_media': True, 'checkpoint': protocol['checkpoint_sha256']}}
            report = compact(evaluate(env, **kwargs))
            report.update(label=label, policy=name)
            results.append(report)
            print(json.dumps({'label': label, 'policy': name, 'passed': [e['playback_targets_met'] for e in report['episodes']],
                              'startup_s': [round(e['metrics']['startup_s'], 3) for e in report['episodes']]}), flush=True)
    (out / 'frozen-simulator.json').write_text(json.dumps({'protocol': protocol, 'results': results}, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
