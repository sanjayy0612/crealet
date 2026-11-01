"""Fixed matched DQN runs on the new ladder/cap; all evaluation uses original QoE."""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from .benchmark import evaluate
from .data import Dataset
from .env import StreamingEnv, StreamConfig, QoEWeights
from .quality import load_quality_utility
from .models.checkpoint import save_trained_model, load_model
from .models.dueling_dqn import DuelingDoubleDQN
from .models.policy import DecisionPolicy
from .models.safety import ActionSafety, allowed_actions
from .models.train_dueling_dqn import copy_observation, learn_batch, checkpoint_eligible
from .recovery import RecoveryConfig
from scripts.evaluate_request_recovery import compact


def make_env(dataset, split, protocol):
    utility_spec = protocol.get('quality_utility')
    utility = (None if utility_spec is None else
               load_quality_utility(utility_spec['path'], dataset,
                                    expected_sha256=utility_spec['sha256']))
    return StreamingEnv(dataset, split=split, config=StreamConfig(**protocol['config']),
                        recovery=RecoveryConfig(**protocol['recovery']), weights=QoEWeights(rebuffer=12),
                        quality_utility=utility)


def score(env, model, safety, stall_limits, stall_tolerance_s=1e-9):
    report = evaluate(env, baseline='model', policy=DecisionPolicy(model, safety=safety),
                      policy_metadata={'kind': 'matched_startup_dqn', 'safety': asdict(safety)})
    episodes = report['episodes']
    return {'mean_reward': float(np.mean([e['reward'] for e in episodes])),
            'completion_rate': report['completion_rate'],
            'stall_constraint_met': all(e['metrics']['rebuffer_s'] <= stall_limits[e['trace']] + stall_tolerance_s for e in episodes),
            'stall_tolerance_s': float(stall_tolerance_s),
            'mean_played_bitrate_kbps': float(np.mean([e['metrics']['mean_played_bitrate_kbps'] for e in episodes])),
            'episodes': [{'trace': e['trace'], 'completed': e['completed'], 'startup_s': e['metrics']['startup_s'],
                          'rebuffer_s': e['metrics']['rebuffer_s'], 'reward': e['reward']} for e in episodes]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--protocol', type=Path, default=Path('runs/startup-ladder/protocol.json'),
                        help='Frozen training protocol JSON')
    parser.add_argument('--output-root', type=Path,
                        help='Run directory root; defaults to runs/startup-ladder')
    args = parser.parse_args()
    protocol_path = args.protocol
    protocol = json.loads(protocol_path.read_text())
    output_root = args.output_root or Path('runs/startup-ladder')
    gate = json.loads(Path('runs/startup-ladder/frozen-native-gate.json').read_text())
    if not gate['training_gate_passed']:
        raise ValueError('Frozen native/engineering gate has not passed')
    hp = protocol['fixed_training']
    if args.seed not in hp['seeds']:
        raise ValueError('Seed not in fixed protocol')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dataset = Dataset.load(protocol['training_dataset'])
    train, validation = make_env(dataset, 'train', protocol), make_env(dataset, 'validation', protocol)
    safety = ActionSafety(**protocol['safety'])
    stall_tolerance_s = float(protocol.get('stall_tolerance_s', 1e-9))
    if not np.isfinite(stall_tolerance_s) or stall_tolerance_s < 0:
        raise ValueError('stall_tolerance_s must be finite and nonnegative')
    rule = evaluate(make_env(dataset, 'validation', protocol), baseline='model',
                    policy=lambda obs, cfg: int(np.flatnonzero(allowed_actions(obs, safety))[-1]),
                    policy_metadata={'kind': 'highest_allowed', 'safety': asdict(safety)})
    rule_reward = float(np.mean([e['reward'] for e in rule['episodes']]))
    stall_limits = {e['trace']: min(8., e['metrics']['rebuffer_s']) for e in rule['episodes']}
    model, target = DuelingDoubleDQN(), DuelingDoubleDQN()
    target.load_state_dict(model.state_dict()); target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])
    replay = deque(maxlen=hp['replay_capacity'])
    output = output_root / f'seed-{args.seed}'
    output.mkdir(exist_ok=True)
    history, best = [], None
    steps = updates = 0
    for episode in range(1, hp['episodes'] + 1):
        obs, _ = train.reset(options={'trace_index': int(rng.integers(len(dataset.splits['train'])))})
        while True:
            epsilon = max(.05, 1. - .95 * steps / hp['epsilon_steps'])
            action = int(rng.choice(np.flatnonzero(allowed_actions(obs, safety)))) if rng.random() < epsilon else DecisionPolicy(model, safety=safety)(obs, train.config)
            before = copy_observation(obs)
            obs, reward, done, truncated, _ = train.step(action)
            terminal = done or truncated
            replay.append((before, action, reward, before if terminal else copy_observation(obs), terminal))
            steps += 1
            if len(replay) >= hp['warmup'] and steps % hp['learn_every'] == 0:
                model.train()
                learn_batch(model, target, optimizer, replay, hp['batch_size'], train.config, 'cpu', hp['gamma'], safety)
                updates += 1
                if updates % hp['target_every'] == 0:
                    target.load_state_dict(model.state_dict()); target.eval()
            if terminal:
                break
        if episode % hp['validation_every'] == 0:
            model.eval()
            result = score(validation, model, safety, stall_limits, stall_tolerance_s)
            eligible = checkpoint_eligible(result, 3.)
            record = {'episode': episode, 'steps': steps, 'updates': updates, 'eligible': eligible, 'validation': result}
            history.append(record)
            diagnostic = protocol.get('checkpoint_selection_diagnostic', {})
            metadata = {'algorithm': 'Dueling Double DQN', 'seed': args.seed, 'episode': episode,
                        'training_dataset_sha256': dataset.fingerprint, 'config': asdict(train.config),
                        'weights': asdict(train.weights), 'safety_filter': asdict(safety),
                        'recovery': protocol['recovery'], 'validation': result,
                        'validation_eligible': eligible,
                        'quality_utility': protocol.get('quality_utility'),
                        'max_startup_s': 3., 'max_rebuffer_s': 8.,
                        'stall_tolerance_s': stall_tolerance_s,
                        'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
            metadata['stall_limits_s'] = stall_limits
            if diagnostic.get('save_all_validation_checkpoints'):
                checkpoint_dir = output / 'validation-checkpoints'
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f'episode-{episode:03d}.npz'
                save_trained_model(model, checkpoint_path, training_metadata=metadata)
                record['checkpoint'] = str(checkpoint_path)
            if eligible and (best is None or result['mean_reward'] > best):
                best = result['mean_reward']
                metadata['best_validation'] = result
                save_trained_model(model, output / 'accepted.npz', training_metadata=metadata)
            print(json.dumps({'seed': args.seed, 'episode': episode, 'validation_reward': result['mean_reward'],
                              'rule_reward': rule_reward, 'eligible': eligible}), flush=True)
    payload = {'seed': args.seed, 'episodes': hp['episodes'], 'steps': steps, 'updates': updates,
               'selected_checkpoint': str(output / 'accepted.npz') if best is not None else None,
               'selected_validation_reward': best, 'rule_validation_reward': rule_reward,
               'beats_rule_on_validation': best is not None and best > rule_reward + 1e-6,
               'history': history, 'validation_rule': compact(rule),
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'metrics.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
