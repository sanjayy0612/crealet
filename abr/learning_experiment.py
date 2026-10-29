"""Small isolated DQN ablation; evaluation always uses original episode QoE."""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from .data import Dataset
from .env import StreamingEnv, StreamConfig, QoEWeights
from .learning_features import QueueObservation, queue_potential, credit_reward
from .models.checkpoint import load_model, save_trained_model
from .models.dueling_dqn import DuelingDoubleDQN, DuelingDoubleDQNConfig
from .models.policy import DecisionPolicy
from .models.safety import ActionSafety, allowed_actions
from .models.train_dueling_dqn import copy_observation, learn_batch, evaluate_policy, baseline_report, baseline_summary, checkpoint_eligible


def experiment_source_hash():
    return hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name('learning_features.py').read_bytes()).hexdigest()


def make_env(dataset, split, queue=False):
    env = StreamingEnv(dataset, split=split, config=StreamConfig(), weights=QoEWeights(rebuffer=12))
    return QueueObservation(env) if queue else env


def initialize_model(seed, queue=False):
    # Match all common weights. Added queue columns start at zero, so initial
    # scores are identical across arms; extra columns can receive gradients.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = DuelingDoubleDQN()
        if not queue:
            return base
        model = DuelingDoubleDQN(DuelingDoubleDQNConfig(queue_features=True))
        state = model.state_dict()
        for key, value in base.state_dict().items():
            if state[key].shape == value.shape:
                state[key] = value.clone()
        state['state_projection.weight'][:, :15] = base.state_projection.weight.detach()
        state['state_projection.weight'][:, 15:] = 0.
        model.load_state_dict(state)
        return model


def mean_stats(records):
    keys = records[0].keys() if records else []
    return {k: float(np.mean([r[k] for r in records if r[k] is not None]))
            for k in keys if any(r[k] is not None for r in records)}


def calibration(env, model, safety, shaped):
    policy = DecisionPolicy(model, safety=safety)
    errors, episodes = [], []
    for index, trace in enumerate(env.dataset.splits[env.split]):
        observation, _ = env.reset(options={'trace_index': index})
        rewards, predictions = [], []
        while True:
            scores = policy.scores(observation, env.config)
            action = int(np.where(allowed_actions(observation, safety), scores, -np.inf).argmax())
            predictions.append(float(scores[action]))
            before = queue_potential(env)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(credit_reward(reward, before, queue_potential(env), terminal=done) if shaped else reward)
            if done:
                break
        returns = np.cumsum(np.array(rewards)[::-1])[::-1]
        errors.extend(np.abs(np.array(predictions) - returns).tolist())
        episodes.append({'trace': trace.name, 'initial_predicted_q': predictions[0],
                         'realized_training_return': float(returns[0]), 'completed': bool(terminated)})
    return {'mean_abs_q_return_error': float(np.mean(errors)),
            'p90_abs_q_return_error': float(np.percentile(errors, 90)), 'episodes': episodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['baseline', 'credit', 'queue'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=120)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('Positive episode count required')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dataset = Dataset.load('data/hard_envivio/dataset.json')
    queue, shaped = args.variant == 'queue', args.variant == 'credit'
    train, validation = make_env(dataset, 'train', queue), make_env(dataset, 'validation', queue)
    safety = ActionSafety()
    baseline = baseline_report(dataset, None, 'validation', StreamConfig(), 'throughput', QoEWeights(rebuffer=12))
    limits = {e['trace']: min(8., e['metrics']['rebuffer_s']) for e in baseline['episodes']}
    online = initialize_model(args.seed, queue)
    target = initialize_model(args.seed, queue)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    replay = deque(maxlen=50000)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history, statistics = [], []
    steps = updates = 0
    best = None
    max_return_error = 0.
    last_episode_reward = 0.
    for episode in range(1, args.episodes + 1):
        observation, _ = train.reset(options={'trace_index': int(rng.integers(len(dataset.splits['train'])))})
        original_return = training_return = 0.
        while True:
            epsilon = max(.05, 1. - .95 * steps / 10000)
            if rng.random() < epsilon:
                action = int(rng.choice(np.flatnonzero(allowed_actions(observation, safety))))
            else:
                action = DecisionPolicy(online, safety=safety)(observation, train.config)
            current = copy_observation(observation)
            before = queue_potential(train)
            observation, reward, terminated, truncated, _ = train.step(action)
            done = bool(terminated or truncated)
            training_reward = credit_reward(reward, before, queue_potential(train), terminal=done) if shaped else reward
            original_return += reward
            training_return += training_reward
            replay.append((current, action, training_reward,
                           current if done else copy_observation(observation), done))
            steps += 1
            if len(replay) >= 500 and steps % 4 == 0:
                online.train()
                statistics.append(learn_batch(online, target, optimizer, replay, 64, train.config,
                                             'cpu', 1., safety, diagnostics=True))
                updates += 1
                if updates % 250 == 0:
                    target.load_state_dict(online.state_dict())
                    target.eval()
            if done:
                break
        max_return_error = max(max_return_error, abs(original_return - training_return))
        if abs(original_return - training_return) > 1e-6:
            raise RuntimeError('Reward credit changed total episode return')
        last_episode_reward = original_return
        if episode % 20 == 0 or episode == args.episodes:
            online.eval()
            report = evaluate_policy(validation, online, safety=safety, stall_limits_s=limits)
            eligible = checkpoint_eligible(report, 3.)
            record = {'episode': episode, 'environment_steps': steps, 'gradient_steps': updates,
                      'target_syncs': updates // 250, 'validation': report, 'eligible': eligible,
                      'learning': mean_stats(statistics), 'train_original_return': original_return,
                      'train_learning_return': training_return}
            history.append(record)
            statistics.clear()
            metadata = {'algorithm': 'Dueling Double DQN', 'variant': args.variant, 'seed': args.seed,
                        'episode': episode, 'environment_steps': steps, 'gradient_steps': updates,
                        'training_dataset_sha256': dataset.fingerprint, 'validation_split': 'validation',
                        'config': asdict(train.config), 'weights': asdict(train.unwrapped.weights),
                        'safety_filter': asdict(safety), 'max_startup_s': 3., 'max_rebuffer_s': 8.,
                        'observation_extension': 'queue_summary_v1' if queue else None,
                        'reward_credit': shaped, 'gamma': 1., 'best_validation': report,
                        'experiment_source_sha256': experiment_source_hash(),
                        'stall_constraint': {'limits_s': limits, 'met': report['stall_constraint_met']}}
            if eligible and (best is None or report['mean_reward'] > best):
                best = report['mean_reward']
                save_trained_model(online, output / 'accepted.npz', training_metadata=metadata)
            print(json.dumps({k: v for k, v in record.items() if k != 'validation'} |
                             {'validation': {k: v for k, v in report.items() if k != 'episodes'}}), flush=True)
    final_report = history[-1]['validation']
    final_calibration = calibration(validation, online, safety, shaped)
    selected = selected_metadata = selected_report = selected_calibration = None
    if best is not None:
        selected, selected_metadata = load_model(output / 'accepted.npz')
        selected_report = evaluate_policy(validation, selected, safety=safety, stall_limits_s=limits)
        selected_calibration = calibration(validation, selected, safety, shaped)
    result = {'variant': args.variant, 'seed': args.seed, 'episodes': args.episodes,
              'environment_steps': steps, 'gradient_steps': updates,
              'dataset_sha256': dataset.fingerprint, 'test_used_for_selection': False,
              'accepted_checkpoint': str(output / 'accepted.npz') if best is not None else None,
              'selected_metadata': selected_metadata, 'selected_validation': selected_report,
              'selected_q_calibration': selected_calibration,
              'final_validation': final_report, 'final_q_calibration': final_calibration,
              'max_episode_return_shift_error': max_return_error, 'history': history}
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
