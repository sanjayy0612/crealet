"""Train a dueling action-value model with Double DQN.

Example:
    python -m abr.models.train_dueling_dqn --episodes 200 --seed 7 --data-seed 7
"""
import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from ..benchmark import evaluate
from ..data import Dataset, synthetic_dataset
from ..env import StreamingEnv, StreamConfig, QoEWeights
from ..media import Media
from .checkpoint import load_model, save_trained_model
from .dueling_dqn import DuelingDoubleDQN
from .inputs import make_batch
from .policy import DecisionPolicy


def copy_observation(observation):
    return {key: np.array(value, copy=True) for key, value in observation.items()}


def checkpoint_eligible(validation, max_startup_s=None):
    return bool(validation['completion_rate'] == 1.0 and validation['stall_constraint_met']
                and (max_startup_s is None or all(e['startup_s'] <= max_startup_s for e in validation['episodes'])))


def evaluate_policy(env, model, *, stall_limits_s=None):
    policy = DecisionPolicy(model)
    rewards, completions, stalls, bitrates = [], [], [], []
    episodes = []
    for index, _trace in enumerate(env.dataset.splits[env.split]):
        observation, _ = env.reset(options={'trace_index': index})
        total_reward = 0.
        selected_bitrates = []
        while True:
            action = policy(observation, env.config)
            selected_bitrates.append(float(observation['candidates'][action, 0] * 1000.))
            observation, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        metrics = env.metrics()
        rewards.append(total_reward)
        completions.append(float(terminated))
        stalls.append(metrics['rebuffer_s'])
        bitrates.append(metrics['mean_played_bitrate_kbps'])
        limit = None if stall_limits_s is None else float(stall_limits_s[_trace.name])
        episodes.append({'trace': _trace.name, 'completed': bool(terminated),
                         'reward': float(total_reward), 'rebuffer_s': float(metrics['rebuffer_s']),
                         'startup_s': float(metrics['startup_s']),
                         'stall_events': int(metrics['stall_events']),
                         'stall_limit_s': limit,
                         'within_stall_limit': None if limit is None else bool(metrics['rebuffer_s'] <= limit + 1e-9),
                         'mean_played_bitrate_kbps': float(metrics['mean_played_bitrate_kbps']),
                         'selected_bitrates_kbps': selected_bitrates})
    return {'mean_reward': float(np.mean(rewards)),
            'completion_rate': float(np.mean(completions)),
            'mean_rebuffer_s': float(np.mean(stalls)),
            'max_rebuffer_s': float(np.max(stalls)),
            'mean_played_bitrate_kbps': float(np.mean(bitrates)),
            'stall_constraint_met': None if stall_limits_s is None else all(x['within_stall_limit'] for x in episodes),
            'episodes': episodes}


def baseline_report(dataset, media, split, config, name, weights=None):
    report = evaluate(StreamingEnv(dataset, media=media, split=split, config=config, weights=weights), baseline=name)
    return report


def baseline_summary(report):
    return {'mean_reward': float(np.mean([item['reward'] for item in report['episodes']])),
            'completion_rate': float(report['completion_rate']),
            'mean_rebuffer_s': float(np.mean([item['metrics']['rebuffer_s'] for item in report['episodes']])),
            'episodes': [{'trace': item['trace'], 'completed': bool(item['completed']),
                          'reward': float(item['reward']),
                          'rebuffer_s': float(item['metrics']['rebuffer_s']),
                          'startup_s': float(item['metrics']['startup_s']),
                          'stall_events': int(item['metrics']['stall_events']),
                          'mean_played_bitrate_kbps': float(item['metrics']['mean_played_bitrate_kbps'])}
                         for item in report['episodes']]}


def learn_batch(online, target, optimizer, replay, batch_size, config, device, gamma):
    items = random.sample(replay, batch_size)
    observations = [item[0] for item in items]
    actions = torch.tensor([item[1] for item in items], dtype=torch.long, device=device)
    rewards = torch.tensor([item[2] for item in items], dtype=torch.float32, device=device)
    next_observations = [item[0] if item[4] else item[3] for item in items]
    dones = torch.tensor([item[4] for item in items], dtype=torch.float32, device=device)

    current_batch = make_batch(observations, [config] * batch_size, online.config, device=device)
    next_batch = make_batch(next_observations, [config] * batch_size, online.config, device=device)
    q_values = online(current_batch).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        # Double DQN: online network selects the next action; target network scores it.
        next_actions = online(next_batch).argmax(dim=1)
        next_q = target(next_batch).gather(1, next_actions[:, None]).squeeze(1)
        targets = rewards + gamma * (1. - dones) * next_q
    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 1.)
    optimizer.step()
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', help='Dataset JSON; defaults to the synthetic engineering fixture')
    parser.add_argument('--media', help='Explicit media JSON, required for live mode')
    parser.add_argument('--mode', choices=('vod', 'live'), default='vod')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--replay-capacity', type=int, default=50000)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-end', type=float, default=0.05)
    parser.add_argument('--epsilon-decay-steps', type=int, default=10000)
    parser.add_argument('--learn-every', type=int, default=4)
    parser.add_argument('--target-update-steps', type=int, default=250)
    parser.add_argument('--evaluate-every', type=int, default=10)
    parser.add_argument('--stall-tolerance-s', type=float, default=0.,
                        help='Allowed rebuffer seconds above the throughput baseline on each validation trace')
    parser.add_argument('--max-rebuffer-s', type=float, default=None,
                        help='Additional absolute stall ceiling per validation episode')
    parser.add_argument('--max-startup-s', type=float, default=None,
                        help='Maximum startup delay on every validation trace')
    parser.add_argument('--rebuffer-penalty', type=float, default=4.3,
                        help='Reward penalty per second of rebuffering, identical for model and baselines')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-seed', type=int,
                        help='Synthetic dataset seed; fix it to compare training seeds')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='runs/dueling-double-dqn')
    args = parser.parse_args()
    if min(args.episodes, args.replay_capacity, args.warmup, args.batch_size,
           args.epsilon_decay_steps, args.learn_every, args.target_update_steps,
           args.evaluate_every) < 1:
        parser.error('episode, replay, warmup, batch and update counts must be positive')
    if not 0 < args.gamma <= 1 or not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        parser.error('gamma must be in (0,1] and epsilon values must satisfy 0 <= end <= start <= 1')
    if not np.isfinite(args.stall_tolerance_s) or args.stall_tolerance_s < 0:
        parser.error('--stall-tolerance-s must be finite and nonnegative')
    if args.max_rebuffer_s is not None and (not np.isfinite(args.max_rebuffer_s) or args.max_rebuffer_s < 0):
        parser.error('--max-rebuffer-s must be finite and nonnegative')
    if args.max_startup_s is not None and (not np.isfinite(args.max_startup_s) or args.max_startup_s < 0):
        parser.error('--max-startup-s must be finite and nonnegative')
    if not np.isfinite(args.rebuffer_penalty) or args.rebuffer_penalty < 0:
        parser.error('--rebuffer-penalty must be finite and nonnegative')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    data_seed = args.seed if args.data_seed is None else args.data_seed
    dataset = Dataset.load(args.dataset) if args.dataset else synthetic_dataset(seed=data_seed)
    media = Media.load(args.media) if args.media else None
    config = StreamConfig(mode=args.mode)
    weights = QoEWeights(rebuffer=args.rebuffer_penalty)
    train_env = StreamingEnv(dataset, media=media, split='train', config=config, weights=weights)
    validation_env = StreamingEnv(dataset, media=media, split='validation', config=config, weights=weights)
    validation_baseline_reports = {
        name: baseline_report(dataset, media, 'validation', config, name, weights)
        for name in ('lowest', 'highest', 'throughput')
    }
    validation_baselines = {name: baseline_summary(report)
                            for name, report in validation_baseline_reports.items()}
    validation_throughput = validation_baseline_reports['throughput']
    stall_limits_s = {item['trace']: min(float(item['metrics']['rebuffer_s']) + args.stall_tolerance_s,
                                       float('inf') if args.max_rebuffer_s is None else args.max_rebuffer_s)
                      for item in validation_throughput['episodes']}

    online = DuelingDoubleDQN().to(device)
    target = DuelingDoubleDQN(online.config).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate)
    replay = deque(maxlen=args.replay_capacity)
    global_steps = gradient_steps = 0
    best_key = None
    best_feasible_key = None
    best_feasible_validation = None
    history = []
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / 'dueling_double_dqn.npz'

    for episode_number in range(1, args.episodes + 1):
        trace_index = int(rng.integers(len(dataset.splits['train'])))
        observation, _ = train_env.reset(options={'trace_index': trace_index})
        episode_reward = 0.
        terminated = truncated = False
        while not (terminated or truncated):
            epsilon = max(args.epsilon_end, args.epsilon_start -
                          (args.epsilon_start - args.epsilon_end) * global_steps /
                          args.epsilon_decay_steps)
            if rng.random() < epsilon:
                action = int(rng.integers(len(observation['candidates'])))
            else:
                batch = make_batch([observation], [config], online.config, device=device)
                with torch.no_grad():
                    action = int(online(batch)[0].argmax().item())

            current = copy_observation(observation)
            next_observation, reward, terminated, truncated, _info = train_env.step(action)
            done = bool(terminated or truncated)
            # Terminal observations have no candidates. Use the current observation
            # as a masked-out placeholder; terminal targets do not bootstrap from it.
            replay.append((current, action, float(reward),
                           current if done else copy_observation(next_observation), done))
            observation = next_observation
            episode_reward += reward
            global_steps += 1

            if len(replay) >= max(args.warmup, args.batch_size) and global_steps % args.learn_every == 0:
                online.train()
                learn_batch(online, target, optimizer, replay, args.batch_size,
                            config, device, args.gamma)
                gradient_steps += 1
                if gradient_steps % args.target_update_steps == 0:
                    target.load_state_dict(online.state_dict())
                    target.eval()

        if episode_number % args.evaluate_every == 0 or episode_number == args.episodes:
            online.eval()
            validation = evaluate_policy(validation_env, online, stall_limits_s=stall_limits_s)
            record = {'episode': episode_number, 'environment_steps': global_steps,
                      'gradient_steps': gradient_steps, 'epsilon': epsilon,
                      'train_reward': float(episode_reward), 'completed': bool(terminated),
                      'checkpoint_eligible': checkpoint_eligible(validation, args.max_startup_s),
                      'validation': validation}
            history.append(record)
            key = (validation['completion_rate'], validation['mean_reward'])
            if best_key is None or key > best_key:
                best_key = key
                save_trained_model(online, checkpoint, training_metadata={
                    'algorithm': 'Dueling Double DQN', 'seed': args.seed,
                    'data_seed': data_seed, 'episode': episode_number,
                    'environment_steps': global_steps, 'gradient_steps': gradient_steps,
                    'training_dataset_sha256': dataset.fingerprint,
                    'training_data': 'synthetic engineering fixture' if not args.dataset else str(args.dataset),
                    'validation_split': 'validation', 'gamma': args.gamma,
                    'learning_rate': args.learning_rate, 'batch_size': args.batch_size,
                    'replay_capacity': args.replay_capacity, 'warmup': args.warmup,
                    'best_validation': validation,
                    'config': asdict(config), 'weights': asdict(weights),
                    'max_startup_s': args.max_startup_s, 'max_rebuffer_s': args.max_rebuffer_s,
                    'stall_constraint': {'baseline': 'throughput per validation trace',
                                         'tolerance_s': args.stall_tolerance_s,
                                         'limits_s': stall_limits_s,
                                         'met': validation['stall_constraint_met']}})
            if record['checkpoint_eligible'] and (best_feasible_key is None or key > best_feasible_key):
                best_feasible_key = key
                best_feasible_validation = validation
                save_trained_model(online, output_dir / 'dueling_double_dqn_feasible.npz', training_metadata={
                    'algorithm': 'Dueling Double DQN', 'seed': args.seed,
                    'data_seed': data_seed, 'episode': episode_number,
                    'environment_steps': global_steps, 'gradient_steps': gradient_steps,
                    'training_dataset_sha256': dataset.fingerprint,
                    'training_data': 'synthetic engineering fixture' if not args.dataset else str(args.dataset),
                    'validation_split': 'validation', 'gamma': args.gamma,
                    'learning_rate': args.learning_rate, 'batch_size': args.batch_size,
                    'replay_capacity': args.replay_capacity, 'warmup': args.warmup,
                    'best_validation': validation,
                    'config': asdict(config), 'weights': asdict(weights),
                    'max_startup_s': args.max_startup_s, 'max_rebuffer_s': args.max_rebuffer_s,
                    'stall_constraint': {'baseline': 'throughput per validation trace',
                                         'tolerance_s': args.stall_tolerance_s,
                                         'limits_s': stall_limits_s, 'met': True}})
            print(json.dumps({**record, 'validation': {k: v for k, v in validation.items() if k != 'episodes'}}), flush=True)

    selected_checkpoint = output_dir / 'dueling_double_dqn_feasible.npz' if best_feasible_key is not None else None
    diagnostic_checkpoint = selected_checkpoint or checkpoint
    best_model, metadata = load_model(diagnostic_checkpoint, device=device)
    validation_report = evaluate_policy(validation_env, best_model, stall_limits_s=stall_limits_s)
    test_env = StreamingEnv(dataset, media=media, split='test', config=config, weights=weights)
    test_report = evaluate_policy(test_env, best_model)
    test_baselines = {name: baseline_summary(baseline_report(dataset, media, 'test', config, name, weights))
                      for name in ('lowest', 'highest', 'throughput')}
    result = {'training': {'algorithm': 'Dueling Double DQN', 'seed': args.seed,
                           'data_seed': data_seed, 'episodes': args.episodes,
                           'environment_steps': global_steps, 'gradient_steps': gradient_steps,
                           'dataset_sha256': dataset.fingerprint,
                           'synthetic_fixture': not bool(args.dataset),
                           'config': asdict(config), 'weights': asdict(weights),
                           'max_rebuffer_s': args.max_rebuffer_s,
                           'max_startup_s': args.max_startup_s,
                           'stall_constraint': {'baseline': 'throughput per validation trace',
                                                'tolerance_s': args.stall_tolerance_s,
                                                'limits_s': stall_limits_s,
                                                'feasible_checkpoint_found': best_feasible_key is not None}},
              'accepted_checkpoint': None if selected_checkpoint is None else str(selected_checkpoint),
              'diagnostic_checkpoint': str(diagnostic_checkpoint), 'checkpoint_metadata': metadata,
              'best_validation': validation_report, 'validation_baselines': validation_baselines,
              'best_feasible_validation': best_feasible_validation,
              'heldout_test': test_report, 'test_baselines': test_baselines,
              'history': history}
    metrics_path = output_dir / 'training_metrics.json'
    metrics_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'candidate_checkpoint': str(checkpoint), 'metrics': str(metrics_path),
                      'accepted_checkpoint': None if selected_checkpoint is None else str(selected_checkpoint),
                      'stall_constraint_met': validation_report['stall_constraint_met'],
                      'best_validation': {key: value for key, value in validation_report.items() if key != 'episodes'},
                      'heldout_test': {key: value for key, value in test_report.items() if key != 'episodes'}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
