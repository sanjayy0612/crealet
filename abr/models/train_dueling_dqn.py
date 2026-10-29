"""Train a dueling action-value model with Double DQN.

Example:
    python -m abr.models.train_dueling_dqn --episodes 200 --seed 7 --data-seed 7
"""
import argparse
from collections import deque
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from ..benchmark import evaluate
from ..data import Dataset, synthetic_dataset
from ..env import StreamingEnv, StreamConfig
from ..media import Media
from .checkpoint import load_model, save_trained_model
from .dueling_dqn import DuelingDoubleDQN
from .inputs import make_batch
from .policy import DecisionPolicy


def copy_observation(observation):
    return {key: np.array(value, copy=True) for key, value in observation.items()}


def evaluate_policy(env, model):
    policy = DecisionPolicy(model)
    rewards, completions, stalls, bitrates = [], [], [], []
    for index, _trace in enumerate(env.dataset.splits[env.split]):
        observation, _ = env.reset(options={'trace_index': index})
        total_reward = 0.
        while True:
            action = policy(observation, env.config)
            observation, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        metrics = env.metrics()
        rewards.append(total_reward)
        completions.append(float(terminated))
        stalls.append(metrics['rebuffer_s'])
        bitrates.append(metrics['mean_played_bitrate_kbps'])
    return {'mean_reward': float(np.mean(rewards)),
            'completion_rate': float(np.mean(completions)),
            'mean_rebuffer_s': float(np.mean(stalls)),
            'mean_played_bitrate_kbps': float(np.mean(bitrates))}


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
    train_env = StreamingEnv(dataset, media=media, split='train', config=config)
    validation_env = StreamingEnv(dataset, media=media, split='validation', config=config)

    online = DuelingDoubleDQN().to(device)
    target = DuelingDoubleDQN(online.config).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate)
    replay = deque(maxlen=args.replay_capacity)
    global_steps = gradient_steps = 0
    best_key = None
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
            validation = evaluate_policy(validation_env, online)
            record = {'episode': episode_number, 'environment_steps': global_steps,
                      'gradient_steps': gradient_steps, 'epsilon': epsilon,
                      'train_reward': float(episode_reward), 'completed': bool(terminated),
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
                    'best_validation': validation})
            print(json.dumps(record), flush=True)

    best_model, metadata = load_model(checkpoint, device=device)
    validation_report = evaluate_policy(validation_env, best_model)
    validation_baselines = {}
    for name in ('lowest', 'highest', 'throughput'):
        report = evaluate(StreamingEnv(dataset, media=media, split='validation', config=config), baseline=name)
        validation_baselines[name] = {
            'mean_reward': float(np.mean([item['reward'] for item in report['episodes']])),
            'completion_rate': report['completion_rate']}
    result = {'training': {'algorithm': 'Dueling Double DQN', 'seed': args.seed,
                           'data_seed': data_seed, 'episodes': args.episodes,
                           'environment_steps': global_steps, 'gradient_steps': gradient_steps,
                           'dataset_sha256': dataset.fingerprint,
                           'synthetic_fixture': not bool(args.dataset)},
              'best_checkpoint': str(checkpoint), 'checkpoint_metadata': metadata,
              'best_validation': validation_report, 'validation_baselines': validation_baselines,
              'history': history}
    metrics_path = output_dir / 'training_metrics.json'
    metrics_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'checkpoint': str(checkpoint), 'metrics': str(metrics_path),
                      'best_validation': validation_report,
                      'validation_baselines': validation_baselines}, indent=2), flush=True)


if __name__ == '__main__':
    main()
