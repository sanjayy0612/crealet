"""Train CandidateTransformer with PPO on the trace-replay environment.

Example:
    python -m abr.models.train_ppo --updates 10 --episodes-per-update 4
    python -m abr.models.train_ppo --dataset data/audited.json --updates 50
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from ..benchmark import evaluate
from ..data import Dataset, synthetic_dataset
from ..env import StreamingEnv, StreamConfig
from ..media import Media
from .candidate_transformer import CandidateTransformer
from .checkpoint import load_model, save_trained_model
from .inputs import make_batch
from .policy import DecisionPolicy


def collect_episode(env, model, rng, *, device):
    trace_index = int(rng.integers(len(env.dataset.splits['train'])))
    observation, _ = env.reset(options={'trace_index': trace_index})
    observations, actions, old_log_probs, values, rewards = [], [], [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        batch = make_batch([observation], [env.config], model.config, device=device)
        with torch.no_grad():
            logits, value = model(batch)
            distribution = Categorical(logits=logits[0])
            action = distribution.sample()
        observations.append({key: np.array(item, copy=True) for key, item in observation.items()})
        actions.append(int(action.item()))
        old_log_probs.append(float(distribution.log_prob(action).item()))
        values.append(float(value.item()))
        observation, reward, terminated, truncated, _ = env.step(actions[-1])
        rewards.append(float(reward))
    return {'observations': observations, 'actions': actions, 'log_probs': old_log_probs,
            'values': values, 'rewards': rewards, 'completed': bool(terminated)}


def advantages_for(episodes, *, gamma, gae_lambda):
    advantages, returns = [], []
    for episode in episodes:
        values = episode['values']
        rewards = episode['rewards']
        gae = 0.
        episode_advantages = [0.] * len(rewards)
        for step in range(len(rewards) - 1, -1, -1):
            next_value = values[step + 1] if step + 1 < len(values) else 0.
            delta = rewards[step] + gamma * next_value - values[step]
            gae = delta + gamma * gae_lambda * gae
            episode_advantages[step] = gae
        advantages.extend(episode_advantages)
        returns.extend(a + v for a, v in zip(episode_advantages, values))
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def update(model, optimizer, episodes, *, device, gamma, gae_lambda, epochs,
           minibatch_size, clip_ratio, value_weight, entropy_weight):
    observations = [observation for ep in episodes for observation in ep['observations']]
    actions = torch.tensor([action for ep in episodes for action in ep['actions']], device=device)
    old_log_probs = torch.tensor([p for ep in episodes for p in ep['log_probs']], device=device)
    advantages, returns = advantages_for(episodes, gamma=gamma, gae_lambda=gae_lambda)
    advantages, returns = advantages.to(device), returns.to(device)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch = make_batch(observations, [episodes[0]['env_config']] * len(observations), model.config, device=device)
    count = len(actions)
    totals = {'policy_loss': 0., 'value_loss': 0., 'entropy': 0., 'updates': 0}
    for _ in range(epochs):
        order = torch.randperm(count, device=device)
        for indices in order.split(minibatch_size):
            logits, predicted_values = model({key: value[indices] for key, value in batch.items()})
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[indices])
            ratio = (new_log_probs - old_log_probs[indices]).exp()
            unclipped = ratio * advantages[indices]
            clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            # Returns are large cumulative sums; normalize the critic loss so it
            # does not dominate the policy gradient and clip its shared encoder.
            value_scale = returns.std(unbiased=False).clamp_min(1.)
            value_loss = torch.nn.functional.mse_loss(
                predicted_values / value_scale, returns[indices] / value_scale)
            entropy = dist.entropy().mean()
            loss = policy_loss + value_weight * value_loss - entropy_weight * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            totals['policy_loss'] += float(policy_loss.detach())
            totals['value_loss'] += float(value_loss.detach())
            totals['entropy'] += float(entropy.detach())
            totals['updates'] += 1
    n = max(totals.pop('updates'), 1)
    return {key: value / n for key, value in totals.items()}, count


def summarize_evaluation(env, policy):
    returns, completions, stalls, bitrates, latencies = [], [], [], [], []
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
        returns.append(total_reward)
        completions.append(float(terminated))
        stalls.append(metrics['rebuffer_s'])
        bitrates.append(metrics['mean_played_bitrate_kbps'])
        latencies.append(metrics['mean_live_latency_s'])
    return {'mean_reward': float(np.mean(returns)),
            'completion_rate': float(np.mean(completions)),
            'mean_rebuffer_s': float(np.mean(stalls)),
            'mean_played_bitrate_kbps': float(np.mean(bitrates)),
            'mean_live_latency_s': float(np.mean(latencies))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', help='Dataset JSON; defaults to the synthetic engineering fixture')
    parser.add_argument('--media', help='Explicit media JSON, required for live mode')
    parser.add_argument('--mode', choices=('vod', 'live'), default='vod')
    parser.add_argument('--updates', type=int, default=10)
    parser.add_argument('--episodes-per-update', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--minibatch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Per-decision discount. 1.0 optimizes total reward for finite streams.')
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--entropy-weight', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-seed', type=int,
                        help='Synthetic dataset seed; fix this to compare different PPO seeds on the same traces')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='runs/ppo-synthetic')
    args = parser.parse_args()
    if min(args.updates, args.episodes_per_update, args.epochs, args.minibatch_size) < 1:
        parser.error('update, episode, epoch and minibatch counts must be positive')
    if not 0 < args.gamma <= 1 or not 0 < args.gae_lambda <= 1:
        parser.error('gamma and gae-lambda must be in (0, 1]')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    data_seed = args.seed if args.data_seed is None else args.data_seed
    dataset = Dataset.load(args.dataset) if args.dataset else synthetic_dataset(seed=data_seed)
    media = Media.load(args.media) if args.media else None
    config = StreamConfig(mode=args.mode)
    train_env = StreamingEnv(dataset, media=media, split='train', config=config)
    validation_env = StreamingEnv(dataset, media=media, split='validation', config=config)
    model = CandidateTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    policy = DecisionPolicy(model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_key = None
    best_checkpoint = output_dir / 'candidate_transformer_ppo.npz'

    for update_number in range(1, args.updates + 1):
        model.train()
        episodes = []
        for _ in range(args.episodes_per_update):
            episode = collect_episode(train_env, model, rng, device=device)
            episode['env_config'] = config
            episodes.append(episode)
        update_metrics, transitions = update(model, optimizer, episodes, device=device,
            gamma=args.gamma, gae_lambda=args.gae_lambda, epochs=args.epochs,
            minibatch_size=args.minibatch_size, clip_ratio=0.2, value_weight=0.5,
            entropy_weight=args.entropy_weight)
        model.eval()
        validation = summarize_evaluation(validation_env, policy)
        train_summary = {'mean_reward': float(np.mean([sum(ep['rewards']) for ep in episodes])),
                         'completion_rate': float(np.mean([ep['completed'] for ep in episodes]))}
        record = {'update': update_number, 'transitions': transitions,
                  'train': train_summary, 'validation': validation, **update_metrics}
        history.append(record)
        # Prefer completing streams; use reward to break ties.
        key = (validation['completion_rate'], validation['mean_reward'])
        if best_key is None or key > best_key:
            best_key = key
            metadata = save_trained_model(model, best_checkpoint, training_metadata={
                'algorithm': 'PPO', 'seed': args.seed, 'data_seed': data_seed,
                'updates': update_number,
                'training_dataset_sha256': dataset.fingerprint,
                'training_data': 'synthetic engineering fixture' if not args.dataset else str(args.dataset),
                'validation_split': 'validation', 'gamma': args.gamma,
                'gae_lambda': args.gae_lambda, 'learning_rate': args.learning_rate,
                'episodes_per_update': args.episodes_per_update, 'transitions_seen':
                    sum(item['transitions'] for item in history),
                'best_validation': validation})
        else:
            metadata = None
        print(json.dumps(record), flush=True)

    result = {'training': {'algorithm': 'PPO', 'seed': args.seed, 'data_seed': data_seed,
                           'updates': args.updates,
                           'episodes_per_update': args.episodes_per_update,
                           'total_transitions': sum(item['transitions'] for item in history),
                           'dataset_sha256': dataset.fingerprint,
                           'synthetic_fixture': not bool(args.dataset)},
              'best_checkpoint': str(best_checkpoint), 'best_validation': history[
                  max(range(len(history)), key=lambda i: (history[i]['validation']['completion_rate'],
                                                           history[i]['validation']['mean_reward']))]['validation'],
              'final_validation': history[-1]['validation'], 'updates': history}
    (output_dir / 'training_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    model, checkpoint_metadata = load_model(best_checkpoint)
    final_env = StreamingEnv(dataset, media=media, split='validation', config=config)
    baseline_env = StreamingEnv(dataset, media=media, split='validation', config=config)
    result['best_validation'] = summarize_evaluation(final_env, DecisionPolicy(model))
    result['validation_baselines'] = {}
    for name in ('lowest', 'highest', 'throughput'):
        baseline_report = evaluate(baseline_env, baseline=name)
        result['validation_baselines'][name] = {
            'mean_reward': float(np.mean([ep['reward'] for ep in baseline_report['episodes']])),
            'completion_rate': baseline_report['completion_rate']}
    result['checkpoint_metadata'] = checkpoint_metadata
    (output_dir / 'training_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'checkpoint': str(best_checkpoint), 'metrics': str(output_dir / 'training_metrics.json'),
                      'best_validation': result['best_validation'],
                      'validation_baselines': result['validation_baselines']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
