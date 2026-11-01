"""Train and select a p20 next-segment throughput predictor on v4 train/validation only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from abr.data import Dataset
from abr.env import QoEWeights, StreamConfig, StreamingEnv
from abr.models.safety import ActionSafety, allowed_actions
from abr.recovery import RecoveryConfig
from abr.throughput_predictor import (FEATURE_COUNT, QuantileNetwork, features,
                                      future_mean_capacity)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'experiments/configs/throughput_predictor_v1.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(dataset, split, protocol):
    cfg = protocol['evaluation']
    env = StreamingEnv(dataset, split=split, history=protocol['training']['history_length'],
                       config=StreamConfig(**cfg['stream_config']),
                       recovery=RecoveryConfig(**cfg['recovery']),
                       weights=QoEWeights(**cfg['qoe_weights']))
    safety = ActionSafety(**cfg['safety'])
    samples, labels = [], []
    for trace_index, trace in enumerate(dataset.splits[split]):
        obs, _ = env.reset(seed=0, options={'trace_index': trace_index})
        while True:
            duration = float(obs['candidates'][0, 2])
            target = future_mean_capacity(trace, env.network.time_s + trace.rtt_s,
                                          duration)
            if target is not None:
                samples.append(features(obs))
                labels.append(target)
            mask = allowed_actions(obs, safety)
            action = int(np.flatnonzero(mask)[-1])
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    if not samples:
        raise RuntimeError(f'No usable forecast samples for split {split}')
    return np.asarray(samples, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def pinball(prediction, target, quantile):
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def main():
    protocol = json.loads(PROTOCOL.read_text())
    if sha256(ROOT / protocol['scenario_spec']) != protocol['scenario_spec_sha256']:
        raise ValueError('Frozen scenario spec hash mismatch')
    torch.set_num_threads(1)
    seed = int(protocol['training']['seed'])
    np.random.seed(seed)
    torch.manual_seed(seed)
    root = ROOT / protocol['dataset_root']
    dataset = Dataset.load(root / 'envivio.json')
    fingerprint = protocol['dataset_fingerprints']['envivio']
    if dataset.fingerprint != fingerprint:
        raise ValueError('Training dataset fingerprint does not match frozen protocol')

    x_train, y_train_mbps = collect(dataset, 'train', protocol)
    x_val, y_val_mbps = collect(dataset, 'validation', protocol)
    quantile = float(protocol['training']['quantile'])
    y_train = np.log1p(y_train_mbps).astype(np.float32)
    y_val = torch.from_numpy(np.log1p(y_val_mbps).astype(np.float32))
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                              batch_size=protocol['training']['batch_size'], shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    model = QuantileNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['training']['learning_rate'])
    best_state, best_loss, best_epoch, stale = None, float('inf'), None, 0
    history = []
    max_epochs = int(protocol['training']['max_epochs'])
    patience = int(protocol['training']['patience'])
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = pinball(model(batch_x), batch_y, quantile)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prediction = model(torch.from_numpy(x_val))
            val_loss = float(pinball(val_prediction, y_val, quantile))
        history.append({'epoch': epoch, 'validation_pinball_log_mbps': val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    artifact = ROOT / protocol['artifact']
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'schema_version': 1, 'state_dict': model.state_dict(),
                'training_dataset_fingerprint': dataset.fingerprint,
                'training_split': 'train', 'selection_split': 'validation',
                'selected_epoch': best_epoch, 'quantile': quantile,
                'feature_count': FEATURE_COUNT, 'seed': seed}, artifact)
    with torch.no_grad():
        prediction_mbps = np.expm1(model(torch.from_numpy(x_val)).numpy())
    harmonic_mbps = []
    # Validation labels were collected from a fixed highest-allowed trajectory.
    # Recreate observations to score the same baseline forecaster on those samples.
    # Dataset-derived observation sequence is saved below by the collection companion.
    report = {
        'protocol_id': protocol['protocol_id'], 'protocol_sha256': sha256(PROTOCOL),
        'dataset_fingerprint': dataset.fingerprint,
        'training_dataset': str((root / 'envivio.json').relative_to(ROOT)),
        'training_samples': int(len(y_train_mbps)), 'validation_samples': int(len(y_val_mbps)),
        'selected_epoch': best_epoch, 'validation_pinball_log_mbps': best_loss,
        'validation_predictor_mae_mbps': float(np.mean(np.abs(prediction_mbps - y_val_mbps))),
        'validation_predictor_p20_coverage': float(np.mean(y_val_mbps <= prediction_mbps)),
        'epochs': history, 'artifact': str(artifact.relative_to(ROOT)),
        'artifact_sha256': sha256(artifact),
        'note': 'Forecast output uses log1p Mbps. Selection used validation pinball loss only.'}
    out = artifact.parent / 'training.json'
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('training_samples', 'validation_samples',
                                             'selected_epoch', 'validation_pinball_log_mbps',
                                             'validation_predictor_mae_mbps',
                                             'validation_predictor_p20_coverage',
                                             'artifact_sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
