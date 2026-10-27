"""Structural and simulator checks; random weights are not a learned ABR policy."""
from dataclasses import replace
import json
import subprocess
import sys
import numpy as np
import pytest

torch = pytest.importorskip('torch', reason='Install requirements-model.txt for model tests')
from abr.models import (CandidateTransformer, CandidateTransformerConfig, make_batch,
                        load_model, save_initial_model)
from abr.models.checkpoint import source_hash
from abr.models.policy import DecisionPolicy
from abr.env import StreamingEnv, StreamConfig
from abr.benchmark import evaluate
from .test_env import fixture


def setup_model(mode='vod'):
    torch.manual_seed(17)
    data, media = fixture()
    env = StreamingEnv(data, media=media, config=StreamConfig(mode=mode))
    obs, _ = env.reset(seed=0)
    return CandidateTransformer().eval(), obs, env


def batch_for(model, obs, env):
    return make_batch([obs], [env.config], model.config)


def test_v1_architecture_and_single_candidate():
    model, obs, env = setup_model()
    assert model.parameter_count == 77826
    assert len(model.layers) == 2
    for layer in model.layers:
        assert layer.norm_first and layer.self_attn.num_heads == 4
        assert layer.linear1.out_features == 128 and layer.dropout.p == 0
    assert not torch.equal(model.layers[0].linear1.weight, model.layers[1].linear1.weight)
    obs['candidates'] = obs['candidates'][:1]
    logits, value = model(batch_for(model, obs, env))
    assert logits.shape == (1, 1) and value.shape == (1,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    assert DecisionPolicy(model)(obs, env.config) == 0


def test_candidate_permutation_equivariance_and_value_invariance():
    model, obs, env = setup_model()
    obs['candidates'] = np.array([[.25,.1,2], [.5,.2,2], [1,.4,2], [2,.8,2]], dtype=np.float32)
    obs['history'][-2:] = [[.5,.2,0,.01,1], [4,.1,0,.01,1]]
    logits, value = model(batch_for(model, obs, env))
    order = [2, 0, 3, 1]
    changed = {**obs, 'candidates': obs['candidates'][order]}
    reordered, reordered_value = model(batch_for(model, changed, env))
    torch.testing.assert_close(reordered, logits[:, order], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reordered_value, value, rtol=1e-5, atol=1e-6)
    duplicate = {**obs, 'candidates': np.repeat(obs['candidates'][:1], 3, axis=0)}
    same, _ = model(batch_for(model, duplicate, env))
    torch.testing.assert_close(same[:, 0], same[:, 1])
    torch.testing.assert_close(same[:, 1], same[:, 2])


def test_batch_padding_cannot_change_policy_or_value():
    model, obs, env = setup_model()
    more = {**obs, 'candidates': np.array([[.25,.1,2], [.5,.2,2], [1,.4,2], [2,.8,2]], dtype=np.float32)}
    alone_logits, alone_value = model(batch_for(model, obs, env))
    batch = make_batch([obs, more], [env.config, env.config], model.config)
    logits, values = model(batch)
    assert logits.shape == (2, 4) and torch.isneginf(logits[0, 2:]).all()
    torch.testing.assert_close(logits[0, :2], alone_logits[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(values[:1], alone_value, rtol=1e-5, atol=1e-6)
    # Even arbitrary low-level padding values are sanitized before projection.
    batch['candidates'][0, 2:] = float('nan')
    batch['history'][~batch['history_mask']] = float('nan')
    changed, changed_value = model(batch)
    torch.testing.assert_close(changed, logits, rtol=0, atol=0)
    torch.testing.assert_close(changed_value, values, rtol=0, atol=0)
    probs = logits.softmax(-1)
    assert torch.equal(probs[0, 2:], torch.zeros(2))


def test_history_mask_and_temporal_positions():
    model, obs, env = setup_model()
    a, av = model(batch_for(model, obs, env))
    obs['history'][:, :4] = 9999
    b, bv = model(batch_for(model, obs, env))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(av, bv, rtol=0, atol=0)
    obs['history'][-2:] = [[.5,.2,0,.01,1], [4,.1,0,.01,1]]
    ordered = model(batch_for(model, obs, env))
    obs['history'][-2:] = obs['history'][-2:][::-1].copy()
    reversed_output = model(batch_for(model, obs, env))
    assert any(not torch.allclose(x, y, rtol=1e-6, atol=1e-8)
               for x, y in zip(ordered, reversed_output))


def test_policy_and_value_gradients_reach_every_parameter():
    model, obs, env = setup_model()
    model.train()
    obs['history'][:] = [[1 + i / 2, .1 + i / 10, .02, .01, 1] for i in range(8)]
    batch = batch_for(model, obs, env)
    batch['history'].requires_grad_()
    batch['player'].requires_grad_()
    batch['candidates'].requires_grad_()
    logits, value = model(batch)
    # Synthetic targets test both computational graphs, not an ABR algorithm.
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1])) + (value - 2).square().mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    for key in ('history', 'player', 'candidates'):
        assert batch[key].grad.abs().sum() > 0


def test_checkpoint_roundtrip_and_fixed_scaling(tmp_path):
    model, obs, env = setup_model()
    # Non-default configuration must also survive serialization.
    model = CandidateTransformer(replace(model.config, hidden_size=32, layers=1)).eval()
    batch = batch_for(model, obs, env)
    path = tmp_path / 'transformer.npz'
    metadata = save_initial_model(model, path, seed=17)
    rng = torch.get_rng_state().clone()
    restored, loaded = load_model(path)
    assert torch.equal(rng, torch.get_rng_state())
    assert loaded == metadata and restored.config == model.config
    for actual, expected in zip(restored(batch), model(batch)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    arrays['candidate_scale'][0] *= 2
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='fixed feature'):
        load_model(path)


def test_recursive_source_hash_covers_gru(monkeypatch):
    from pathlib import Path
    original = Path.read_bytes
    before = source_hash()
    def changed(path):
        data = original(path)
        return data + b'\n# drift\n' if path.name == 'history.py' else data
    monkeypatch.setattr(Path, 'read_bytes', changed)
    assert before != source_hash()


@pytest.mark.parametrize('mode', ['vod', 'live'])
def test_environment_integration_and_no_persistent_state(mode):
    model, obs, env = setup_model(mode)
    batch = batch_for(model, obs, env)
    assert set(batch) == {'history', 'player', 'candidates', 'candidate_mask', 'history_mask'}
    initial = tuple(x.detach().clone() for x in model(batch))
    policy = DecisionPolicy(model)
    report = evaluate(env, baseline='model', policy=policy,
                      policy_metadata={'training_status': 'untrained'})
    assert report['completion_rate'] == 1
    assert all(0 <= step['action'] < env.action_space.n for step in report['episodes'][0]['steps'])
    for actual, expected in zip(model(batch), initial):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('change', ['empty_candidates', 'history_shape', 'mask_dtype'])
def test_invalid_low_level_masks_and_shapes(change):
    model, obs, env = setup_model()
    batch = batch_for(model, obs, env)
    if change == 'empty_candidates':
        batch['candidate_mask'].zero_()
    elif change == 'history_shape':
        batch['history_mask'] = batch['history_mask'][:, :1]
    else:
        batch['history_mask'] = batch['history_mask'].float()
    with pytest.raises(ValueError):
        model(batch)


@pytest.mark.parametrize('settings', [dict(hidden_size=63), dict(layers=0),
                                    dict(attention_heads=True), dict(dropout=float('nan'))])
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        CandidateTransformerConfig(**settings)


def test_cli_init_inspect_and_guarded_rollout(tmp_path):
    def cli(*args):
        return subprocess.run([sys.executable, '-m', 'abr.models', *map(str, args)],
                              capture_output=True, text=True)
    path = tmp_path / 'transformer.npz'
    result = cli('init', '--kind', 'candidate_transformer', '--output', path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['parameter_count'] == 77826
    inspected = cli('inspect', '--checkpoint', path)
    assert inspected.returncode == 0, inspected.stderr
    assert json.loads(inspected.stdout)['model_config']['kind'] == 'candidate_transformer'
    out = tmp_path / 'report.json'
    rejected = cli('rollout', '--checkpoint', path, '--dataset', 'unused.json', '--output', out)
    assert rejected.returncode == 2 and 'UNTRAINED' in rejected.stderr
    data, media = fixture()
    dataset, media_path = tmp_path / 'dataset.json', tmp_path / 'media.json'
    data.save(dataset)
    media.save(media_path)
    for mode in ('vod', 'live'):
        result = cli('rollout', '--checkpoint', path, '--dataset', dataset,
                     '--media', media_path, '--mode', mode, '--allow-untrained', '--output', out)
        assert result.returncode == 0, result.stderr
        report = json.loads(out.read_text())
        assert report['completion_rate'] == 1
        assert report['provenance']['policy_metadata']['training_status'] == 'untrained'
