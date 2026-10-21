"""Architecture and integration tests, not evidence of trained ABR quality."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

torch = pytest.importorskip('torch', reason='Install requirements-model.txt for model tests')
from abr.models import CandidateScorer, ModelConfig, make_batch, load_model, save_initial_model
from abr.models.policy import DecisionPolicy
from abr.env import StreamingEnv, StreamConfig
from abr.benchmark import evaluate
from .test_env import fixture


def model_inputs(kind='gru', *, history=8, mode='vod'):
    data, media = fixture()
    env = StreamingEnv(data, media=media, config=StreamConfig(mode=mode), history=history)
    obs, _ = env.reset(seed=4)
    model = CandidateScorer(ModelConfig(kind=kind, history_length=history))
    return model, obs, env


@pytest.mark.parametrize('kind,expected', [('dense',10849), ('gru',14305)])
def test_shapes_and_parameter_budget(kind, expected):
    model, obs, env = model_inputs(kind)
    batch = make_batch([obs], [env.config], model.config)
    scores = model(batch)
    assert scores.shape == (1,2) and torch.isfinite(scores).all()
    assert model.parameter_count == expected


@pytest.mark.parametrize('kind', ['dense','gru'])
def test_candidate_order_equivariance_and_shared_scoring(kind):
    model, obs, env = model_inputs(kind)
    original = model(make_batch([obs], [env.config], model.config)).detach()
    reverse = {**obs, 'candidates': obs['candidates'][::-1].copy()}
    reordered = model(make_batch([reverse], [env.config], model.config)).detach()
    torch.testing.assert_close(original.flip(1), reordered, rtol=1e-5, atol=1e-6)
    duplicate = {**obs, 'candidates': np.repeat(obs['candidates'][:1], 3, axis=0)}
    scores = model(make_batch([duplicate], [env.config], model.config))
    torch.testing.assert_close(scores[:,0], scores[:,1])
    torch.testing.assert_close(scores[:,1], scores[:,2])


@pytest.mark.parametrize('kind', ['dense','gru'])
def test_variable_candidates_batch_padding_is_ignored(kind):
    model, obs, env = model_inputs(kind)
    more = {**obs, 'candidates': np.array([[.25,.1,2],[.5,.2,2],[1,.4,2],[2,.8,2]], dtype=np.float32)}
    batch = make_batch([obs, more], [env.config, env.config], model.config)
    values = model(batch).detach()
    assert values.shape == (2,4) and torch.isneginf(values[0,2:]).all()
    alone = model(make_batch([obs], [env.config], model.config)).detach()
    torch.testing.assert_close(values[0,:2], alone[0], rtol=1e-5, atol=1e-6)
    batch['candidates'][0,2:] = 9999
    torch.testing.assert_close(model(batch)[0,:2], alone[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('kind', ['dense','gru'])
def test_masked_history_values_cannot_affect_decision(kind):
    model, obs, env = model_inputs(kind)
    a = model(make_batch([obs], [env.config], model.config))
    obs['history'][:,:4] = 9999
    b = model(make_batch([obs], [env.config], model.config))
    torch.testing.assert_close(a,b)


def test_gru_uses_order_of_valid_history_but_not_padding_updates():
    torch.manual_seed(3)
    model, obs, env = model_inputs()
    obs['history'][-2:] = [[.5, .2, 0, .01, 1], [4, .1, 0, .01, 1]]
    batch = make_batch([obs], [env.config], model.config)
    a = model(batch)
    # With no persistent state, moving the same valid sequence across padding is invariant.
    moved = {k:v.clone() for k,v in batch.items()}
    moved['history'].zero_()
    moved['history'][:,:2] = batch['history'][:,-2:]
    torch.testing.assert_close(a, model(moved))
    reversed_history = {k:v.clone() for k,v in batch.items()}
    reversed_history['history'][:,-2:] = batch['history'][:,-2:].flip(1)
    assert not torch.allclose(a, model(reversed_history), rtol=1e-6, atol=1e-8)


def test_public_context_is_explicit_and_future_network_is_absent():
    model, obs, env = model_inputs()
    live = replace(env.config, mode='live', target_latency_s=1.5)
    a = make_batch([obs], [env.config], model.config)
    b = make_batch([obs], [live], model.config)
    assert a['player'].shape == (1,15)
    assert a['player'][0,5] == 0 and b['player'][0,5] == 1
    assert b['player'][0,6] == 1.5
    assert set(a) == {'history','player','candidates','candidate_mask'}


@pytest.mark.parametrize('change', ['nan','negative','history','mask','terminal','remaining'])
def test_invalid_inputs_rejected(change):
    model, obs, env = model_inputs()
    if change == 'nan': obs['state'][0] = np.nan
    elif change == 'negative': obs['candidates'][0,0] = -1
    elif change == 'history': obs['history'] = obs['history'][:2]
    elif change == 'mask': obs['history'][0,4] = .5
    elif change == 'terminal': obs['candidates'][:] = 0
    else: obs['state'][3] = 2
    with pytest.raises(ValueError):
        make_batch([obs], [env.config], model.config)


def test_all_masked_candidates_rejected():
    model, obs, env = model_inputs()
    batch = make_batch([obs], [env.config], model.config)
    batch['candidate_mask'].zero_()
    with pytest.raises(ValueError, match='at least one'):
        model(batch)


@pytest.mark.parametrize('kind', ['dense','gru'])
def test_gradients_reach_history_state_and_shared_scorer(kind):
    torch.manual_seed(6)
    model, obs, env = model_inputs(kind)
    obs, *_ = env.step(0)
    batch = make_batch([obs], [env.config], model.config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Artificial regression target tests autograd only; it is not an ABR training run.
    initial = {name:p.detach().clone() for name,p in model.named_parameters()}
    loss = (model(batch) - torch.tensor([[1.,-1.]])).square().mean()
    loss.backward()
    for module in (model.history_encoder, model.state_encoder, model.fusion, model.scorer):
        grads = [p.grad for p in module.parameters()]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    optimizer.step()
    assert any(not torch.equal(initial[n],p) for n,p in model.named_parameters())


@pytest.mark.parametrize('kind', ['dense','gru'])
def test_checkpoint_roundtrip_safe_types_and_random_state(kind, tmp_path):
    torch.manual_seed(5)
    model, obs, env = model_inputs(kind)
    path = tmp_path/'checkpoint.npz'
    meta = save_initial_model(model, path, seed=5)
    before = torch.get_rng_state().clone()
    restored, loaded = load_model(path)
    assert torch.equal(before, torch.get_rng_state())
    assert loaded == meta and loaded['training_status'] == 'untrained'
    batch = make_batch([obs], [env.config], model.config)
    torch.testing.assert_close(model(batch), restored(batch), rtol=0, atol=0)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[k].dtype != object for k in archive.files)


def test_checkpoint_rejects_source_drift_and_changed_scaling(tmp_path):
    model, _, _ = model_inputs()
    path = tmp_path/'checkpoint.npz'
    save_initial_model(model,path,seed=0)
    with np.load(path,allow_pickle=False) as archive:
        arrays = {k:archive[k].copy() for k in archive.files}
    arrays['player_scale'][0] = 1
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match='fixed feature'):
        load_model(path)
    save_initial_model(model,path,seed=0)
    with np.load(path,allow_pickle=False) as archive:
        arrays = {k:archive[k].copy() for k in archive.files}
    meta = json.loads(arrays['__metadata__'].tobytes())
    meta['model_source_sha256'] = 'outdated'
    arrays['__metadata__'] = np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8)
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match='source differs'):
        load_model(path)


@pytest.mark.parametrize('kind', ['dense','gru'])
@pytest.mark.parametrize('mode', ['vod','live'])
def test_untrained_policy_runs_both_environments_and_has_no_hidden_state(kind, mode):
    torch.manual_seed(8)
    model, obs, env = model_inputs(kind,mode=mode)
    policy = DecisionPolicy(model)
    initial = policy.scores(obs,env.config)
    report = evaluate(env,baseline='model',policy=policy,
                      policy_metadata={'training_status':'untrained','purpose':'analytical integration test'})
    assert report['completion_rate'] == 1
    assert report['provenance']['policy_metadata']['training_status'] == 'untrained'
    np.testing.assert_array_equal(initial,policy.scores(obs,env.config))
    json.dumps(report,allow_nan=False)


def test_environment_import_has_no_torch_dependency():
    script = "import sys; import abr; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr


def test_cli_initialization_and_untrained_guard(tmp_path):
    artifact = tmp_path/'gru.npz'
    result = subprocess.run([sys.executable,'-m','abr.models','init','--output',str(artifact)],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    assert json.loads(result.stdout)['parameter_count'] == 14305
    result = subprocess.run([sys.executable,'-m','abr.models','rollout','--checkpoint',str(artifact),
        '--dataset','unused.json','--output',str(tmp_path/'out.json')],capture_output=True,text=True)
    assert result.returncode == 2 and 'UNTRAINED' in result.stderr
