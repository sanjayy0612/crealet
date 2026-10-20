import json
import numpy as np
import pytest
from abr.data import Dataset
from abr.trace_import import load_ghent4g, load_interval_csv, load_timestamp_mbps
from abr.benchmark import evaluate
from .test_env import fixture, env_for


def test_ghent_units_outage_and_provenance(tmp_path):
    path = tmp_path/'ghent.log'
    path.write_text('1000 1000 0 0 250000 1000\n2000 2000 0 0 0 1000\n')
    t = load_ghent4g(path, source_group='session-a', source_url='test fixture')
    np.testing.assert_array_equal(t.bandwidth_kbps, [2000, 0])
    assert t.source_group == 'session-a' and 'raw_sha256=' in t.source
    assert t.semantics == 'observed_goodput'
    path.write_text('1000 1000 0 0 1 1000\n3000 3000 0 0 1 1000\n')
    with pytest.raises(ValueError, match='Missing/overlapping'):
        load_ghent4g(path, source_group='a', source_url='fixture')


def test_canonical_csv_variable_rtt_and_roundtrip(tmp_path):
    path = tmp_path/'trace.csv'
    path.write_text('duration_s,goodput_kbps,rtt_s\n1,0,0.1\n2,2000,0.2\n')
    t = load_interval_csv(path, source_group='session', source_url='fixture')
    np.testing.assert_array_equal(t.rtt_series_s, [.1,.2])
    data, _ = fixture()
    splits = dict(data.splits)
    splits['train'] = (t,)
    data = Dataset(data.video, splits, 'fixture')
    data.save(tmp_path/'data.json')
    loaded = Dataset.load(tmp_path/'data.json')
    assert loaded.fingerprint == data.fingerprint


def test_timestamp_alignment_is_explicit(tmp_path):
    path = tmp_path/'5g.txt'
    path.write_text('0 1\n1 2\n3 0\n')
    kw = dict(source_group='session', source_url='fixture')
    a = load_timestamp_mbps(path, alignment='start', **kw)
    b = load_timestamp_mbps(path, alignment='end', **kw)
    np.testing.assert_array_equal(a.durations_s, [1,2])
    np.testing.assert_array_equal(a.bandwidth_kbps, [1000,2000])
    np.testing.assert_array_equal(b.bandwidth_kbps, [2000,0])
    with pytest.raises(ValueError):
        load_timestamp_mbps(path, alignment='auto', **kw)


@pytest.mark.parametrize('baseline', ['lowest', 'highest', 'throughput'])
def test_v2_evaluation_reproducible_with_raw_metrics(baseline):
    env = env_for()
    a = evaluate(env, baseline=baseline, seed=0)
    b = evaluate(env, baseline=baseline, seed=0)
    assert a == b
    assert json.loads(json.dumps(a, allow_nan=False)) == a
    assert a['completion_rate'] == 1
    assert a['provenance']['environment_version'] == 2
    assert a['episodes'][0]['metrics']['played_s'] == pytest.approx(4)
