"""Analytical telemetry fixtures. These tests do not run a real browser."""
import json
from pathlib import Path
import shutil
import subprocess
import pytest
from abr.benchmark import evaluate
from abr.validate_player import compare, player_metrics, prepare_case, DEFAULT_PROTOCOL
from .test_env import env_for


@pytest.fixture
def validation_case():
    report = evaluate(env_for(), baseline='lowest')
    return prepare_case(report)


def capture_for(case):
    # Analytical reference: startup 1s, four media seconds, no stalls.
    return {'schema_version': 1, 'origin': 'browser_capture', 'identity': case['identity'],
        'completed': True, 'issues': [], 'end_s': 5., 'network_setup_evidence': 'test-fixture-not-real-emulation',
        'events': [{'type': 'start', 't_s': 0., 'media_time_s': 0.},
                   {'type': 'playing', 't_s': 1., 'media_time_s': 0.},
                   {'type': 'ended', 't_s': 5., 'media_time_s': 4.}],
        'samples': []}


def test_comparison_accepts_matching_metrics_but_does_not_claim_equivalence(validation_case):
    protocol = json.loads(DEFAULT_PROTOCOL.read_text())
    result = compare(validation_case, capture_for(validation_case), protocol)
    assert result['status'] == 'within_tolerances'
    assert result['checks']['played_s']['browser'] == 4
    assert result['real_player_equivalence_established'] is False
    assert result['protocol_status'] == 'draft'
    json.dumps(result, allow_nan=False)


def test_different_metrics_fail_numerical_check(validation_case):
    capture = capture_for(validation_case)
    capture['events'][-1]['t_s'] = capture['end_s'] = 6.
    result = compare(validation_case, capture, json.loads(DEFAULT_PROTOCOL.read_text()))
    assert result['status'] == 'different'
    assert not result['checks']['wall_time_s']['within_tolerance']


def test_stalls_count_once_and_startup_waiting_is_separate(validation_case):
    c = capture_for(validation_case)
    c['end_s'] = 8.
    c['events'] = [
        {'type': 'start', 't_s': 0., 'media_time_s': 0.},
        {'type': 'waiting', 't_s': .5, 'media_time_s': 0.},
        {'type': 'playing', 't_s': 1., 'media_time_s': 0.},
        {'type': 'waiting', 't_s': 2., 'media_time_s': 1.},
        {'type': 'waiting', 't_s': 3., 'media_time_s': 1.},
        {'type': 'playing', 't_s': 5., 'media_time_s': 1.},
        {'type': 'ended', 't_s': 8., 'media_time_s': 4.}]
    m = player_metrics(c, 'vod')
    assert m['startup_s'] == 1 and m['rebuffer_s'] == 3 and m['stall_events'] == 1


@pytest.mark.parametrize('change', ['identity', 'incomplete', 'seeking', 'pause', 'clock', 'evidence', 'nan', 'offset'])
def test_invalid_captures_rejected(validation_case, change):
    c = capture_for(validation_case)
    if change == 'identity':
        c['identity'] = {**c['identity'], 'trace_sha256': 'different'}
    elif change == 'incomplete':
        c['completed'] = False
    elif change == 'seeking':
        c['events'].insert(2, {'type': 'seeking', 't_s': 2., 'media_time_s': 1.})
    elif change == 'pause':
        c['events'].insert(2, {'type': 'pause', 't_s': 2., 'media_time_s': 1.})
    elif change == 'clock':
        c['events'][1]['t_s'] = 10
    elif change == 'evidence':
        c['network_setup_evidence'] = ''
    elif change == 'nan':
        c['end_s'] = float('nan')
    else:
        c['events'][1]['media_time_s'] = 2
    with pytest.raises(ValueError):
        compare(validation_case, c, json.loads(DEFAULT_PROTOCOL.read_text()))


def test_prepare_rejects_final_test_and_mismatched_policy():
    report = evaluate(env_for(), baseline='highest')
    with pytest.raises(ValueError, match='fixed-lowest'):
        prepare_case(report)
    report = evaluate(env_for(), baseline='lowest')
    with pytest.raises(ValueError, match='nonnegative'):
        prepare_case(report, -1)
    report['provenance']['split'] = 'test'
    with pytest.raises(ValueError, match='reserved final test'):
        prepare_case(report)


def test_tampered_case_config_rejected(validation_case):
    c = capture_for(validation_case)
    validation_case['config']['startup_s'] = 42
    with pytest.raises(ValueError, match='config differs'):
        compare(validation_case, c, json.loads(DEFAULT_PROTOCOL.read_text()))


def test_live_latency_wall_time_includes_stalls_and_requires_dense_samples(validation_case):
    c = capture_for(validation_case)
    c['samples'] = [{'t_s': t/10, 'media_time_s': max(0, t/10-1), 'live_latency_s': 2.}
                    for t in range(10,51)]
    assert player_metrics(c, 'live')['mean_live_latency_s'] == pytest.approx(2)
    c['samples'] = c['samples'][::10]
    with pytest.raises(ValueError, match='gap'):
        player_metrics(c, 'live')
    c['samples'] = [{'t_s': 1., 'media_time_s': 0., 'live_latency_s': None},
                    {'t_s': 5., 'media_time_s': 4., 'live_latency_s': None}]
    with pytest.raises(ValueError, match='live latency'):
        player_metrics(c, 'live')


def test_command_line_reports_mismatch_with_nonzero_exit(validation_case, tmp_path):
    import sys
    case, capture, out = (tmp_path/x for x in ('case.json','capture.json','result.json'))
    case.write_text(json.dumps(validation_case))
    c = capture_for(validation_case)
    c['events'][-1]['t_s'] = c['end_s'] = 6.
    capture.write_text(json.dumps(c))
    result = subprocess.run([sys.executable, '-m', 'abr.validate_player', 'compare',
        '--case', str(case), '--capture', str(capture), '--output', str(out)], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert json.loads(out.read_text())['status'] == 'different'


def test_browser_recorder_with_deterministic_fake_media(tmp_path):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node required for recorder unit test')
    module = Path(__file__).resolve().parent.parent/'experiments/validation/recorder.mjs'
    script = tmp_path/'recorder-test.mjs'
    script.write_text('''import assert from 'node:assert/strict';
import {PlaybackRecorder} from ''' + json.dumps(module.as_uri()) + ''';
let time = 0;
const listeners = new Map();
const video = {currentTime:0, playbackRate:1, ended:false,
  buffered:{length:1,start:()=>0,end:()=>4},
  addEventListener:(type,fn)=>listeners.set(type,fn),
  removeEventListener:type=>listeners.delete(type)};
const recorder = new PlaybackRecorder(video,{mode:'vod'}, {clock:()=>time});
recorder.start();
assert.throws(()=>recorder.export());
time=1; listeners.get('playing')();
time=2; video.currentTime=1; listeners.get('waiting')();
time=3; listeners.get('playing')();
time=6; video.currentTime=4; video.ended=true; listeners.get('ended')();
const result=recorder.export();
assert.equal(result.completed,true);
assert.equal(result.end_s,6);
assert.equal(result.events.at(-1).type,'ended');
assert.equal(result.samples.at(-1).buffer_s,0);
assert.equal(result.samples[0].live_latency_s,null);
assert.deepEqual(result.issues,[]);
assert.equal(listeners.size,0);
console.log(JSON.stringify(result));
''')
    result = subprocess.run([node, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    capture = json.loads(result.stdout)
    capture['network_setup_evidence'] = 'synthetic event test'
    m = player_metrics(capture, 'vod')
    assert m['startup_s'] == 1 and m['rebuffer_s'] == 1 and m['stall_events'] == 1


def test_capture_page_module_parses_without_execution():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node required for JavaScript syntax check')
    html = (Path(__file__).resolve().parent.parent/'experiments/validation/player.html').read_text()
    script = html.split('<script type="module">', 1)[1].split('</script>', 1)[0]
    result = subprocess.run([node, '--input-type=module', '--check'], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
