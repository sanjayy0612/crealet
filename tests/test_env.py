import json
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from abr.data import Dataset, Trace, Video
from abr.env import StreamingEnv, StreamConfig
from abr.media import Media
from abr.playback import Player
from abr.transport import Replay


def fixture(*, segments=2, duration=2., sizes=None, rates=(8.,), intervals=(100.,), rtt=0., part_s=0.5):
    if sizes is None:
        sizes = np.full((segments, 2), 1000.)
    video = Video([500., 1000.], [duration] * len(sizes), sizes)
    splits = {split: (Trace(split, [*intervals[:-1], intervals[-1]+i], rates, rtt),)
              for i, split in enumerate(('train', 'validation', 'test'))}
    n = round(duration / part_s)
    media = Media(video, tuple(np.full(n, part_s) for _ in sizes),
                  tuple(np.tile(row / n, (n, 1)) for row in video.sizes_bytes), 'synthetic analytical parts')
    return Dataset(video, splits, 'analytical test fixture'), media


def env_for(**kwargs):
    config = kwargs.pop('config', StreamConfig(catchup=False))
    data, media = fixture(**kwargs)
    return StreamingEnv(data, media=media, config=config)


def finish(env, action=0):
    env.reset(seed=1)
    rows = []
    while True:
        obs, reward, term, trunc, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert reward == pytest.approx(sum(info['reward_terms'].values()))
        rows.append(info)
        if term or trunc:
            return term, trunc, rows


def test_transport_crosses_intervals_and_outages():
    r = Replay(Trace('t', [1, 2, 1], [8, 0, 16], 0))
    result = r.transfer(2000)
    assert result.complete and result.bytes_received == 2000
    assert result.elapsed_s == 3.5
    assert r.transfer(2000).bytes_received == 1000
    assert r.time_s == 4
    assert r.transfer(1).elapsed_s == 0


def test_timeout_preserves_partial_bytes():
    r = Replay(Trace('t', [10], [8], 0))
    result = r.transfer(2000, deadline_s=0.75)
    assert not result.complete
    assert result.bytes_received == 750 and result.elapsed_s == 0.75


def test_variable_rtt_and_offset():
    r = Replay(Trace('t', [1, 2], [8, 16], 0, rtt_series_s=[0.1, 0.4]), offset_s=1)
    assert r.rtt_s == 0.4
    assert r.transfer(1000).elapsed_s == 0.5
    with pytest.raises(ValueError):
        Replay(r.trace, 3)


def test_vod_drains_playback_tail_and_conserves_media():
    term, trunc, rows = finish(env_for())
    s = rows[-1]['metrics']
    assert term and not trunc
    assert s['played_s'] == pytest.approx(4)
    assert s['buffer_s'] == pytest.approx(0)
    assert s['downloaded_media_s'] == pytest.approx(4)
    assert s['startup_s'] == pytest.approx(1)
    assert s['wall_time_s'] == pytest.approx(5)
    assert s['rebuffer_s'] == 0
    assert rows[-1]['termination_reason'] == 'playback_end'
    assert rows[-1]['playback_tail_s'] == pytest.approx(3.)
    assert rows[-1]['elapsed_s'] == pytest.approx(4.)


def test_trace_end_can_be_exactly_final_download_and_tail_still_plays():
    term, trunc, rows = finish(env_for(segments=1, intervals=(1.,)))
    assert term and not trunc
    assert rows[-1]['metrics']['wall_time_s'] == pytest.approx(3.)


def test_incomplete_trace_truncates_without_counting_partial_unit_as_video():
    term, trunc, rows = finish(env_for(segments=1, intervals=(0.5,)))
    assert trunc and not term
    assert rows[0]['bytes_received'] == 500
    assert rows[0]['metrics']['downloaded_media_s'] == 0
    assert rows[0]['termination_reason'] == 'trace_end'


def test_request_timeout_ends_outage_without_looping_trace():
    cfg = StreamConfig(request_timeout_s=2)
    term, trunc, rows = finish(env_for(rates=(0.,), config=cfg))
    assert trunc and rows[-1]['termination_reason'] == 'request_timeout'
    assert rows[-1]['metrics']['startup_s'] == pytest.approx(2)


def test_live_parts_start_playback_before_segment_finishes_with_one_rtt():
    cfg = StreamConfig(mode='live', source_age_s=0, startup_s=0.5, resume_s=0.5, catchup=False)
    env = env_for(segments=1, rates=(800.,), rtt=0.1, config=cfg)
    term, trunc, rows = finish(env)
    row = rows[0]
    assert term and not trunc
    assert len(row['parts']) == 4
    assert row['rtt_s'] == pytest.approx(0.1)
    assert row['body_transfer_s'] == pytest.approx(0.01)
    assert row['request_s'] == pytest.approx(2.0025)
    assert row['source_wait_s'] == pytest.approx(1.8925)
    assert row['parts'][1]['playhead_s'] > 0
    assert row['parts'][0]['received_at_s'] == pytest.approx(0.5025)
    assert env.history[-1, 0] == pytest.approx(0.8)
    assert row['metrics']['startup_s'] == pytest.approx(0.5025)
    assert row['metrics']['mean_live_latency_s'] > 0


def test_live_requires_actual_part_table():
    data, _ = fixture()
    with pytest.raises(ValueError, match='explicit part'):
        StreamingEnv(data, config=StreamConfig(mode='live'))


def test_inconsistent_media_rejected():
    data, media = fixture()
    with pytest.raises(ValueError, match='sizes must sum'):
        Media(data.video, media.part_durations_s, tuple(x*2 for x in media.part_sizes_bytes))
    with pytest.raises(ValueError):
        StreamConfig(mode='live', known_vod_sizes=True)


def test_media_roundtrip(tmp_path):
    _, media = fixture()
    media.save(tmp_path/'media.json')
    loaded = Media.load(tmp_path/'media.json')
    np.testing.assert_array_equal(loaded.part_sizes_bytes[0], media.part_sizes_bytes[0])


def test_startup_resume_and_stall_count_are_distinct():
    p = Player(1, 1, 500)
    p.append(0.5, 500)
    p.advance(2, 1)
    assert p.stats['startup_s'] == 2 and not p.started
    p.append(0.5, 500)
    p.advance(2, 1)
    assert p.stats['played_s'] == 1 and p.stats['rebuffer_s'] == 1
    p.advance(1, 1)
    assert p.stats['stall_events'] == 1
    p.append(0.5, 500)
    p.advance(1, 1)
    assert p.stats['rebuffer_s'] == 3
    p.end()
    p.advance(0.5, 1)
    assert p.stats['played_s'] == 1.5


def test_latency_and_playback_rate_accounting():
    p = Player(0, 0, 500)
    p.append(2, 1000)
    p.advance(1, 2, edge_start=4, edge_speed=1, target_latency=3)
    assert p.stats['played_s'] == 2
    assert p.stats['latency_integral'] == pytest.approx(3.5)
    assert p.stats['latency_excess_integral'] == pytest.approx(0.5)
    assert p.stats['quality_integral'] == pytest.approx(2*np.log(2))
    assert p.stats['rate_deviation_s'] == 1


def test_size_and_capacity_do_not_leak_into_observation():
    low = env_for(sizes=[[100,200],[100,200]], rates=(1.,))
    high = env_for(sizes=[[10000,20000],[10000,20000]], rates=(10000.,))
    a, _ = low.reset(seed=4)
    b, _ = high.reset(seed=4)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_source_group_cannot_cross_splits_and_schema1_still_loads(tmp_path):
    data, _ = fixture()
    splits = dict(data.splits)
    splits['train'] = (Trace('a', [1], [1], source_group='same-session'),)
    splits['test'] = (Trace('b', [2], [2], source_group='same-session'),)
    with pytest.raises(ValueError, match='session/group'):
        Dataset(data.video, splits, 'leak')
    obj = data.to_dict()
    obj['schema_version'] = 1
    for items in obj['splits'].values():
        for t in items:
            for key in ('source', 'source_group', 'semantics', 'rtt_series_s'):
                t.pop(key)
    path = tmp_path/'old.json'
    path.write_text(json.dumps(obj))
    assert Dataset.load(path).video.sizes_bytes.shape == (2,2)


@pytest.mark.parametrize('mode', ['vod', 'live'])
def test_gym_contract(mode):
    check_env(env_for(config=StreamConfig(mode=mode)), skip_render_check=True)


@pytest.mark.parametrize('mode', ['vod', 'live'])
def test_random_rollout_invariants(mode):
    data, media = fixture(segments=20, rates=(100, 0, 900), intervals=(4, 3, 100))
    env = StreamingEnv(data, media=media, config=StreamConfig(mode=mode, max_buffer_s=4, startup_s=.5, resume_s=.5))
    env.reset(seed=5)
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs, _, term, trunc, info = env.step(int(rng.integers(2)))
        s = info['metrics']
        assert env.observation_space.contains(obs)
        assert 0 <= s['buffer_s'] <= 4 + 1e-8
        assert s['played_s'] + s['buffer_s'] == pytest.approx(s['downloaded_media_s'])
        assert .95 <= env._rate() <= 1.05
        if term or trunc:
            break
    assert term and not trunc
    assert s['played_s'] == pytest.approx(40.)


def test_live_latency_includes_stalls_after_startup():
    p = Player(0, 0, 500)
    p.append(1, 500)
    p.advance(1, 1, edge_start=3)
    p.advance(2, 1, edge_start=4)
    assert p.stats['rebuffer_s'] == 2
    assert p.stats['latency_wall_s'] == 3
    assert p.stats['latency_integral'] == pytest.approx(11)
    assert p.stats['max_live_latency_s'] == 5


def test_cap_backpressure_does_not_count_as_network_body_time():
    env = env_for(segments=10, rates=(80000.,), config=StreamConfig(max_buffer_s=3, startup_s=.5, resume_s=.5))
    term, trunc, rows = finish(env)
    assert term and not trunc
    assert sum(r['buffer_wait_s'] for r in rows) > 10
    assert sum(r['body_transfer_s'] for r in rows) == pytest.approx(.001)
    assert rows[-1]['metrics']['peak_buffer_s'] <= 3 + 1e-9


def test_catchup_changes_latency_within_rate_bounds():
    data, media = fixture(segments=8, rates=(80000.,))
    def run(catchup):
        cfg = StreamConfig(mode='live', source_age_s=5, target_latency_s=2, catchup=catchup,
                           max_buffer_s=6, startup_s=.5, resume_s=.5)
        return finish(StreamingEnv(data, media=media, config=cfg))[2][-1]['metrics']
    controlled, normal = run(True), run(False)
    assert controlled['rate_deviation_s'] > 0
    assert controlled['mean_live_latency_s'] < normal['mean_live_latency_s']
    assert controlled['played_s'] == pytest.approx(normal['played_s'])


def test_final_short_video_starts_below_startup_threshold():
    env = env_for(segments=1, duration=.5, part_s=.5, config=StreamConfig(startup_s=1))
    term, trunc, rows = finish(env)
    assert term and not trunc
    assert rows[-1]['metrics']['played_s'] == pytest.approx(.5)
    assert rows[-1]['metrics']['startup_s'] == pytest.approx(1)


def test_rtt_timeout_and_partial_live_segment_are_explicit():
    env = env_for(rtt=2, config=StreamConfig(request_timeout_s=1))
    term, trunc, rows = finish(env)
    assert trunc and rows[0]['rtt_s'] == 1 and rows[0]['bytes_received'] == 0
    env = env_for(segments=1, rates=(8.,), intervals=(.75,),
                  config=StreamConfig(mode='live', source_age_s=10, startup_s=.5, resume_s=.5, catchup=False))
    term, trunc, rows = finish(env)
    assert trunc and not term
    assert len(rows[0]['parts']) == 3
    assert rows[0]['metrics']['downloaded_media_s'] == 1.5
    assert rows[0]['metrics']['played_s'] + rows[0]['metrics']['buffer_s'] == pytest.approx(1.5)
