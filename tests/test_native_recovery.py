from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from abr.data import Trace
from abr.env import StreamConfig
from abr.player_server import Handler
from abr.recovery import RecoveryConfig


def run_node(body):
    if shutil.which('node') is None:
        pytest.skip('Node required for native download implementation')
    url = Path('experiments/validation/request-recovery.mjs').resolve().as_uri()
    program = f"import {{downloadSegment, abandonReason}} from '{url}';\n" + body
    subprocess.run(['node', '--input-type=module', '-e', program], check=True,
                   capture_output=True, text=True, timeout=10)


def test_native_abandon_decisions_match_python_controller():
    config = RecoveryConfig()
    base = dict(elapsed_s=3., received_bytes=100., content_bytes=10000., idle_s=0.,
                buffer_s=1., lowest_estimate_bytes=1000., at_lowest=False)
    changes = [{}, {'idle_s': 3.}, {'at_lowest': True}, {'elapsed_s': 1.},
               {'received_bytes': 9500.}, {'buffer_s': 400.}, {'received_bytes': 0.}]
    rows = [{'probe': base | change, 'expected': config.abandon_reason(**(base | change))} for change in changes]
    run_node('import assert from "node:assert/strict";\n' +
             f'const cfg={json.dumps(asdict(config))}; const rows={json.dumps(rows)};\n' +
             'for(const row of rows) assert.equal(abandonReason(cfg,row.probe),row.expected);')


def test_partial_body_is_discarded_and_retry_waits_for_cancel_ack():
    run_node('''
import assert from 'node:assert/strict';
let calls=0, acknowledged=false;
const attempts=[];
const result=await downloadSegment({session:'s',segmentIndex:2,action:1,duration:4,
 representations:[{id:'low',bitrate_kbps:300},{id:'high',bitrate_kbps:750}],timeoutS:1,
 recovery:{max_retries:1,early_abandon:false,poll_s:.01,backoff_s:0},
 signal:new AbortController().signal,getBuffer:()=>20,ensureInit:async()=>{},
 cancelRequest:async()=>{await new Promise(r=>setTimeout(r,10));acknowledged=true;},
 onAttempt:r=>attempts.push(r), fetchFn:async(url)=>{
   calls++;
   if(calls===1) return new Response(new ReadableStream({start(c){
      c.enqueue(new Uint8Array([1,2]));setTimeout(()=>c.error(Error('truncated')),5);
   }}),{headers:{'Content-Length':'5','X-Request-ID':'1'}});
   assert.ok(acknowledged);assert.ok(url.includes('/low/3.m4s'));
   return new Response(new Uint8Array([9,8,7]),{headers:{'Content-Length':'3','X-Request-ID':'2'}});
 }});
assert.equal(result.action,0);assert.equal(calls,2);
assert.deepEqual([...new Uint8Array(result.body)],[9,8,7]);
assert.deepEqual(attempts.map(a=>a.bytes_received),[2,3]);
assert.deepEqual(attempts.map(a=>a.outcome),['network_error','complete']);
''')


def test_no_progress_watchdog_downgrades_without_waiting_for_a_read():
    run_node('''
import assert from 'node:assert/strict';
let calls=0;const attempts=[];
const result=await downloadSegment({session:'s',segmentIndex:0,action:1,duration:1,
 representations:[{id:'low',bitrate_kbps:.008},{id:'high',bitrate_kbps:750}],timeoutS:1,
 recovery:{max_retries:1,early_abandon:true,poll_s:.005,min_observation_s:.001,
           no_progress_s:.01,buffer_reserve_s:2,restart_fraction:.8,backoff_s:0},
 signal:new AbortController().signal,getBuffer:()=>20,ensureInit:async()=>{},cancelRequest:async()=>{},
 onAttempt:r=>attempts.push(r), fetchFn:async(url,{signal})=>{
   calls++;
   if(calls===1) return new Response(new ReadableStream({start(c){
      signal.addEventListener('abort',()=>c.error(signal.reason),{once:true});
   }}),{headers:{'Content-Length':'100000'}});
   return new Response(new Uint8Array([1]),{headers:{'Content-Length':'1'}});
 }});
assert.equal(result.action,0);assert.equal(attempts[0].outcome,'no_progress');
assert.equal(calls,2);
''')


def test_hanging_lowest_requests_stop_at_retry_limit_and_clear_timers():
    run_node('''
import assert from 'node:assert/strict';
const attempts=[];let cancels=0;
await assert.rejects(downloadSegment({session:'s',segmentIndex:0,action:0,duration:4,
 representations:[{id:'low',bitrate_kbps:300}],timeoutS:.02,
 recovery:{max_retries:2,early_abandon:false,poll_s:.005,backoff_s:0},
 signal:new AbortController().signal,getBuffer:()=>20,ensureInit:async()=>{},
 cancelRequest:async()=>{cancels++;},onAttempt:r=>attempts.push(r),
 fetchFn:async(url,{signal})=>new Response(new ReadableStream({start(c){
   signal.addEventListener('abort',()=>c.error(signal.reason),{once:true});
 }}),{headers:{'Content-Length':'1000'}})}));
assert.equal(attempts.length,3);assert.equal(cancels,3);
assert.ok(attempts.every(a=>a.outcome==='request_timeout'));
''')


def test_real_server_cancel_interrupts_zero_service_and_releases_next_request(tmp_path):
    root = tmp_path / 'r'
    root.mkdir()
    (root / '1.m4s').write_bytes(b'x' * 2000)
    session = {'trace': Trace('outage', [100.], [0.], 0.), 'start': time.monotonic(),
               'requests': [], 'decisions': [], 'lock': threading.Lock(), 'cancel': threading.Event()}
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    server.app = SimpleNamespace(sessions={'s': session}, index={'files': {'r/1.m4s': {}}},
                                 media_dir=tmp_path, config=StreamConfig())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        response = urllib.request.urlopen(base + '/media/s/r/1.m4s', timeout=3)
        start = time.monotonic()
        req = urllib.request.Request(base + '/cancel', data=b'{"session":"s"}',
                                     headers={'Content-Type': 'application/json'})
        assert json.load(urllib.request.urlopen(req, timeout=3))['cancelled']
        assert time.monotonic() - start < 2.
        response.close()
        assert session['requests'][0]['error'] == 'cancelled'
        assert not session['lock'].locked()
        session['trace'] = Trace('recovered', [100.], [8000.], 0.)
        with urllib.request.urlopen(base + '/media/s/r/1.m4s', timeout=3) as good:
            assert good.read() == b'x' * 2000
            assert good.headers['X-Request-ID'] == '2'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
