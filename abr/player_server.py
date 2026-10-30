"""Local MSE player, frozen policy inference, and paced HTTP trace replay.

This is an application-level test server, not a packet-level network emulator.
Run: python -m abr.player_server --checkpoint runs/learning-ablation/baseline-seed-7/accepted.npz
"""
import argparse
from dataclasses import asdict, replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

import numpy as np
import torch

from .data import Dataset
from .env import StreamConfig
from .models.checkpoint import load_model
from .models.inputs import make_batch
from .models.policy import DecisionPolicy
from .models.train_dueling_dqn import checkpoint_eligible
from .models.safety import ActionSafety, allowed_actions
from .transport import Replay

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ('dqn', 'highest_allowed', 'throughput', 'lowest', 'highest')


def select_action(policy, observation, config, model_policy=None):
    if policy == 'dqn':
        return model_policy(observation, config)
    if policy == 'lowest':
        return 0
    if policy == 'highest':
        return len(observation['candidates']) - 1
    if policy == 'highest_allowed':
        if model_policy is None or model_policy.safety is None:
            raise ValueError('Highest-allowed comparison requires the checkpoint action filter')
        return int(np.flatnonzero(allowed_actions(observation, model_policy.safety))[-1])
    if policy != 'throughput':
        raise ValueError('Unknown policy')
    samples = observation['history'][observation['history'][:, -1] > 0, 0]
    if not len(samples):
        return 0
    estimate = .85 * len(samples) / np.sum(1 / np.maximum(samples, 1e-12))
    eligible = np.flatnonzero(observation['candidates'][:, 0] <= estimate)
    return int(eligible[-1]) if len(eligible) else 0


def experimental_replay(dataset, config, replay_dataset=None, max_buffer_s=None):
    """Explicit research overrides; retain trained data/config identity separately."""
    selected = dataset if replay_dataset is None else replay_dataset
    for key in ('bitrates_kbps', 'durations_s', 'sizes_bytes'):
        if not np.array_equal(getattr(dataset.video, key), getattr(selected.video, key)):
            raise ValueError('Replay video differs from checkpoint training media')
    if max_buffer_s is not None:
        config = replace(config, max_buffer_s=max_buffer_s)
    if config.max_buffer_s < max(selected.video.durations_s) + max(config.startup_s, config.resume_s):
        raise ValueError('Buffer cannot hold segment plus playback threshold')
    return selected, config


class PlayerApp:
    def __init__(self, dataset, checkpoint, media_dir, *, replay_dataset=None, max_buffer_s=None):
        torch.set_num_threads(1)
        self.dataset = dataset
        self.model, self.metadata = load_model(checkpoint)
        if getattr(self.model.config, 'queue_features', False):
            raise ValueError('Queue-summary research checkpoints require the offline QueueObservation adapter; native deployment is not enabled')
        training = self.metadata['training']
        safety = training.get('safety_filter')
        self.policy = DecisionPolicy(self.model, safety=ActionSafety(**safety) if safety else None)
        self.config = StreamConfig(**training['config'])
        if self.config.mode != 'vod':
            raise ValueError('This player supports finite VOD only')
        validation = training['best_validation']
        if not checkpoint_eligible(validation, training.get('max_startup_s')):
            raise ValueError('Player requires an accepted checkpoint')
        if training['training_dataset_sha256'] != dataset.fingerprint:
            raise ValueError('Checkpoint and dataset differ')
        self.training_dataset_sha256 = dataset.fingerprint
        self.training_config = asdict(self.config)
        self.dataset, self.config = experimental_replay(dataset, self.config, replay_dataset, max_buffer_s)
        self.experimental_overrides = {'max_buffer_s': max_buffer_s} if max_buffer_s is not None else {}
        self.checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self.media_dir = Path(media_dir)
        self.index = json.loads((self.media_dir / 'index.json').read_text())
        self.reps = self.index['representations']
        if [r['bitrate_kbps'] for r in self.reps] != dataset.video.bitrates_kbps.tolist():
            raise ValueError('Cached media ladder differs from training')
        for i, rep in enumerate(self.reps):
            for j in range(len(dataset.video.durations_s)):
                name = f"{rep['id']}/{j + 1}.m4s"
                if (self.media_dir / name).stat().st_size != dataset.video.sizes_bytes[j, i]:
                    raise ValueError(f'Cached segment size mismatch: {name}')
        self.sessions = {}
        self.inference_lock = threading.Lock()

    def create_session(self, request):
        policy, split, index = request['policy'], request['split'], request['trace_index']
        if policy not in POLICIES or split not in self.dataset.splits or type(index) is not int:
            raise ValueError('Invalid session selection')
        traces = self.dataset.splits[split]
        if not 0 <= index < len(traces):
            raise ValueError('Invalid trace index')
        if len(self.sessions) >= 64:
            raise ValueError('Session limit reached; restart the local test server')
        token = uuid.uuid4().hex
        trace = traces[index]
        self.sessions[token] = {'policy': policy, 'trace': trace, 'start': time.monotonic(),
                                'requests': [], 'decisions': [], 'lock': threading.Lock()}
        return {'session': token, 'policy': policy, 'trace': trace.name, 'split': split,
                'trace_sha256': trace.fingerprint, 'trace_source': trace.source,
                'trace_semantics': trace.semantics, 'config': asdict(self.config),
                'durations_s': self.dataset.video.durations_s.tolist(), 'representations': self.reps,
                'dataset_sha256': self.dataset.fingerprint, 'checkpoint_sha256': self.checkpoint_sha256,
                'training_dataset_sha256': self.training_dataset_sha256, 'training_config': self.training_config,
                'experimental_overrides': self.experimental_overrides, 'request_recovery': False,
                'checkpoint_episode': self.metadata['training']['episode'],
                'safety_filter': self.metadata['training'].get('safety_filter'),
                'source_commit': self.index['source_commit'],
                'network_method': 'HTTP body pacing from finite trace, 16 KiB chunks; assumed per-request RTT; includes init requests; no packet/radio emulation'}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    @property
    def app(self):
        return self.server.app

    def log_message(self, *_args):
        pass

    def respond(self, value, status=200, kind='application/json'):
        body = json.dumps(value, allow_nan=False).encode() if kind == 'application/json' else value
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            if self.headers.get('Origin') not in (None, f'http://{self.headers.get("Host")}'):
                raise ValueError('Cross-origin requests are disabled')
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 65536:
                raise ValueError('Invalid request size')
            request = json.loads(self.rfile.read(length))
            if self.path == '/session':
                return self.respond(self.app.create_session(request))
            if self.path != '/decide':
                return self.respond({'error': 'Not found'}, 404)
            session = self.app.sessions[request['session']]
            index = request['segment_index']
            if type(index) is not int or not 0 <= index < len(self.app.dataset.video.durations_s):
                raise ValueError('Invalid segment index')
            duration = self.app.dataset.video.durations_s[index]
            ladder = self.app.dataset.video.bitrates_kbps
            observation = {'history': np.asarray(request['history'], dtype=np.float32),
                           'state': np.asarray(request['state'], dtype=np.float32),
                           'candidates': np.column_stack((ladder / 1000, ladder * duration / 8000,
                                                         np.full(len(ladder), duration))).astype(np.float32)}
            start = time.perf_counter()
            with self.app.inference_lock:
                make_batch([observation], [self.app.config], self.app.model.config)
                action = select_action(session['policy'], observation, self.app.config, self.app.policy)
            result = {'action': action, 'bitrate_kbps': float(ladder[action]),
                      'filter_applied': session['policy'] in ('dqn', 'highest_allowed') and self.app.policy.safety is not None,
                      'allowed_actions': np.flatnonzero(allowed_actions(observation, self.app.policy.safety if session['policy'] in ('dqn', 'highest_allowed') else None)).tolist(),
                      'decision_ms': (time.perf_counter() - start) * 1000}
            session['decisions'].append({'segment_index': index, 'state': observation['state'].tolist(), **result})
            self.respond(result)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            self.respond({'error': str(exc)}, 400)

    def do_GET(self):
        route = urlsplit(self.path)
        if route.path == '/config':
            return self.respond({'policies': POLICIES, 'splits': {k: [t.name for t in v] for k, v in self.app.dataset.splits.items()}})
        if route.path == '/audit':
            token = parse_qs(route.query).get('session', [''])[0]
            session = self.app.sessions.get(token)
            if session is None:
                return self.respond({'error': 'Unknown session'}, 404)
            return self.respond({'requests': list(session['requests']), 'decisions': list(session['decisions']),
                                 'in_flight': session['lock'].locked()})
        static = {'/': ('policy-player.html', 'text/html'),
                  '/policy-player.mjs': ('policy-player.mjs', 'text/javascript')}
        if route.path in static:
            name, kind = static[route.path]
            return self.respond((ROOT / 'experiments/validation' / name).read_bytes(), kind=kind)
        parts = route.path.strip('/').split('/')
        if len(parts) != 4 or parts[0] != 'media':
            return self.respond({'error': 'Not found'}, 404)
        _, token, rep, filename = parts
        session = self.app.sessions.get(token)
        name = rep + '/' + filename
        if session is None or name not in self.app.index['files']:
            return self.respond({'error': 'Unknown session or media'}, 404)
        if not session['lock'].acquire(blocking=False):
            return self.respond({'error': 'One media request per session required'}, 409)
        row = {'file': name, 'start_s': time.monotonic() - session['start'], 'bytes_sent': 0, 'complete': False}
        try:
            replay = Replay(session['trace'], row['start_s'])
            deadline = row['start_s'] + self.app.config.request_timeout_s
            rtt = replay.rtt_s
            elapsed, ok = replay.wait(rtt, deadline)
            time.sleep(elapsed)
            if not ok:
                return self.respond({'error': 'Trace exhausted'}, 410)
            body = (self.app.media_dir / name).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Assumed-RTT-S', str(rtt))
            self.end_headers()
            self.wfile.flush()
            replay = Replay(session['trace'], time.monotonic() - session['start'])
            for offset in range(0, len(body), 16384):
                block = body[offset:offset + 16384]
                transfer = replay.transfer(len(block), deadline)
                time.sleep(max(0., session['start'] + replay.time_s - time.monotonic()))
                sent = int(round(transfer.bytes_received))
                self.wfile.write(block[:sent])
                self.wfile.flush()
                row['bytes_sent'] += sent
                if not transfer.complete:
                    break
            row['complete'] = row['bytes_sent'] == len(body)
            if not row['complete']:
                self.close_connection = True
        except (BrokenPipeError, ConnectionResetError, ValueError) as exc:
            row['error'] = str(exc)
            self.close_connection = True
        finally:
            row['end_s'] = time.monotonic() - session['start']
            session['requests'].append(row)
            session['lock'].release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/learning-ablation/baseline-seed-7/accepted.npz')
    parser.add_argument('--dataset', default='data/hard_envivio/dataset.json')
    parser.add_argument('--media-dir', default='artifacts/player-media')
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--replay-dataset', help='Research-only traces; video must exactly match training media')
    parser.add_argument('--experimental-max-buffer-s', type=float, help='Explicit frozen-model buffer-cap experiment')
    args = parser.parse_args()
    app = PlayerApp(Dataset.load(args.dataset), args.checkpoint, args.media_dir,
                    replay_dataset=Dataset.load(args.replay_dataset) if args.replay_dataset else None,
                    max_buffer_s=args.experimental_max_buffer_s)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    server.app = app
    print(json.dumps({'url': f'http://127.0.0.1:{args.port}', 'checkpoint_sha256': app.checkpoint_sha256}), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
