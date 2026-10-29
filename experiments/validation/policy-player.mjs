const el = id => document.getElementById(id);
const video = el('video');
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const params = new URLSearchParams(location.search);
let settings;
window.runStatus = {done: false, started: false};

async function post(path, body) {
    const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const result = await response.json();
    if (!response.ok) throw Error(result.error || `HTTP ${response.status}`);
    return result;
}

function bufferSeconds() {
    for (let i = 0; i < video.buffered.length; i++) {
        if (video.currentTime >= video.buffered.start(i) - .05 && video.currentTime <= video.buffered.end(i) + .05)
            return Math.max(0, video.buffered.end(i) - video.currentTime);
    }
    return 0;
}

function appendOperation(buffer, action) {
    return new Promise((resolve, reject) => {
        const done = () => {cleanup(); resolve();};
        const error = () => {cleanup(); reject(Error('Media Source buffer error'));};
        const cleanup = () => {buffer.removeEventListener('updateend', done); buffer.removeEventListener('error', error);};
        buffer.addEventListener('updateend', done, {once: true});
        buffer.addEventListener('error', error, {once: true});
        try {action();} catch (e) {cleanup(); reject(e);}
    });
}

async function run() {
    el('start').disabled = true;
    for (const id of ['policy', 'split', 'trace']) el(id).disabled = true;
    const clock = performance.now();
    const t = () => (performance.now() - clock) / 1000;
    const capture = {schema_version: 1, origin: 'native_mse_policy_capture', events: [], samples: [], segments: [],
                     completed: false, user_agent: navigator.userAgent, time_origin: performance.timeOrigin};
    const abort = new AbortController();
    const wallLimit = Number(params.get('wall_limit_s') || 300);
    capture.wall_limit_s = wallLimit;
    let started = false, startup = null, stallStart = null, stallSeconds = 0, stallEvents = 0, finalized = false;
    let timer, context, source, sourceBuffer, codec, lastAction = null;
    const history = Array.from({length: 8}, () => [0, 0, 0, 0, 0]);
    const initializations = new Map();
    const event = type => capture.events.push({type, t_s: t(), media_time_s: video.currentTime});
    const playing = () => {
        event('playing');
        if (!started) {startup = t(); started = true;}
        if (stallStart !== null) {stallSeconds += t() - stallStart; stallStart = null;}
    };
    const waiting = () => {
        event('waiting');
        if (started && stallStart === null && !video.ended) {stallStart = t(); stallEvents++;}
    };
    video.addEventListener('playing', playing);
    video.addEventListener('waiting', waiting);
    async function finish(completed, error = null) {
        if (finalized) return;
        finalized = true;
        clearInterval(timer);
        if (stallStart !== null) stallSeconds += t() - stallStart;
        let position = 0, integral = 0;
        if (context) for (const row of capture.segments) {
            const duration = context.durations_s[row.segment_index];
            integral += Math.max(0, Math.min(video.currentTime, position + duration) - position) * row.bitrate_kbps;
            position += duration;
        }
        const quality = video.getVideoPlaybackQuality();
        capture.completed = completed;
        capture.error = error;
        capture.context = context;
        capture.metrics = {startup_s: startup, rebuffer_s: stallSeconds, stall_events: stallEvents,
                           played_s: video.currentTime, wall_time_s: t(),
                           mean_played_bitrate_kbps: video.currentTime > 0 ? integral / video.currentTime : 0,
                           dropped_frames: quality.droppedVideoFrames, total_frames: quality.totalVideoFrames};
        if (context) capture.transport_audit = await fetch(`/audit?session=${context.session}`).then(r => r.json());
        window.playbackResult = capture;
        window.runStatus = {done: true, completed, error, ...capture.metrics};
        el('status').textContent = JSON.stringify({policy: context?.policy, trace: context?.trace, completed, error, ...capture.metrics}, null, 2);
        el('save').disabled = false;
    }
    try {
        window.runStatus = {started: true, done: false, visibility: document.visibilityState};
        event('start');
        context = await post('/session', {policy: el('policy').value, split: el('split').value, trace_index: Number(el('trace').value)});
        source = new MediaSource();
        video.src = URL.createObjectURL(source);
        await new Promise(resolve => source.addEventListener('sourceopen', resolve, {once: true}));
        const total = context.durations_s.reduce((a, b) => a + b, 0);
        source.duration = total;
        codec = `video/mp4; codecs="${context.representations[0].codec}"`;
        if (!MediaSource.isTypeSupported(codec)) throw Error(`Browser cannot decode ${codec}`);
        sourceBuffer = source.addSourceBuffer(codec);
        const ended = new Promise((resolve, reject) => {
            video.addEventListener('ended', () => {event('ended'); resolve();}, {once: true});
            video.addEventListener('error', () => reject(Error(`Video decode error ${video.error?.code}: ${video.error?.message}`)), {once: true});
            abort.signal.addEventListener('abort', () => reject(abort.signal.reason), {once: true});
        });
        // Attach a handler immediately; a decode failure may occur during fetching.
        ended.catch(() => {});
        timer = setInterval(() => {
            if (t() > wallLimit && !finalized) {
                event('wall_budget_exceeded');
                abort.abort(Error('Playback exceeded the declared wall-time budget'));
                video.pause();
                finish(false, 'Playback exceeded the declared wall-time budget');
                return;
            }
            const row = {t_s: t(), media_time_s: video.currentTime, buffer_s: bufferSeconds(), ready_state: video.readyState,
                         visibility: document.visibilityState};
            capture.samples.push(row);
            window.runStatus = {done: false, started: true, ...row, segments: capture.segments.length};
            el('status').textContent = `${context.policy} · ${context.trace}\n${video.currentTime.toFixed(1)} / ${total.toFixed(1)} s · buffer ${row.buffer_s.toFixed(1)} s · ${capture.segments.length} segments`;
        }, 100);
        for (let index = 0; index < context.durations_s.length; index++) {
            const stepStart = performance.now();
            const decision = await post('/decide', {session: context.session, segment_index: index, history,
                state: [bufferSeconds(), 0, 1, 1 - index / context.durations_s.length,
                        lastAction === null ? 0 : context.representations[lastAction].bitrate_kbps / 1000]});
            const {action, bitrate_kbps, decision_ms} = decision;
            const rep = context.representations[action];
            const duration = context.durations_s[index];
            while (bufferSeconds() + duration > context.config.max_buffer_s + .01) {abort.signal.throwIfAborted(); await pause(50);}
            const desiredCodec = `video/mp4; codecs="${rep.codec}"`;
            if (lastAction !== action) {
                if (codec !== desiredCodec) {sourceBuffer.changeType(desiredCodec); codec = desiredCodec;}
                if (!initializations.has(action)) {
                    const r = await fetch(`/media/${context.session}/${rep.id}/Header.m4s`, {signal: AbortSignal.any([abort.signal, AbortSignal.timeout(30000)])});
                    if (!r.ok) throw Error(`Initialization HTTP ${r.status}`);
                    initializations.set(action, await r.arrayBuffer());
                }
                await appendOperation(sourceBuffer, () => sourceBuffer.appendBuffer(initializations.get(action)));
            }
            const requestStart = performance.now();
            const response = await fetch(`/media/${context.session}/${rep.id}/${index + 1}.m4s`, {signal: AbortSignal.any([abort.signal, AbortSignal.timeout(30000)])});
            if (!response.ok) throw Error(`Media HTTP ${response.status}`);
            const headersAt = performance.now();
            const body = await response.arrayBuffer();
            const receivedAt = performance.now();
            const bodySeconds = Math.max((receivedAt - headersAt) / 1000, .000001);
            await appendOperation(sourceBuffer, () => sourceBuffer.appendBuffer(body));
            history.shift();
            history.push([body.byteLength * 8 / bodySeconds / 1e6, (receivedAt - stepStart) / 10000,
                          0, (headersAt - requestStart) / 10000, 1]);
            capture.segments.push({segment_index: index, action, bitrate_kbps, decision_ms, bytes: body.byteLength,
                                   received_at_s: t(), body_s: bodySeconds,
                                   request_s: (receivedAt - requestStart) / 1000, buffer_s: bufferSeconds()});
            lastAction = action;
            if (index === 0) await video.play();
            if (video.error) throw Error(`Decode error: ${video.error.message}`);
            const removeBefore = video.currentTime - 12;
            if (removeBefore > 0 && sourceBuffer.buffered.length && sourceBuffer.buffered.start(0) < removeBefore)
                await appendOperation(sourceBuffer, () => sourceBuffer.remove(0, removeBefore));
        }
        source.endOfStream();
        await ended;
        await finish(true);
    } catch (e) {
        event('error');
        video.pause();
        await finish(false, e.message);
    }
}

el('save').onclick = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(window.playbackResult, null, 2)], {type: 'application/json'}));
    a.download = 'policy-playback.json'; a.click(); URL.revokeObjectURL(a.href);
};
el('start').onclick = run;
try {
    settings = await fetch('/config').then(r => r.json());
    el('policy').replaceChildren(...settings.policies.map(p => new Option(p, p)));
    el('split').replaceChildren(...Object.keys(settings.splits).map(p => new Option(p, p)));
    el('split').value = params.get('split') || 'validation';
    const setTraces = () => el('trace').replaceChildren(...settings.splits[el('split').value].map((name, i) => new Option(name, i)));
    el('split').onchange = setTraces;
    setTraces();
    el('policy').value = params.get('policy') || 'dqn';
    el('trace').value = params.get('trace') || '0';
    el('start').disabled = false;
    el('status').textContent = 'Ready. DQN uses the frozen checkpoint. Measurements include native playback events and a server transport audit.';
    if (params.get('autostart') === '1') run();
} catch (e) {
    el('status').textContent = e.message;
    window.runStatus = {done: true, error: e.message};
}
