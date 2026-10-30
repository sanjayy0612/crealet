// Whole-segment recovery. Nothing is appended until one complete body arrives.
export function abandonReason(config, p) {
    if (!config.early_abandon || p.at_lowest || p.elapsed_s < config.min_observation_s) return null;
    const remaining = Math.max(0, p.content_bytes - p.received_bytes);
    if (p.lowest_estimate_bytes >= config.restart_fraction * remaining) return null;
    if (p.idle_s >= config.no_progress_s) return 'no_progress';
    if (p.received_bytes > 0 && remaining / (p.received_bytes / p.elapsed_s) >
        Math.max(config.min_observation_s, p.buffer_s - config.buffer_reserve_s)) return 'buffer_risk';
    return null;
}

function sleep(seconds, signal) {
    signal.throwIfAborted();
    return new Promise((resolve, reject) => {
        const abort = () => {clearTimeout(timer); reject(signal.reason);};
        const timer = setTimeout(() => {signal.removeEventListener('abort', abort); resolve();}, seconds * 1000);
        signal.addEventListener('abort', abort, {once: true});
    });
}

export async function downloadSegment({session, segmentIndex, action, duration, representations,
    timeoutS, recovery, signal, getBuffer, ensureInit, cancelRequest, onAttempt = () => {}, onProgress = () => {},
    fetchFn = fetch, now = () => performance.now() / 1000}) {
    const config = recovery || {max_retries: 0};
    const initialAction = action;
    const lowest = representations.reduce((best, rep, i) =>
        rep.bitrate_kbps < representations[best].bitrate_kbps ? i : best, 0);
    const attempts = [];
    let totalBodyS = 0, totalHeaderS = 0;
    for (let attempt = 0; attempt <= config.max_retries; attempt++) {
        signal.throwIfAborted();
        const controller = new AbortController();
        const combined = AbortSignal.any([signal, controller.signal]);
        const start = now();
        let bodyStart = null, requestStart = null, lastProgress = start, received = 0;
        let reason = null, contentBytes = null, requestId = null, reader = null, probe = null;
        let watchdog = null, fatal = false, chunks = [];
        const armDeadline = () => setTimeout(() => {reason = 'request_timeout'; controller.abort(Error(reason));}, timeoutS * 1000);
        let deadline = armDeadline();
        try {
            await ensureInit(action, combined);
            // Initialization and media are distinct requests, each bounded.
            clearTimeout(deadline);
            deadline = armDeadline();
            requestStart = now();
            const rep = representations[action];
            const response = await fetchFn(`/media/${session}/${rep.id}/${segmentIndex + 1}.m4s`, {signal: combined});
            if (!response.ok) {fatal = [400, 401, 403, 404, 410].includes(response.status); throw Error(`Media HTTP ${response.status}`);}
            requestId = response.headers.get('X-Request-ID');
            const length = Number(response.headers.get('Content-Length'));
            contentBytes = Number.isFinite(length) && length > 0 ? length : null;
            bodyStart = lastProgress = now();
            totalHeaderS += bodyStart - requestStart;
            if (recovery && attempt < config.max_retries) watchdog = setInterval(() => {
                if (contentBytes === null || received >= contentBytes) return;
                const current = now();
                probe = {elapsed_s: current - bodyStart, received_bytes: received, content_bytes: contentBytes,
                         idle_s: current - lastProgress, buffer_s: getBuffer(),
                         lowest_estimate_bytes: representations[lowest].bitrate_kbps * 1000 / 8 * duration,
                         at_lowest: action === lowest};
                reason = abandonReason(config, probe);
                if (reason) controller.abort(Error(reason));
            }, config.poll_s * 1000);
            reader = response.body.getReader();
            while (true) {
                const chunk = await reader.read();
                if (chunk.done) break;
                received += chunk.value.byteLength;
                chunks.push(chunk.value);
                lastProgress = now();
                onProgress({attempt, action, bytes_received: received, content_bytes: contentBytes});
            }
            if (contentBytes !== null && received !== contentBytes) throw Error('Incomplete response body');
            combined.throwIfAborted();
            totalBodyS += now() - bodyStart;
            const row = {attempt, action, policy_action: initialAction, request_id: requestId,
                         elapsed_s: now() - start, bytes_received: received, content_bytes: contentBytes,
                         outcome: 'complete', last_probe: probe};
            attempts.push(row); onAttempt(row);
            const bytes = new Uint8Array(received);
            let offset = 0;
            for (const chunk of chunks) {bytes.set(chunk, offset); offset += chunk.byteLength;}
            return {action, body: bytes.buffer, attempts, body_s: totalBodyS, header_s: totalHeaderS};
        } catch (error) {
            if (bodyStart !== null) totalBodyS += now() - bodyStart;
            reason = signal.aborted ? 'playback_aborted' : reason || 'network_error';
            const row = {attempt, action, policy_action: initialAction, request_id: requestId,
                         elapsed_s: now() - start, bytes_received: received, content_bytes: contentBytes,
                         outcome: reason, error: error.message, last_probe: probe};
            attempts.push(row); onAttempt(row);
            // Release body buffers before a retry. Failed bodies are never reused.
            chunks = [];
            controller.abort();
            clearInterval(watchdog); clearTimeout(deadline);
            if (reader) await reader.cancel().catch(() => {});
            // Wait for explicit server cancellation acknowledgement before retrying.
            await cancelRequest();
            if (signal.aborted || fatal || attempt === config.max_retries) throw error;
            if (reason !== 'buffer_risk') await sleep(config.backoff_s, signal);
            action = lowest;
        } finally {
            clearInterval(watchdog); clearTimeout(deadline);
            if (reader) reader.releaseLock();
        }
    }
    throw Error('Retry limit exhausted');
}
