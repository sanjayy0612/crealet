/** Browser-native playback telemetry. No inference calls or playback control. */
export class PlaybackRecorder {
    constructor(video, identity, {clock = () => performance.now() / 1000, liveLatency = () => null} = {}) {
        this.video = video;
        this.clock = clock;
        this.liveLatency = liveLatency;
        this.identity = structuredClone(identity);
        this.events = [];
        this.samples = [];
        this.issues = [];
        this.listeners = [];
        this.active = false;
    }
    start() {
        if (this.active || this.origin !== undefined) throw new Error('Use a new recorder for each run');
        this.origin = this.clock();
        this.active = true;
        for (const type of ['playing', 'waiting', 'pause', 'seeking', 'ratechange', 'ended', 'error']) {
            const fn = () => this.record(type);
            this.video.addEventListener(type, fn);
            this.listeners.push([type, fn]);
        }
        this.record('start');
    }
    time() { return this.clock() - this.origin; }
    record(type) {
        if (!this.active) return;
        this.events.push({type, t_s: this.time(), media_time_s: this.video.currentTime});
        if (type === 'seeking') this.issues.push('seeking');
        if (type === 'pause' && !this.video.ended) this.issues.push('intentional_or_external_pause');
        if (type === 'error') this.issues.push('media_error');
        this.sample();
        if (type === 'ended') this.stop(true);
    }
    sample() {
        if (!this.active) return;
        let buffer = 0;
        for (let i = 0; i < this.video.buffered.length; i++) {
            if (this.video.buffered.start(i) <= this.video.currentTime && this.video.currentTime <= this.video.buffered.end(i)) {
                buffer = this.video.buffered.end(i) - this.video.currentTime;
                break;
            }
        }
        let latency = null;
        try { latency = this.liveLatency(); } catch { /* Missing latency stays missing. */ }
        this.samples.push({t_s: this.time(), media_time_s: this.video.currentTime,
            buffer_s: buffer, playback_rate: this.video.playbackRate,
            live_latency_s: Number.isFinite(latency) && latency >= 0 ? latency : null});
    }
    stop(completed = false) {
        if (!this.active) return;
        this.sample();
        this.active = false;
        this.completed = completed;
        this.end_s = this.time();
        for (const [type, fn] of this.listeners) this.video.removeEventListener(type, fn);
    }
    export() {
        if (this.active || this.origin === undefined) throw new Error('Stop the capture before exporting');
        return {schema_version: 1, origin: 'browser_capture', identity: this.identity,
            completed: this.completed, end_s: this.end_s, events: this.events, samples: this.samples,
            issues: [...new Set(this.issues)],
            measurement: 'Wall clock starts before player initialization; includes manifest/init/decode costs absent from simulator'};
    }
}
