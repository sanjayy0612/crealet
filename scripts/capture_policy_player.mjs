// Real-time Chrome playback capture using the built-in DevTools protocol.
// node scripts/capture_policy_player.mjs [--smoke] [--port=8787]
import {spawn, execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {mkdir, mkdtemp, writeFile, open} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

const smoke = process.argv.includes('--smoke');
const serverPort = Number(process.argv.find(x => x.startsWith('--port='))?.split('=')[1] || 8787);
const outputArg = process.argv.find(x => x.startsWith('--output='))?.slice('--output='.length);
const out = outputArg || (smoke ? 'runs/player-hard-smoke' : 'runs/player-hard');
const health = await fetch(`http://127.0.0.1:${serverPort}/config`, {signal: AbortSignal.timeout(5000)});
if (!health.ok) throw Error('Player server is unavailable');
await mkdir(out, {recursive: true});
const sleep = ms => new Promise(r => setTimeout(r, ms));
const {readFile} = await import('node:fs/promises');
const browsers = [];
const measureMemory = process.argv.includes('--memory');
const processList = promisify(execFile);

async function rssSnapshot() {
    const {stdout} = await processList('/bin/ps', ['-axo', 'pid=,ppid=,rss=']);
    return stdout.trim().split('\n').map(line => {
        const [pid, parent, kib] = line.trim().split(/\s+/).map(Number);
        return {pid, parent, bytes: kib * 1024};
    });
}

function treeRss(rows, root) {
    if (!rows.some(row => row.pid === root)) return null;
    const pids = new Set([root]);
    let changed = true;
    while (changed) {
        changed = false;
        for (const row of rows) if (pids.has(row.parent) && !pids.has(row.pid)) {
            pids.add(row.pid); changed = true;
        }
    }
    return rows.filter(row => pids.has(row.pid)).reduce((sum, row) => sum + row.bytes, 0);
}

async function startBrowser(name) {
    const profile = await mkdtemp(join(tmpdir(), 'abr-chrome-'));
    const log = await open(`${out}/${name}-chrome.log`, 'w');
    const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
        '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
        '--no-first-run', '--no-default-browser-check', '--autoplay-policy=no-user-gesture-required',
        '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
        '--disable-backgrounding-occluded-windows', '--window-size=1100,950', 'about:blank'
    ], {stdio: ['ignore', 'ignore', log.fd]});
    browsers.push({chrome, log});
    let debugPort;
    for (let tries = 0; tries < 100; tries++) {
        try {debugPort = Number((await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]); break;} catch {}
        await sleep(200);
    }
    if (!debugPort) throw Error(`Chrome did not start for ${name}; inspect its log`);
    return {debug: `http://127.0.0.1:${debugPort}`, pid: chrome.pid};
}

async function connect(url) {
    const socket = new WebSocket(url);
    await new Promise((resolve, reject) => {socket.onopen = resolve; socket.onerror = reject;});
    let id = 0;
    const pending = new Map();
    socket.onmessage = e => {
        const response = JSON.parse(e.data);
        const item = pending.get(response.id);
        if (!item) return;
        pending.delete(response.id); clearTimeout(item.timer);
        if (response.error) item.reject(Error(JSON.stringify(response.error)));
        else item.resolve(response.result);
    };
    return {
        call(method, params = {}) {
            return new Promise((resolve, reject) => {
                const requestId = ++id;
                const timer = setTimeout(() => {pending.delete(requestId); reject(Error(`CDP timed out: ${method}`));}, 15000);
                pending.set(requestId, {resolve, reject, timer});
                socket.send(JSON.stringify({id: requestId, method, params}));
            });
        },
        close() {socket.close();}
    };
}

let cases = smoke ? [{name: 'decode-smoke', policy: 'lowest', split: 'validation', trace: 2}] : [
    {name: 'dqn-measured', policy: 'dqn', split: 'test', trace: 0},
    {name: 'throughput-measured', policy: 'throughput', split: 'test', trace: 0},
    {name: 'dqn-outage', policy: 'dqn', split: 'test', trace: 2},
    {name: 'throughput-outage', policy: 'throughput', split: 'test', trace: 2},
    {name: 'highest-outage', policy: 'highest', split: 'test', trace: 2}
];
const policies = process.argv.find(x => x.startsWith('--policies='))?.slice('--policies='.length).split(',');
const casesFile = process.argv.find(x => x.startsWith('--cases='))?.slice('--cases='.length);
if (casesFile) {
    cases = JSON.parse(await readFile(casesFile, 'utf8'));
    if (!Array.isArray(cases) || !cases.length || new Set(cases.map(c => c.name)).size !== cases.length ||
        cases.some(c => !/^[a-z0-9-]+$/.test(c.name) || !Number.isInteger(c.trace) || c.trace < 0 ||
                    (c.port !== undefined && (!Number.isInteger(c.port) || c.port < 1 || c.port > 65535))))
        throw Error('Invalid custom playback cases');
}
if (!smoke && policies) cases = cases.filter(c => policies.includes(c.policy));
if (!cases.length) throw Error('No matching playback cases');
const connections = [];
try {
    const browserInfo = [];
    for (const item of cases) {
        const {debug, pid} = await startBrowser(item.name);
        item.browserPid = pid;
        item.memory = [];
        browserInfo.push({name: item.name, ...await fetch(debug + '/json/version').then(r => r.json())});
        await writeFile(`${out}/browser.json`, JSON.stringify(browserInfo, null, 2));
        const target = (await fetch(debug + '/json/list').then(r => r.json())).find(t => t.type === 'page');
        const client = await connect(target.webSocketDebuggerUrl);
        connections.push(client); item.client = client;
        await client.call('Page.enable');
        await client.call('Runtime.enable');
        if (measureMemory) {
            const rows = await rssSnapshot();
            item.baselineRss = treeRss(rows, pid);
        }
        const url = `http://127.0.0.1:${item.port || serverPort}/?policy=${item.policy}&split=${item.split}&trace=${item.trace}&autostart=1&wall_limit_s=${item.wall_limit_s || 210}`;
        await client.call('Page.navigate', {url});
    }
    const start = Date.now();
    const summaries = [];
    while (cases.some(c => !c.done)) {
        const processes = measureMemory ? await rssSnapshot() : null;
        for (const item of cases.filter(c => !c.done)) {
            if (measureMemory) {
                const heap = await item.client.call('Runtime.getHeapUsage');
                item.memory.push({t_s: (Date.now() - start) / 1000,
                                  chrome_process_tree_rss_bytes: treeRss(processes, item.browserPid),
                                  javascript_heap_used_bytes: heap.usedSize,
                                  javascript_heap_total_bytes: heap.totalSize,
                                  array_buffer_backing_bytes: heap.backingStorageSize ?? null});
            }
            const {result, exceptionDetails} = await item.client.call('Runtime.evaluate', {expression: 'window.runStatus', returnByValue: true});
            if (exceptionDetails) throw Error(JSON.stringify(exceptionDetails));
            const status = result.value;
            if (status?.visibility === 'hidden') throw Error(`Experiment page is hidden: ${item.name}`);
            if (smoke && status?.media_time_s >= 2 && status.segments >= 2) {
                const q = await item.client.call('Runtime.evaluate', {expression: 'document.getElementById("video").getVideoPlaybackQuality().totalVideoFrames', returnByValue: true});
                if (!(q.result.value > 0)) throw Error('No frames decoded');
                await writeFile(`${out}/${item.name}.json`, JSON.stringify({status, decoded_frames: q.result.value}, null, 2));
                item.done = true; summaries.push({name: item.name, decoded_frames: q.result.value, ...status});
            } else if (status?.done) {
                const data = await item.client.call('Runtime.evaluate', {expression: 'window.playbackResult', returnByValue: true});
                // A failed fetch can finish before the server logs its final bytes.
                if (data.result.value?.context?.session) {
                    for (let tries = 0; tries < 25; tries++) {
                        const audit = await fetch(`http://127.0.0.1:${item.port || serverPort}/audit?session=${data.result.value.context.session}`).then(r => r.json());
                        data.result.value.transport_audit = audit;
                        if (!audit.in_flight) break;
                        await sleep(200);
                    }
                }
                await writeFile(`${out}/${item.name}.json`, JSON.stringify(data.result.value || {automation_error: status}, null, 2));
                item.done = true; summaries.push({name: item.name, ...status});
            }
            if (item.done) {
                if (measureMemory) await writeFile(`${out}/${item.name}-memory.json`, JSON.stringify({
                    method: '2-second whole isolated Chrome process-tree RSS sum; shared pages may be counted multiple times; JS heap separately',
                    baseline_rss_bytes: item.baselineRss, samples: item.memory,
                    peak_rss_bytes: Math.max(...item.memory.map(s => s.chrome_process_tree_rss_bytes ?? 0)),
                    peak_javascript_heap_bytes: Math.max(...item.memory.map(s => s.javascript_heap_used_bytes))
                }, null, 2));
                const shot = await item.client.call('Page.captureScreenshot', {format: 'png'});
                await writeFile(`${out}/${item.name}.png`, Buffer.from(shot.data, 'base64'));
                console.log(JSON.stringify(summaries.at(-1)));
            }
        }
        if (Date.now() - start > (smoke ? 60000 : 600000)) throw Error('Playback capture exceeded its real-time budget');
        if (cases.some(c => !c.done)) await sleep(2000);
    }
    await writeFile(`${out}/summary.json`, JSON.stringify(summaries, null, 2) + '\n');
    if (smoke && summaries.some(x => !x.decoded_frames)) throw Error('Decode smoke failed');
    if (!smoke && summaries.some(x => !x.name.startsWith('highest') && x.completed !== true))
        throw Error('DQN or throughput playback did not complete; inspect raw captures');
} finally {
    for (const client of connections) client.close();
    for (const {chrome, log} of browsers) {
        chrome.kill('SIGTERM');
        await log.close();
    }
}
