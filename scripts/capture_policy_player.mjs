// Real-time Chrome playback capture using the built-in DevTools protocol.
// node scripts/capture_policy_player.mjs [--smoke] [--port=8787]
import {spawn} from 'node:child_process';
import {mkdir, mkdtemp, writeFile, open} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

const smoke = process.argv.includes('--smoke');
const serverPort = Number(process.argv.find(x => x.startsWith('--port='))?.split('=')[1] || 8787);
const out = smoke ? 'runs/player-hard-smoke' : 'runs/player-hard';
const health = await fetch(`http://127.0.0.1:${serverPort}/config`, {signal: AbortSignal.timeout(5000)});
if (!health.ok) throw Error('Player server is unavailable');
await mkdir(out, {recursive: true});
const sleep = ms => new Promise(r => setTimeout(r, ms));
const {readFile} = await import('node:fs/promises');
const browsers = [];

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
    return `http://127.0.0.1:${debugPort}`;
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

const cases = smoke ? [{name: 'decode-smoke', policy: 'lowest', split: 'validation', trace: 2}] : [
    {name: 'dqn-measured', policy: 'dqn', split: 'test', trace: 0},
    {name: 'throughput-measured', policy: 'throughput', split: 'test', trace: 0},
    {name: 'dqn-outage', policy: 'dqn', split: 'test', trace: 2},
    {name: 'throughput-outage', policy: 'throughput', split: 'test', trace: 2},
    {name: 'highest-outage', policy: 'highest', split: 'test', trace: 2}
];
const connections = [];
try {
    const browserInfo = [];
    for (const item of cases) {
        const debug = await startBrowser(item.name);
        browserInfo.push({name: item.name, ...await fetch(debug + '/json/version').then(r => r.json())});
        await writeFile(`${out}/browser.json`, JSON.stringify(browserInfo, null, 2));
        const target = (await fetch(debug + '/json/list').then(r => r.json())).find(t => t.type === 'page');
        const client = await connect(target.webSocketDebuggerUrl);
        connections.push(client); item.client = client;
        await client.call('Page.enable');
        await client.call('Runtime.enable');
        const url = `http://127.0.0.1:${serverPort}/?policy=${item.policy}&split=${item.split}&trace=${item.trace}&autostart=1&wall_limit_s=210`;
        await client.call('Page.navigate', {url});
    }
    const start = Date.now();
    const summaries = [];
    while (cases.some(c => !c.done)) {
        for (const item of cases.filter(c => !c.done)) {
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
                await writeFile(`${out}/${item.name}.json`, JSON.stringify(data.result.value || {automation_error: status}, null, 2));
                item.done = true; summaries.push({name: item.name, ...status});
            }
            if (item.done) {
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
