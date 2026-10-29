"""Cache and verify the actual Envivio fragments for local browser playback."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import xml.etree.ElementTree as ET

from abr.data import Dataset

API = 'https://api.github.com/repos/hongzimao/pensieve'
OUT = Path('artifacts/player-media')


def fetch(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'abr-decision-engine/1.0'})
            return urllib.request.urlopen(req, timeout=45).read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def blob_hash(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commit = json.loads(fetch(API + '/commits/master'))['sha']
    tree = json.loads(fetch(API + f'/git/trees/{commit}?recursive=1'))
    if tree.get('truncated'):
        raise ValueError('Truncated Git tree')
    entries = {x['path']: x for x in tree['tree'] if x['type'] == 'blob'}
    root_url = f'https://raw.githubusercontent.com/hongzimao/pensieve/{commit}/video_server/'
    manifest = fetch(root_url + 'Manifest.mpd')
    (OUT / 'Manifest.mpd').write_bytes(manifest)
    ns = {'d': 'urn:mpeg:dash:schema:mpd:2011'}
    root = ET.fromstring(manifest)
    reps = sorted([{'id': r.attrib['id'], 'bitrate_kbps': int(r.attrib['bandwidth']) / 1000,
                    'codec': r.attrib['codecs']} for r in root.findall('.//d:Representation', ns)],
                  key=lambda r: r['bitrate_kbps'])
    dataset = Dataset.load('data/hard_envivio/dataset.json')
    if [r['bitrate_kbps'] for r in reps] != dataset.video.bitrates_kbps.tolist():
        raise ValueError('Media ladder differs from training input')
    paths = []
    for column, rep in enumerate(reps):
        for index in range(len(dataset.video.durations_s)):
            path = f"{rep['id']}/{index + 1}.m4s"
            entry = entries['video_server/' + path]
            if entry['size'] != dataset.video.sizes_bytes[index, column]:
                raise ValueError(f'Media bytes differ from training input: {path}')
            paths.append(path)
        paths.append(f"{rep['id']}/Header.m4s")

    def download(path):
        entry = entries['video_server/' + path]
        local = OUT / path
        if local.exists() and blob_hash(local.read_bytes()) == entry['sha']:
            return
        body = fetch(root_url + path)
        if len(body) != entry['size'] or blob_hash(body) != entry['sha']:
            raise ValueError(f'Corrupt media: {path}')
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(body)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, _ in enumerate(pool.map(download, paths), 1):
            if index % 30 == 0:
                print(f'Verified {index}/{len(paths)} media files', flush=True)
    manifest_index = {'source_commit': commit, 'source_url': root_url, 'representations': reps,
                      'files': {p: {'bytes': entries['video_server/' + p]['size'],
                                    'git_blob_sha1': entries['video_server/' + p]['sha']} for p in paths}}
    (OUT / 'index.json').write_text(json.dumps(manifest_index, indent=2) + '\n')
    print(json.dumps({'files': len(paths), 'bytes': sum(x['bytes'] for x in manifest_index['files'].values()),
                      'source_commit': commit}), flush=True)


if __name__ == '__main__':
    main()
