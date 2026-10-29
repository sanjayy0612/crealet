"""Build a small VOD ABR dataset from public DASH test media and Ghent 4G logs.

Only JSON metadata and raw network logs are retained. The video segments are
not downloaded; their exact byte lengths come from the source repository's Git
tree metadata.
"""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from abr.data import Dataset, Video
from abr.media import Media
from abr.trace_import import load_ghent4g


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "realistic_envivio"
PENSIEVE_API = "https://api.github.com/repos/hongzimao/pensieve"
MPD_URL = "https://raw.githubusercontent.com/hongzimao/pensieve/master/video_server/Manifest.mpd"
TRACE_ZIP_URL = "https://users.ugent.be/~jvdrhoof/dataset-4g/logs/logs_all.zip"
TRACE_INDEX_URL = "https://users.ugent.be/~jvdrhoof/dataset-4g/logs/"
NS = {"d": "urn:mpeg:dash:schema:mpd:2011"}
LADDER_KBPS = (300, 750, 1200, 1850, 2850, 4300)
SPLIT_FILES = {
    "train": ("report_bus_0001.log", "report_car_0001.log", "report_foot_0001.log", "report_tram_0001.log"),
    "validation": ("report_bicycle_0001.log", "report_train_0001.log"),
    "test": ("report_bus_0002.log", "report_car_0002.log"),
}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "abr-decision-engine/1.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read()


def video_metadata():
    mpd = fetch(MPD_URL)
    root = ET.fromstring(mpd)
    tree = json.loads(fetch(PENSIEVE_API + "/git/trees/master?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("Git tree was truncated; exact media file sizes are not available")
    tree_sha = tree["sha"]
    file_sizes = {item["path"]: item["size"] for item in tree["tree"] if item["type"] == "blob"}
    reps = {}
    template = None
    for adaptation in root.findall(".//d:AdaptationSet", NS):
        if adaptation.attrib.get("mimeType") != "video/mp4":
            continue
        template = adaptation.find("d:SegmentTemplate", NS)
        for rep in adaptation.findall("d:Representation", NS):
            rate = round(int(rep.attrib["bandwidth"]) / 1000)
            if rate in LADDER_KBPS:
                reps[rate] = rep.attrib["id"]
    if set(reps) != set(LADDER_KBPS) or template is None:
        raise RuntimeError(f"MPD did not contain the expected video ladder: {reps}")

    duration = int(template.attrib["duration"]) / int(template.attrib["timescale"])
    total_duration = float(root.attrib["mediaPresentationDuration"].removeprefix("PT").removesuffix("S"))
    count = int(np.ceil(total_duration / duration))
    requests = []
    for rate in LADDER_KBPS:
        for index in range(count):
            name = template.attrib["media"].replace("$RepresentationID$", reps[rate]).replace("$Number$", str(index + int(template.attrib.get("startNumber", 1))))
            path = "video_server/" + name
            if path not in file_sizes:
                raise RuntimeError(f"Git tree is missing media segment {path}")
            requests.append((rate, index, file_sizes[path]))
    sizes = np.empty((count, len(LADDER_KBPS)), dtype=np.float64)
    for rate, index, size in requests:
        sizes[index, LADDER_KBPS.index(rate)] = size
    durations = np.full(count, duration, dtype=np.float64)
    durations[-1] = total_duration - duration * (count - 1)
    video = {"bitrates_kbps": list(LADDER_KBPS), "durations_s": durations.tolist(), "sizes_bytes": sizes.tolist()}
    provenance = {
        "video": MPD_URL,
        "video_repository": "https://github.com/hongzimao/pensieve",
        "video_git_tree_sha": tree_sha,
        "video_mpd_sha256": hashlib.sha256(mpd).hexdigest(),
        "segments": "Exact Git blob size for every selected DASH video segment; bytes only, no media payload retained.",
        "trace_collection": "Ghent 4G/LTE logs measured 2015-12-16 through 2016-02-04; route-session files are split independently.",
        "trace_archive": TRACE_ZIP_URL,
        "trace_index": TRACE_INDEX_URL,
        "rtt": "0.08 s assumed by existing Ghent importer; RTT was not present in source logs.",
        "limitations": "Small historical 4G sample; observed HTTP goodput can depend on the source download workload. Application-level replay only.",
    }
    return video, provenance


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    video_data, provenance = video_metadata()
    archive = zipfile.ZipFile(io.BytesIO(fetch(TRACE_ZIP_URL)))
    trace_map = {Path(name).name: name for name in archive.namelist() if name.endswith(".log")}
    splits = {}
    for split, filenames in SPLIT_FILES.items():
        traces = []
        for filename in filenames:
            if filename not in trace_map:
                raise RuntimeError(f"Trace archive is missing {filename}")
            raw = archive.read(trace_map[filename])
            (OUT / filename).write_bytes(raw)
            trace = load_ghent4g(OUT / filename, source_group=filename.removesuffix(".log"),
                                 source_url=TRACE_INDEX_URL)
            traces.append(trace)
        splits[split] = tuple(traces)
    dataset = Dataset(
        video=Video(**video_data),
        splits=splits,
        provenance=json.dumps(provenance, sort_keys=True),
    )
    media = Media(dataset.video, provenance=json.dumps(provenance, sort_keys=True))
    dataset.save(OUT / "dataset.json")
    media.save(OUT / "media.json")
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "segments": len(dataset.video.durations_s),
                      "duration_s": float(dataset.video.durations_s.sum()),
                      "ladder_kbps": list(LADDER_KBPS),
                      "traces": {split: [trace.name for trace in traces] for split, traces in splits.items()},
                      "dataset_sha256": dataset.fingerprint}, indent=2))


if __name__ == "__main__":
    main()
