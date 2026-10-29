# Envivio DASH VOD example

This is a small, reproducible input set for the Dueling Double DQN trainer.
It includes the manifest-derived video ladder, exact byte sizes for each of
the 49 DASH video segments, and eight measured network logs. The video media
payloads are not included.

The video comes from the EnvivioDash3 test stream in the public
[Pensieve repository](https://github.com/hongzimao/pensieve). Its
[manifest](https://github.com/hongzimao/pensieve/blob/master/video_server/Manifest.mpd)
lists six H.264 video representations at 300, 750, 1,200,
1,850, 2,850 and 4,300 kbps. It describes a 193.68 second video, with segments
of about four seconds. `dataset.json` and `media.json` contain the exact segment
byte sizes from the Git blob sizes. `provenance.json` records source URLs and
the Git tree hash used to read those sizes.

Network traces come from the [Ghent 4G/LTE logs](https://users.ugent.be/~jvdrhoof/dataset/)
collected in Belgium during 2015–2016. The importer converts interval bytes and milliseconds into kbps.
The source logs do not record RTT, so this dataset declares the importer's
0.08 second RTT assumption. They are observed HTTP goodput samples, not a
controlled link-capacity trace. The collection method downloaded a large file,
so the measured goodput can depend on that workload.

Each network log is kept wholly within one split:

- Train: bus 0001, car 0001, foot 0001, tram 0001.
- Validation: bicycle 0001, train 0001.
- Test: bus 0002, car 0002.

This is an initial realism check, not broad evidence. It uses one video and
eight historical cellular traces. Keep the test traces out of training and
checkpoint selection. Do not treat the resulting simulator metrics as player
or current-network results.

To rebuild the inputs, run:

```bash
PYTHONPATH=. .venv/bin/python scripts/prepare_realistic_vod.py
```

The script reads public manifests and trace logs, and does not download the
video segment payloads.
