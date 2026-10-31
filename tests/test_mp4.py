import struct

import pytest

from abr.mp4 import boxes, split_generated_mp4


def box(kind, payload):
    return struct.pack('>I4s', len(payload) + 8, kind) + payload


def fixture():
    mdhd = b'\0' * 12 + struct.pack('>I', 12800) + b'\0' * 8
    init = box(b'ftyp', b'isom') + box(b'moov',
        box(b'trak', box(b'mdia', box(b'mdhd', mdhd))) + box(b'avcC', bytes([1, 66, 192, 10, 0, 0])))
    tfhd = bytes([0, 0, 0, 8]) + struct.pack('>II', 1, 512)
    trun = b'\0' * 4 + struct.pack('>I', 100)
    fragment = box(b'moof', box(b'traf', box(b'tfhd', tfhd) + box(b'trun', trun))) + box(b'mdat', b'frames')
    return init, fragment


def test_exact_fragment_bytes_codec_and_sample_timing():
    init, fragment = fixture()
    header, fragments, durations, codec = split_generated_mp4(init + fragment * 2)
    assert header == init
    assert fragments == [fragment, fragment]
    assert durations == [4., 4.]
    assert codec == 'avc1.42C00A'


@pytest.mark.parametrize('raw', [b'x', struct.pack('>I4s', 7, b'free'),
                               struct.pack('>I4s', 100, b'free'), struct.pack('>I4s', 1, b'free')])
def test_rejects_truncated_or_invalid_boxes(raw):
    with pytest.raises(ValueError):
        list(boxes(raw))


def test_extended_and_to_end_box_sizes():
    raw = struct.pack('>I4sQ', 1, b'free', 19) + b'abc'
    assert list(boxes(raw))[0][1] == b'abc'
    assert list(boxes(struct.pack('>I4s', 0, b'free') + b'abc'))[0][1] == b'abc'


def test_rejects_missing_movie_or_missing_media_body():
    init, fragment = fixture()
    with pytest.raises(ValueError):
        split_generated_mp4(fragment)
    moof_only = list(boxes(fragment))[0][2]
    with pytest.raises(ValueError, match='followed by mdat'):
        split_generated_mp4(init + moof_only)
