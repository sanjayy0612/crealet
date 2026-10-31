"""Small strict reader for the unencrypted fragmented MP4s we generate."""
import struct


def boxes(raw):
    offset = 0
    while offset < len(raw):
        if len(raw) - offset < 8:
            raise ValueError('Truncated MP4 box header')
        size, kind = struct.unpack_from('>I4s', raw, offset)
        header = 8
        if size == 1:
            if len(raw) - offset < 16:
                raise ValueError('Truncated extended MP4 box header')
            size, header = struct.unpack_from('>Q', raw, offset + 8)[0], 16
        if size == 0:
            size = len(raw) - offset
        if size < header or offset + size > len(raw):
            raise ValueError('Invalid MP4 box length')
        yield kind, raw[offset + header:offset + size], raw[offset:offset + size]
        offset += size


def fragment_ticks(moof):
    total = 0
    for kind, traf, _ in boxes(moof):
        if kind != b'traf':
            continue
        default = None
        children = list(boxes(traf))
        for child, payload, _ in children:
            if child == b'tfhd':
                flags = int.from_bytes(payload[1:4], 'big')
                offset = 8 + (8 if flags & 1 else 0) + (4 if flags & 2 else 0)
                if flags & 8:
                    default = struct.unpack_from('>I', payload, offset)[0]
        for child, payload, _ in children:
            if child != b'trun':
                continue
            flags = int.from_bytes(payload[1:4], 'big')
            count = struct.unpack_from('>I', payload, 4)[0]
            offset = 8 + (4 if flags & 1 else 0) + (4 if flags & 4 else 0)
            for _ in range(count):
                duration = struct.unpack_from('>I', payload, offset)[0] if flags & 0x100 else default
                if duration is None or duration <= 0:
                    raise ValueError('Explicit positive sample duration required')
                total += duration
                offset += sum(4 for flag in (0x100, 0x200, 0x400, 0x800) if flags & flag)
    if total <= 0:
        raise ValueError('No timed samples in fragment')
    return total


def split_generated_mp4(raw):
    top = list(boxes(raw))
    movies = [payload for kind, payload, _ in top if kind == b'moov']
    if len(movies) != 1:
        raise ValueError('Exactly one movie initialization required')
    moov = movies[0]
    timescales = []
    for kind, trak, _ in boxes(moov):
        if kind == b'trak':
            for child, mdia, _ in boxes(trak):
                if child == b'mdia':
                    for tag, payload, _ in boxes(mdia):
                        if tag == b'mdhd':
                            timescales.append(struct.unpack_from('>I', payload, 20 if payload[0] == 1 else 12)[0])
    if len(timescales) != 1 or timescales[0] <= 0:
        raise ValueError('Exactly one video track required')
    index = moov.find(b'avcC')
    if index < 4 or moov[index + 4] != 1:
        raise ValueError('H.264 configuration required')
    codec = 'avc1.' + moov[index + 5:index + 8].hex().upper()
    init = b''.join(whole for kind, _, whole in top if kind in (b'ftyp', b'moov'))
    fragments, durations = [], []
    for i, (kind, payload, whole) in enumerate(top):
        if kind == b'moof':
            if i + 1 >= len(top) or top[i + 1][0] != b'mdat':
                raise ValueError('Each generated moof must be followed by mdat')
            fragments.append(whole + top[i + 1][2])
            durations.append(fragment_ticks(payload) / timescales[0])
    if not init or not fragments:
        raise ValueError('Initialization and media fragments required')
    return init, fragments, durations, codec
