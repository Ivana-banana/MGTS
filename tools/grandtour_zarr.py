"""Read local GrandTour Zarr-in-TAR archives (Blosc1 + LZ4 + byte shuffle) without numcodecs."""
import json
import struct
import tarfile
from pathlib import Path

import lz4.block
import numpy as np


def blosc_decompress(buf):
    version, _, flags, typesize, nbytes, blocksize, cbytes = struct.unpack_from('<BBBBIII', buf)
    if version not in (2, 3) or cbytes != len(buf):
        raise ValueError(f'Unsupported Blosc header: version {version}, {cbytes} != {len(buf)} bytes')
    if flags & 0x04:
        raise ValueError('Bit shuffle is not supported')
    if flags & 0x02:
        return bytes(buf[16:16 + nbytes])
    nblocks = -(-nbytes // blocksize)
    starts = struct.unpack_from(f'<{nblocks}i', buf, 16)
    out = bytearray(nbytes)
    for j, src in enumerate(starts):
        size = min(blocksize, nbytes - j * blocksize)
        streams = 1 if (flags & 0x10 or size != blocksize) else typesize
        if streams > 1 and size % streams:
            streams = 1
        block = bytearray()
        for _ in range(streams):
            (csize,) = struct.unpack_from('<i', buf, src)
            src += 4
            part = size // streams
            block += buf[src:src + csize] if csize == part else lz4.block.decompress(
                bytes(buf[src:src + csize]), uncompressed_size=part)
            src += csize
        if len(block) != size:
            raise ValueError('Blosc block size mismatch')
        if flags & 0x01 and typesize > 1:
            whole = size - size % typesize
            body = np.frombuffer(bytes(block[:whole]), np.uint8).reshape(typesize, -1).T.reshape(-1)
            block[:whole] = body.tobytes()
        out[j * blocksize:j * blocksize + size] = block
    return bytes(out)


class Archive:
    """One GrandTour topic archive, e.g. velodyne_points.tar."""

    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.stem
        self.tar = tarfile.open(self.path)
        self.attrs = json.load(self.tar.extractfile(f'{self.name}/.zattrs'))

    def meta(self, key):
        return json.load(self.tar.extractfile(f'{self.name}/{key}/.zarray'))

    def chunk(self, key, index=0):
        m = self.meta(key)
        member = '.'.join([str(index)] + ['0'] * (len(m['shape']) - 1))
        raw = blosc_decompress(self.tar.extractfile(f'{self.name}/{key}/{member}').read())
        return np.frombuffer(raw, m['dtype']).reshape(m['chunks'])

    def array(self, key):
        """Whole array; for keys stored in one chunk along the first axis or concatenated."""
        m = self.meta(key)
        count = -(-m['shape'][0] // m['chunks'][0])
        return np.concatenate([self.chunk(key, i) for i in range(count)])[:m['shape'][0]]
