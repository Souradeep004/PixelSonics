"""
PixelSonics - Error Correction Layer (ECC)

Phase 1: Simple parity-byte protection per block (lightweight, no dependencies).
Phase 2: Placeholder interface for Reed-Solomon (upgrade path).

Block layout after encoding:
  [DATA_BYTES (block_size B)] [PARITY_BYTE (1 B)]

The parity byte is the XOR of all data bytes in the block.  On decoding, if
the XOR of all bytes (including parity) is non-zero, an error is detected and
a ECCError is raised.  Single-bit corrections are NOT supported in Phase-1;
they are left for the Reed-Solomon upgrade in Phase 2.
"""

from __future__ import annotations

DEFAULT_BLOCK_SIZE = 64   # bytes of data per ECC block


class ECCError(Exception):
    """Raised when an uncorrectable ECC error is detected."""


# ─── XOR Parity (Phase 1) ────────────────────────────────────────────────────

def _xor_parity(block: bytes) -> int:
    """Compute XOR parity byte for a block."""
    parity = 0
    for b in block:
        parity ^= b
    return parity


def encode(data: bytes, block_size: int = DEFAULT_BLOCK_SIZE) -> bytes:
    """
    Append a 1-byte XOR parity word to every *block_size* bytes of data.

    The last block is zero-padded to *block_size* if needed; a 2-byte
    little-endian suffix records the original data length so decode() can
    strip padding exactly.

    Layout:
        [ECC blocks …] [original_length : 4 bytes little-endian]

    Args:
        data:       Raw bytes to protect.
        block_size: Number of data bytes per ECC block.

    Returns:
        ECC-encoded bytes with embedded parity bytes and length trailer.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    import struct as _struct

    # Pad data so it fills complete blocks
    remainder = len(data) % block_size
    if remainder:
        padded = data + b"\x00" * (block_size - remainder)
    else:
        padded = data

    out = bytearray()
    for i in range(0, len(padded), block_size):
        block = padded[i:i + block_size]
        out.extend(block)
        out.append(_xor_parity(block))

    # Append original length (4 bytes LE) so decode can strip padding
    out.extend(_struct.pack("<I", len(data)))
    return bytes(out)


def decode(data: bytes, block_size: int = DEFAULT_BLOCK_SIZE) -> bytes:
    """
    Verify and strip the parity bytes (and length trailer) added by *encode*.

    Args:
        data:       ECC-encoded bytes produced by encode().
        block_size: Must match the value used during encoding.

    Returns:
        Original data bytes (padding and parity stripped).

    Raises:
        ECCError: If a parity check fails in any block.
        ValueError: If data length is inconsistent with block_size.
    """
    import struct as _struct

    if len(data) < 4:
        raise ValueError("ECC data too short to contain length trailer.")

    # Strip 4-byte original length trailer
    original_length, = _struct.unpack("<I", data[-4:])
    ecc_body = data[:-4]

    stride = block_size + 1
    if len(ecc_body) % stride != 0:
        raise ValueError(
            f"ECC body length {len(ecc_body)} is not a multiple of stride {stride}. "
            "Are you sure this was encoded with the same block_size?"
        )

    out = bytearray()
    for block_idx, i in enumerate(range(0, len(ecc_body), stride)):
        block_with_parity = ecc_body[i:i + stride]
        payload = block_with_parity[:-1]
        stored_parity = block_with_parity[-1]
        computed_parity = _xor_parity(payload)

        if computed_parity != stored_parity:
            raise ECCError(
                f"Parity error in block {block_idx} "
                f"(byte offset {i}): "
                f"expected 0x{stored_parity:02X}, got 0x{computed_parity:02X}"
            )
        out.extend(payload)

    return bytes(out[:original_length])


def overhead_bytes(data_len: int, block_size: int = DEFAULT_BLOCK_SIZE) -> int:
    """
    Return the number of parity bytes that will be added for a given data length.
    """
    import math
    return math.ceil(data_len / block_size)


# ─── Phase 2: Reed-Solomon Interface (stub) ──────────────────────────────────

class ReedSolomonECC:
    """
    Stub for Phase 2 Reed-Solomon ECC upgrade.

    Drop in the `reedsolo` package (pip install reedsolo) and implement
    the encode/decode methods below to upgrade error correction power
    from single-byte parity detection to multi-byte correction.

    Usage (future):
        rs = ReedSolomonECC(nsym=10)
        encoded = rs.encode(data)
        recovered = rs.decode(encoded)
    """

    def __init__(self, nsym: int = 10):
        """
        Args:
            nsym: Number of Reed-Solomon error-correction symbols (bytes).
                  Can correct up to nsym//2 byte errors.
        """
        self.nsym = nsym
        self._rs = None
        try:
            import reedsolo
            self._rs = reedsolo.RSCodec(nsym)
        except ImportError:
            pass  # RS not available; methods will raise NotImplementedError

    def encode(self, data: bytes) -> bytes:
        if self._rs is None:
            raise NotImplementedError(
                "Install `reedsolo` to enable Reed-Solomon ECC: "
                "pip install reedsolo"
            )
        return bytes(self._rs.encode(data))

    def decode(self, data: bytes) -> bytes:
        if self._rs is None:
            raise NotImplementedError(
                "Install `reedsolo` to enable Reed-Solomon ECC: "
                "pip install reedsolo"
            )
        decoded, _, _ = self._rs.decode(data)
        return bytes(decoded)
