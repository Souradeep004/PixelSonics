"""
PixelSonics - Error Correction Layer (ECC)

Phase 2: Reed-Solomon is now the primary ECC codec (via `reedsolo`).
Phase 1: XOR parity is retained as an explicit fallback if reedsolo is absent.

Reed-Solomon overview
─────────────────────
RS works over blocks of up to 255 bytes (GF(2^8)).  For each block of
*data_block_size* bytes it appends *nsym* error-correction symbols.

  • Can DETECT up to  nsym   symbol (byte) errors per block.
  • Can CORRECT up to nsym//2 symbol (byte) errors per block.

Example with nsym=10 (the default):
  • Corrects any 5 corrupted bytes per 245-byte data block.
  • Encoded block size = 245 + 10 = 255 bytes.

A 4-byte little-endian length trailer is appended after all encoded blocks so
the decoder can strip zero-padding from the final block exactly.

Layout on the wire:
  [RS block 0] [RS block 1] … [RS block N] [original_length : 4 B LE]

where each RS block = [data (data_block_size B)] [ECC symbols (nsym B)]

Module-level API (drop-in replacement for the old XOR functions)
────────────────────────────────────────────────────────────────
  encode(data, nsym=10)  → bytes
  decode(data, nsym=10)  → bytes

Class API (explicit, configurable)
───────────────────────────────────
  rs = ReedSolomonECC(nsym=10)
  encoded   = rs.encode(data)
  recovered = rs.decode(encoded)

XOR parity fallback
────────────────────
  xor_encode(data, block_size=64) → bytes
  xor_decode(data, block_size=64) → bytes
"""

from __future__ import annotations

import math
import struct

# ─── Constants ───────────────────────────────────────────────────────────────

_GF_MAX = 255
DEFAULT_NSYM = 10
DEFAULT_BLOCK_SIZE = _GF_MAX - DEFAULT_NSYM  # 245 data bytes per block


# ─── Exceptions ──────────────────────────────────────────────────────────────


class ECCError(Exception):
    """Raised when an uncorrectable ECC error is detected."""


# ─── Reed-Solomon implementation ─────────────────────────────────────────────

try:
    import reedsolo as _reedsolo

    _RS_AVAILABLE = True
except ImportError:
    _reedsolo = None
    _RS_AVAILABLE = False


class ReedSolomonECC:
    """
    Reed-Solomon ECC codec wrapping the `reedsolo` library.

    Args:
        nsym: Number of ECC symbols (bytes) appended per block.
              Corrects up to nsym//2 byte errors per block.
              Must satisfy: 1 <= nsym < 255.  Default is 10.

    Example::

        rs = ReedSolomonECC(nsym=10)
        encoded   = rs.encode(b"Hello PixelSonics!")
        recovered = rs.decode(encoded)
        assert recovered == b"Hello PixelSonics!"
    """

    def __init__(self, nsym: int = DEFAULT_NSYM):
        if not _RS_AVAILABLE:
            raise ImportError("reedsolo is not installed. Run: pip install reedsolo")
        if not (1 <= nsym < _GF_MAX):
            raise ValueError(f"nsym must be in [1, 254], got {nsym}")

        self.nsym = nsym
        self.data_block_size = _GF_MAX - nsym
        self._codec = _reedsolo.RSCodec(nsym)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _pad_to_blocks(self, data: bytes) -> tuple[bytes, int]:
        """Zero-pad *data* so its length is a multiple of data_block_size."""
        original_len = len(data)
        remainder = original_len % self.data_block_size
        if remainder:
            data = data + b"\x00" * (self.data_block_size - remainder)
        return data, original_len

    # ── public API ────────────────────────────────────────────────────────────

    def encode(self, data: bytes) -> bytes:
        """
        Encode *data* with Reed-Solomon ECC.

        Each data_block_size-byte chunk gets nsym ECC symbols appended.
        A 4-byte LE length trailer is appended at the very end.

        Returns:
            ECC-encoded bytes.
        """
        if not isinstance(data, bytes):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")

        padded, original_len = self._pad_to_blocks(data)

        out = bytearray()
        for i in range(0, len(padded), self.data_block_size):
            block = padded[i : i + self.data_block_size]
            out.extend(bytes(self._codec.encode(block)))

        out.extend(struct.pack("<I", original_len))
        return bytes(out)

    def decode(self, data: bytes) -> bytes:
        """
        Decode and error-correct *data* produced by encode().

        Returns:
            Original data bytes (padding stripped).

        Raises:
            ECCError: If a block has too many errors to correct.
            ValueError: If the data length is inconsistent.
        """
        if not isinstance(data, bytes):
            raise TypeError(f"Expected bytes, got {type(data).__name__}")
        if len(data) < 4:
            raise ValueError("ECC data too short to contain length trailer.")

        (original_len,) = struct.unpack("<I", data[-4:])
        ecc_body = data[:-4]

        codeword_size = self.data_block_size + self.nsym
        if len(ecc_body) % codeword_size != 0:
            raise ValueError(
                f"ECC body length {len(ecc_body)} is not a multiple of "
                f"codeword size {codeword_size} "
                f"(data_block={self.data_block_size}, nsym={self.nsym})."
            )

        out = bytearray()
        n_blocks = len(ecc_body) // codeword_size
        for block_idx in range(n_blocks):
            start = block_idx * codeword_size
            codeword = ecc_body[start : start + codeword_size]
            try:
                decoded_data, _, _ = self._codec.decode(codeword)
                out.extend(bytes(decoded_data))
            except _reedsolo.ReedSolomonError as exc:
                raise ECCError(
                    f"Reed-Solomon could not correct block {block_idx} "
                    f"(byte offset {start}): {exc}"
                ) from exc

        return bytes(out[:original_len])

    def max_correctable_errors(self) -> int:
        """Maximum byte errors correctable per block."""
        return self.nsym // 2

    def overhead_bytes(self, data_len: int) -> int:
        """ECC overhead bytes for a given data length."""
        n_blocks = math.ceil(max(data_len, 1) / self.data_block_size)
        return n_blocks * self.nsym + 4  # +4 for length trailer


# ─── Module-level convenience functions ──────────────────────────────────────


def encode(data: bytes, nsym: int = DEFAULT_NSYM) -> bytes:
    """
    Encode *data* with Reed-Solomon ECC.
    Falls back to XOR parity if reedsolo is not installed.
    """
    if _RS_AVAILABLE:
        return ReedSolomonECC(nsym).encode(data)
    return xor_encode(data)


def decode(data: bytes, nsym: int = DEFAULT_NSYM) -> bytes:
    """
    Decode *data* produced by encode().
    Falls back to XOR parity if reedsolo is not installed.
    """
    if _RS_AVAILABLE:
        return ReedSolomonECC(nsym).decode(data)
    return xor_decode(data)


def overhead_bytes(data_len: int, nsym: int = DEFAULT_NSYM) -> int:
    """Return ECC overhead bytes for *data_len* bytes of input."""
    if _RS_AVAILABLE:
        return ReedSolomonECC(nsym).overhead_bytes(data_len)
    return math.ceil(data_len / 64) + 4


# ─── XOR Parity fallback (Phase 1) ───────────────────────────────────────────

_XOR_DEFAULT_BLOCK = 64


def _xor_parity(block: bytes) -> int:
    parity = 0
    for b in block:
        parity ^= b
    return parity


def xor_encode(data: bytes, block_size: int = _XOR_DEFAULT_BLOCK) -> bytes:
    """
    XOR parity fallback encoder.
    One parity byte per block. Detects errors but cannot correct them.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    remainder = len(data) % block_size
    padded = data + b"\x00" * (block_size - remainder) if remainder else data
    out = bytearray()
    for i in range(0, len(padded), block_size):
        block = padded[i : i + block_size]
        out.extend(block)
        out.append(_xor_parity(block))
    out.extend(struct.pack("<I", len(data)))
    return bytes(out)


def xor_decode(data: bytes, block_size: int = _XOR_DEFAULT_BLOCK) -> bytes:
    """
    XOR parity fallback decoder.
    Verifies and strips parity bytes and length trailer.
    """
    if len(data) < 4:
        raise ValueError("XOR ECC data too short.")
    (original_len,) = struct.unpack("<I", data[-4:])
    ecc_body = data[:-4]
    stride = block_size + 1
    if len(ecc_body) % stride != 0:
        raise ValueError(
            f"XOR ECC body length {len(ecc_body)} not a multiple of stride {stride}."
        )
    out = bytearray()
    for block_idx, i in enumerate(range(0, len(ecc_body), stride)):
        chunk = ecc_body[i : i + stride]
        payload, stored = chunk[:-1], chunk[-1]
        if _xor_parity(payload) != stored:
            raise ECCError(f"XOR parity error in block {block_idx} at byte offset {i}.")
        out.extend(payload)
    return bytes(out[:original_len])
