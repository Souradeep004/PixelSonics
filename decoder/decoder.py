from __future__ import annotations
from framing.framer import SYNC_MAGIC, SYNC_SIZE, HEADER_SIZE, CRC_SIZE
import struct as _struct

import io
import json
import struct
from dataclasses import dataclass

import numpy as np

from compression import smart_decompress
from ecc import decode as ecc_decode, ECCError
from framing import deframe, FramingError
from modulation import (
    demodulate,
    SAMPLE_RATE,
    BIT_DURATION,
    PREAMBLE_BITS,
    FREQ_0,
    FREQ_1,
)

"""
PixelSonics - Decoder

Reverses the full encoding pipeline:

  Audio → Demodulate → Deframe → ECC Decode → Decompress → Strip Header → Original Bytes

The decoder returns both the recovered bytes and the metadata header so the
caller knows the original file type and size.
"""


# ─── Result type ─────────────────────────────────────────────────────────────


@dataclass
class DecodeResult:
    """Holds the recovered payload plus its original metadata."""

    data: bytes  # Recovered original bytes
    media_type: str  # e.g. "text/plain", "image/png", "application/pdf"
    original_size: int  # Declared size from header

    def as_text(self, encoding: str = "utf-8") -> str:
        """Convenience: decode bytes as text."""
        return self.data.decode(encoding)

    def save(self, path: str) -> str:
        """Write recovered bytes to *path* and return the path."""
        with open(path, "wb") as f:
            f.write(self.data)
        return path


# ─── Header parsing ──────────────────────────────────────────────────────────


class DecoderError(Exception):
    """Raised for any unrecoverable decoding error."""


def _parse_header(stream: io.BytesIO) -> tuple[str, int]:
    """
    Read and parse the length-prefixed JSON header from *stream*.

    Returns:
        (media_type, original_size)
    """
    length_bytes = stream.read(4)
    if len(length_bytes) < 4:
        raise DecoderError("Stream too short to contain header length prefix.")

    (header_len,) = struct.unpack(">I", length_bytes)
    header_json = stream.read(header_len)
    if len(header_json) < header_len:
        raise DecoderError(
            f"Truncated header: expected {header_len} B, got {len(header_json)} B"
        )

    try:
        meta = json.loads(header_json.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DecoderError(f"Invalid header JSON: {exc}") from exc

    media_type = meta.get("type", "application/octet-stream")
    original_size = meta.get("size", -1)
    return media_type, original_size


# ─── Main Decoder ─────────────────────────────────────────────────────────────


class Decoder:
    """
    Full PixelSonics decoding pipeline.

    Mirrors the Encoder exactly — must be initialised with the same parameters.

    Usage::

        dec = Decoder()
        result = dec.decode_from_waveform(waveform)
        print(result.media_type)   # "text/plain"
        print(result.as_text())    # "Hello, world!"
    """

    def __init__(
        self,
        ecc_nsym: int = 10,
        freq_0: float = FREQ_0,
        freq_1: float = FREQ_1,
        sample_rate: int = SAMPLE_RATE,
        bit_duration: float = BIT_DURATION,
        preamble_bits: int = PREAMBLE_BITS,
    ):
        self.ecc_nsym = ecc_nsym
        self.freq_0 = freq_0
        self.freq_1 = freq_1
        self.sample_rate = sample_rate
        self.bit_duration = bit_duration
        self.preamble_bits = preamble_bits

    # ── Step helpers ─────────────────────────────────────────────────────────

    def demodulate_packet(self, waveform: np.ndarray) -> bytes:
        """Demodulate a single packet's waveform to bytes."""
        return demodulate(
            waveform,
            freq_0=self.freq_0,
            freq_1=self.freq_1,
            sample_rate=self.sample_rate,
            bit_duration=self.bit_duration,
            preamble_bits=self.preamble_bits,
        )

    def ecc_decode(self, data: bytes) -> bytes:
        return ecc_decode(data, nsym=self.ecc_nsym)

    def decompress(self, data: bytes) -> bytes:
        return smart_decompress(data)

    def strip_header(self, data: bytes) -> DecodeResult:
        stream = io.BytesIO(data)
        media_type, original_size = _parse_header(stream)
        payload = stream.read()
        return DecodeResult(
            data=payload,
            media_type=media_type,
            original_size=original_size,
        )

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def decode_from_raw_packets(self, raw_packets: list[bytes]) -> DecodeResult:
        """
        Decode a list of raw (already-demodulated) packet byte strings.

        Args:
            raw_packets: List of framed packet bytes (output of encoder.frame()).

        Returns:
            DecodeResult with recovered data and metadata.
        """
        try:
            assembled = deframe(raw_packets)
        except FramingError as exc:
            raise DecoderError(f"Framing error: {exc}") from exc

        try:
            ecc_checked = self.ecc_decode(assembled)
        except ECCError as exc:
            raise DecoderError(f"ECC error: {exc}") from exc

        try:
            decompressed = self.decompress(ecc_checked)
        except Exception as exc:
            raise DecoderError(f"Decompression error: {exc}") from exc

        return self.strip_header(decompressed)

    def decode_from_waveform(
        self,
        waveform: np.ndarray,
        packet_waveform_size: int | None = None,
    ) -> DecodeResult:
        """
        Full pipeline: audio waveform → recovered data.

        Args:
            waveform:             Float32 mono audio array.
            packet_waveform_size: If known, the number of samples per packet
                                  waveform (for slicing). If None, the entire
                                  waveform is treated as a single packet stream.

        Returns:
            DecodeResult.
        """
        # For now treat entire waveform as one stream (single-packet path).
        # Multi-packet splitting will be driven by framing's SYNC detection.
        raw_bytes = self.demodulate_packet(waveform)
        # raw_bytes is the concatenation of all framed packets – split them.
        # Since deframe() accepts the assembled concatenated bytes, we need to
        # parse individual packets from the byte stream.
        packets = _split_packets(raw_bytes)
        return self.decode_from_raw_packets(packets)

    def decode_from_wav(self, wav_path: str) -> DecodeResult:
        """
        Load a WAV file and run the full decode pipeline.

        Args:
            wav_path: Path to a WAV file produced by Encoder.encode_to_wav().

        Returns:
            DecodeResult.
        """
        import wave

        with wave.open(wav_path, "r") as wf:
            n_frames = wf.getnframes()
            raw_pcm = wf.readframes(n_frames)
            sr = wf.getframerate()

        if sr != self.sample_rate:
            raise DecoderError(
                f"WAV sample rate {sr} does not match decoder sample rate "
                f"{self.sample_rate}"
            )

        pcm = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32767.0
        return self.decode_from_waveform(pcm)


# ─── Packet stream splitter ──────────────────────────────────────────────────


def _split_packets(data: bytes) -> list[bytes]:
    """
    Scan *data* for SYNC words and extract complete packet byte-strings.

    This is the low-level synchronization step: it finds each SYNC_MAGIC
    boundary and slices out exactly one packet at a time.
    """
    packets = []
    i = 0
    while i <= len(data) - (SYNC_SIZE + HEADER_SIZE + CRC_SIZE):
        if data[i : i + SYNC_SIZE] != SYNC_MAGIC:
            i += 1
            continue

        # Found a potential sync word — read the header to know payload length
        header_start = i + SYNC_SIZE
        header_end = header_start + HEADER_SIZE
        if header_end > len(data):
            break

        _seq, _total, length = _struct.unpack(">IHI", data[header_start:header_end])
        packet_end = header_end + length + CRC_SIZE

        if packet_end > len(data):
            break  # truncated packet — stop

        packets.append(data[i:packet_end])
        i = packet_end

    if not packets:
        raise DecoderError(
            "No valid SYNC words found in demodulated byte stream. "
            "Check modulation parameters or signal quality."
        )

    return packets
