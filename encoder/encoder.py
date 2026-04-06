"""
PixelSonics - Encoder

Accepts any supported input (text, image path, CSV path, PDF path, or raw bytes)
and produces a transmission-ready audio waveform by running the full pipeline:

  Input → Byte Stream → Compress (zstd) → ECC Encode → Frame → Modulate → Audio

The byte stream is prefixed with a compact JSON header so the decoder can
reconstruct the original file format without out-of-band information.
"""

from __future__ import annotations

import io
import json
import os
import struct

import numpy as np

from compression import compress
from ecc import encode as ecc_encode
from framing import frame, DEFAULT_CHUNK_SIZE
from modulation import (
    modulate,
    SAMPLE_RATE,
    BIT_DURATION,
    PREAMBLE_BITS,
    FREQ_0,
    FREQ_1,
)

# ─── Supported input types ───────────────────────────────────────────────────

SUPPORTED_TYPES = ("text", "image", "csv", "pdf", "bytes")


# ─── Header helpers ──────────────────────────────────────────────────────────


def _build_header(media_type: str, original_size: int) -> bytes:
    """
    Build a length-prefixed JSON header.

    Layout: [header_length : 4 bytes big-endian] [header_json : UTF-8]
    """
    meta = {
        "type": media_type,
        "size": original_size,
    }
    header_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    length_prefix = struct.pack(">I", len(header_json))
    return length_prefix + header_json


# ─── Input normalisation ─────────────────────────────────────────────────────


def _load_text(text: str) -> tuple[bytes, str]:
    return text.encode("utf-8"), "text/plain"


def _load_file(path: str) -> tuple[bytes, str]:
    ext = os.path.splitext(path)[1].lower()
    type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".csv": "text/csv",
        ".pdf": "application/pdf",
    }
    media_type = type_map.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        return f.read(), media_type


def _normalise_input(
    source,
    input_type: str | None = None,
) -> tuple[bytes, str]:
    """
    Return (raw_bytes, media_type) regardless of what *source* is.

    Args:
        source:     str (text or file path) | bytes | file-like object.
        input_type: hint – one of SUPPORTED_TYPES. Auto-detected if None.
    """
    if isinstance(source, bytes):
        return source, "application/octet-stream"

    if hasattr(source, "read"):  # file-like object
        data = source.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return data, "application/octet-stream"

    if isinstance(source, str):
        if input_type == "text":
            return _load_text(source)
        # If it looks like a path that exists, treat as file
        if os.path.exists(source):
            return _load_file(source)
        # Otherwise treat as plain text
        return _load_text(source)

    raise TypeError(
        f"Unsupported source type: {type(source).__name__}. "
        f"Pass str, bytes, or a file-like object."
    )


# ─── Main Encoder ─────────────────────────────────────────────────────────────


class Encoder:
    """
    Full PixelSonics encoding pipeline.

    Usage::

        enc = Encoder()
        waveform = enc.encode_to_waveform("Hello, world!", input_type="text")
        # waveform is a numpy float32 array ready for playback / WAV export
    """

    def __init__(
        self,
        compression_level: int = 3,
        ecc_nsym: int = 10,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        freq_0: float = FREQ_0,
        freq_1: float = FREQ_1,
        sample_rate: int = SAMPLE_RATE,
        bit_duration: float = BIT_DURATION,
        preamble_bits: int = PREAMBLE_BITS,
    ):
        self.compression_level = compression_level
        self.ecc_nsym = ecc_nsym
        self.chunk_size = chunk_size
        self.freq_0 = freq_0
        self.freq_1 = freq_1
        self.sample_rate = sample_rate
        self.bit_duration = bit_duration
        self.preamble_bits = preamble_bits

    # ── Step helpers ─────────────────────────────────────────────────────────

    def normalise(self, source, input_type=None) -> tuple[bytes, str]:
        """Convert any source to (raw_bytes, media_type)."""
        return _normalise_input(source, input_type)

    def add_header(self, raw: bytes, media_type: str) -> bytes:
        """Prepend metadata header to raw bytes."""
        return _build_header(media_type, len(raw)) + raw

    def compress(self, data: bytes) -> bytes:
        return compress(data, level=self.compression_level)

    def ecc_encode(self, data: bytes) -> bytes:
        return ecc_encode(data, nsym=self.ecc_nsym)

    def frame(self, data: bytes) -> list[bytes]:
        return frame(data, chunk_size=self.chunk_size)

    def modulate_packets(self, packets: list[bytes]) -> np.ndarray:
        """Concatenate all packet waveforms into one audio stream."""
        segments = []
        for pkt in packets:
            segments.append(
                modulate(
                    pkt,
                    freq_0=self.freq_0,
                    freq_1=self.freq_1,
                    sample_rate=self.sample_rate,
                    bit_duration=self.bit_duration,
                    preamble_bits=self.preamble_bits,
                )
            )
        return np.concatenate(segments) if segments else np.array([], dtype=np.float32)

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def encode_to_bytes(self, source, input_type: str | None = None) -> bytes:
        """
        Run everything up to (but not including) modulation.

        Returns the ECC-encoded, compressed, framed byte stream
        (concatenated raw packet bytes).
        """
        raw, media_type = self.normalise(source, input_type)
        with_header = self.add_header(raw, media_type)
        compressed = self.compress(with_header)
        ecc_encoded = self.ecc_encode(compressed)
        packets = self.frame(ecc_encoded)
        return b"".join(packets)

    def encode_to_waveform(self, source, input_type: str | None = None) -> np.ndarray:
        """
        Full pipeline → audio waveform (float32 numpy array).

        Args:
            source:     Text string, file path, bytes, or file-like object.
            input_type: Optional hint ("text", "image", "csv", "pdf", "bytes").

        Returns:
            numpy float32 array (mono), amplitude in [-1, 1].
        """
        raw, media_type = self.normalise(source, input_type)
        with_header = self.add_header(raw, media_type)
        compressed = self.compress(with_header)
        ecc_encoded = self.ecc_encode(compressed)
        packets = self.frame(ecc_encoded)
        return self.modulate_packets(packets)

    def encode_to_wav(
        self, source, wav_path: str, input_type: str | None = None
    ) -> str:
        """
        Full pipeline → WAV file on disk.

        Args:
            source:     Input (text, path, bytes, file-like).
            wav_path:   Destination path for the WAV file.
            input_type: Optional format hint.

        Returns:
            The wav_path that was written.
        """
        import wave

        waveform = self.encode_to_waveform(source, input_type)
        pcm = (waveform * 32767).astype(np.int16)

        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())

        return wav_path
