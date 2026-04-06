"""
PixelSonics - Modulation Layer (FSK)

Implements Binary FSK (Frequency Shift Keying):
  Bit 0 → freq_0  (default 18 000 Hz)
  Bit 1 → freq_1  (default 19 000 Hz)

The module converts a bit-stream to/from a numpy float32 audio waveform.

Transmission structure:
  [PREAMBLE tones] [DATA bits]

The preamble is a known alternating sequence used by the demodulator for
synchronisation and automatic threshold calibration.
"""

from __future__ import annotations

import numpy as np

# ─── Defaults ────────────────────────────────────────────────────────────────

SAMPLE_RATE   = 44100     # Hz
FREQ_0        = 18000     # Hz  (represents bit 0)
FREQ_1        = 19000     # Hz  (represents bit 1)
BIT_DURATION  = 0.005     # seconds per bit  (5 ms → ~200 bps raw)
PREAMBLE_BITS = 32        # alternating 0/1 preamble symbols


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _bit_tone(bit: int,
              freq_0: float = FREQ_0,
              freq_1: float = FREQ_1,
              sample_rate: int = SAMPLE_RATE,
              bit_duration: float = BIT_DURATION) -> np.ndarray:
    """Generate one FSK tone for a single bit."""
    freq = freq_1 if bit else freq_0
    n_samples = int(sample_rate * bit_duration)
    t = np.linspace(0, bit_duration, n_samples, endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _bytes_to_bits(data: bytes) -> list[int]:
    """Convert bytes to a list of bits (MSB first)."""
    bits = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def _bits_to_bytes(bits: list[int]) -> bytes:
    """Convert a list of bits (MSB first) back to bytes."""
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for shift, b in zip(range(7, -1, -1), bits[i:i + 8]):
            byte |= b << shift
        out.append(byte)
    return bytes(out)


# ─── Modulation ──────────────────────────────────────────────────────────────

def modulate(
    data: bytes,
    freq_0: float = FREQ_0,
    freq_1: float = FREQ_1,
    sample_rate: int = SAMPLE_RATE,
    bit_duration: float = BIT_DURATION,
    preamble_bits: int = PREAMBLE_BITS,
) -> np.ndarray:
    """
    Convert raw bytes into an FSK audio waveform (float32, mono).

    Args:
        data:          Bytes to transmit (already framed + ECC-encoded).
        freq_0:        Frequency for bit 0 (Hz).
        freq_1:        Frequency for bit 1 (Hz).
        sample_rate:   Audio sample rate (Hz).
        bit_duration:  Duration of each bit symbol (seconds).
        preamble_bits: Number of alternating preamble symbols to prepend.

    Returns:
        numpy float32 array (mono waveform), amplitude in [-1, 1].
    """
    # Build preamble: alternating 0, 1, 0, 1 …
    preamble = [i % 2 for i in range(preamble_bits)]
    bits = preamble + _bytes_to_bits(data)

    tones = [
        _bit_tone(b, freq_0, freq_1, sample_rate, bit_duration)
        for b in bits
    ]
    waveform = np.concatenate(tones)
    return waveform


# ─── Demodulation ────────────────────────────────────────────────────────────

def _goertzel(samples: np.ndarray, freq: float, sample_rate: int) -> float:
    """
    Goertzel algorithm: compute the power of a single frequency in *samples*.
    Much cheaper than a full FFT for checking just two frequencies.
    """
    N = len(samples)
    k = int(0.5 + N * freq / sample_rate)
    omega = 2 * np.pi * k / N
    coeff = 2 * np.cos(omega)
    s_prev, s_prev2 = 0.0, 0.0
    for s in samples:
        s_curr = s + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s_curr
    power = s_prev2 ** 2 + s_prev ** 2 - coeff * s_prev * s_prev2
    return float(power)


def demodulate(
    waveform: np.ndarray,
    freq_0: float = FREQ_0,
    freq_1: float = FREQ_1,
    sample_rate: int = SAMPLE_RATE,
    bit_duration: float = BIT_DURATION,
    preamble_bits: int = PREAMBLE_BITS,
) -> bytes:
    """
    Recover bytes from an FSK audio waveform.

    The function:
      1. Slices the waveform into symbol windows.
      2. Uses the Goertzel algorithm to decide 0 or 1 per window.
      3. Strips the known preamble symbols.
      4. Reassembles bytes.

    Args:
        waveform:      Float32 mono audio array.
        freq_0:        Frequency for bit 0 (Hz).
        freq_1:        Frequency for bit 1 (Hz).
        sample_rate:   Audio sample rate (Hz).
        bit_duration:  Duration of each bit symbol (seconds).
        preamble_bits: Number of preamble symbols to strip.

    Returns:
        Decoded bytes.
    """
    samples_per_bit = int(sample_rate * bit_duration)
    n_symbols = len(waveform) // samples_per_bit

    bits = []
    for i in range(n_symbols):
        segment = waveform[i * samples_per_bit:(i + 1) * samples_per_bit]
        p0 = _goertzel(segment, freq_0, sample_rate)
        p1 = _goertzel(segment, freq_1, sample_rate)
        bits.append(1 if p1 > p0 else 0)

    # Strip preamble
    data_bits = bits[preamble_bits:]
    return _bits_to_bytes(data_bits)


# ─── Utility ─────────────────────────────────────────────────────────────────

def estimated_duration(data: bytes,
                       bit_duration: float = BIT_DURATION,
                       preamble_bits: int = PREAMBLE_BITS) -> float:
    """Return the estimated audio duration (seconds) for *data*."""
    n_bits = len(data) * 8 + preamble_bits
    return n_bits * bit_duration
