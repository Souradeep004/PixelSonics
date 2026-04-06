import io
import json
import struct
import sys
import os
import time

from decoder.decoder import Decoder, DecoderError

"""
PixelSonics - mainOutput.py
===========================
Decodes a WAV file produced by mainInput.py and reconstructs the original
file using the full PixelSonics reverse pipeline:

    WAV → FSK Demodulate → Deframe → ECC Decode (RS) → Decompress → Original File

Usage
-----
    python mainOutput.py                             # interactive prompts
    python mainOutput.py path/to/encoded.wav         # WAV given as argument
    python mainOutput.py encoded.wav output_dir/     # WAV + custom output directory
    python mainOutput.py encoded.wav recovered.pdf   # WAV + explicit output file path

The original file extension is inferred automatically from the media-type
header embedded inside the WAV — no need to specify it manually.
"""


# ── Make sure PixelSonics root is on sys.path regardless of where this
#    script lives or how it is invoked. ────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ─── Media-type → file extension map ─────────────────────────────────────────

_MEDIA_TO_EXT = {
    "text/plain":            ".txt",
    "text/csv":              ".csv",
    "application/pdf":       ".pdf",
    "image/png":             ".png",
    "image/jpeg":            ".jpg",
    "image/gif":             ".gif",
    "image/bmp":             ".bmp",
    "application/octet-stream": ".bin",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _ext_for_media_type(media_type: str) -> str:
    """Return a file extension (with leading dot) for the given media type."""
    return _MEDIA_TO_EXT.get(media_type.lower().split(";")[0].strip(), ".bin")


def _default_output_path(wav_path: str, media_type: str) -> str:
    """Derive a sensible output path from the WAV filename and media type."""
    base = os.path.basename(wav_path)
    # Strip common encoder suffixes so "report_encoded.wav" → "report"
    for suffix in ("_encoded", "_tx", "_out"):
        if base.lower().endswith(suffix + ".wav"):
            base = base[:-(len(suffix) + 4)]
            break
    else:
        base = os.path.splitext(base)[0]

    ext = _ext_for_media_type(media_type)
    directory = os.path.dirname(os.path.abspath(wav_path))
    return os.path.join(directory, f"{base}_recovered{ext}")


def _resolve_output_path(user_path: str, media_type: str) -> str:
    """
    If *user_path* is a directory, place the recovered file inside it.
    If it has no extension, append the correct one for *media_type*.
    """
    if os.path.isdir(user_path):
        ext = _ext_for_media_type(media_type)
        return os.path.join(user_path, f"recovered{ext}")
    _, ext = os.path.splitext(user_path)
    if not ext:
        user_path += _ext_for_media_type(media_type)
    return user_path


# ─── Core decode function ─────────────────────────────────────────────────────

def decode_file(wav_path: str, output_path: str) -> None:
    """
    Run the full PixelSonics decoding pipeline on *wav_path* and write the
    recovered original file to *output_path*.
    """
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path!r}")
    if not wav_path.lower().endswith(".wav"):
        raise ValueError(f"Expected a .wav file, got: {wav_path!r}")

    wav_size = os.path.getsize(wav_path)

    print()
    print("=" * 56)
    print("  PixelSonics Decoder")
    print("=" * 56)
    print(f"  Input WAV : {wav_path}")
    print(f"  WAV size  : {_fmt_bytes(wav_size)}")
    print("=" * 56)
    print()

    dec = Decoder()

    print("  [1/4] Loading WAV and demodulating (FSK) ...")
    t0 = time.time()

    import wave as _wave
    import numpy as np

    with _wave.open(wav_path, "r") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / sample_rate
        raw_pcm = wf.readframes(n_frames)

    print(f"         Sample rate : {sample_rate} Hz")
    print(f"         Duration    : {duration:.1f} s  ({n_frames:,} samples)")

    if sample_rate != dec.sample_rate:
        raise DecoderError(
            f"WAV sample rate ({sample_rate} Hz) does not match decoder "
            f"({dec.sample_rate} Hz). Was this file made by PixelSonics?"
        )

    pcm = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32767.0

    print("  [2/4] Splitting packets and deframing ...")

    raw_bytes = dec.demodulate_packet(pcm)

    from decoder.decoder import _split_packets
    packets = _split_packets(raw_bytes)
    print(f"         Found {len(packets)} packet(s)")

    from framing import deframe
    assembled = deframe(packets)
    print(f"         Assembled   : {_fmt_bytes(len(assembled))}")

    print("  [3/4] Applying Reed-Solomon ECC decode ...")
    ecc_checked = dec.ecc_decode(assembled)
    print(f"         After ECC   : {_fmt_bytes(len(ecc_checked))}")

    print("  [4/4] Decompressing (zstd) and reading header ...")
    decompressed = dec.decompress(ecc_checked)

    stream = io.BytesIO(decompressed)
    length_bytes = stream.read(4)
    header_len, = struct.unpack(">I", length_bytes)
    header_json = stream.read(header_len)
    meta = json.loads(header_json.decode("utf-8"))
    media_type = meta.get("type", "application/octet-stream")
    declared_size = meta.get("size", -1)
    payload = stream.read()

    elapsed = time.time() - t0

    print(f"         Media type  : {media_type}")
    print(f"         Declared sz : {_fmt_bytes(declared_size)}")
    print(f"         Recovered   : {_fmt_bytes(len(payload))}")

    # ── Integrity check ───────────────────────────────────────────────────────
    if declared_size >= 0 and len(payload) != declared_size:
        print(f"\n  WARNING: size mismatch! "
              f"Expected {declared_size} B, got {len(payload)} B.")
        print("           The file may be corrupted or truncated.\n")
    else:
        print("         Integrity   : OK (size matches header)")

    # ── Resolve final output path now that we know media_type ─────────────────
    output_path = _resolve_output_path(output_path, media_type)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(payload)

    print()
    print("=" * 56)
    print("  Decoding complete!")
    print(f"  Recovered type : {media_type}")
    print(f"  Recovered size : {_fmt_bytes(len(payload))}")
    print(f"  Time taken     : {elapsed:.2f} s")
    print(f"  Saved to       : {output_path}")
    print("=" * 56)
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    # ── Resolve WAV input path ────────────────────────────────────────────────
    if len(sys.argv) >= 2:
        wav_path = sys.argv[1].strip()
    else:
        print()
        print("  PixelSonics Decoder — Interactive Mode")
        print()
        wav_path = input("  Enter path to encoded WAV file: ").strip()
        if not wav_path:
            print("  No path entered. Exiting.")
            sys.exit(1)

    if not os.path.isfile(wav_path):
        print(f"\n  Error: File not found: {wav_path!r}\n")
        sys.exit(1)

    # ── Peek at the WAV to find the media type (needed for default out path) ──
    # We do a minimal pre-decode just to read the header for the suggested name.
    # The full decode in decode_file() will redo this cleanly.

    # ── Resolve output path ───────────────────────────────────────────────────
    if len(sys.argv) >= 3:
        output_path = sys.argv[2].strip()
    else:
        # We won't know the extension until decode is done, so ask for a dir
        # or a full path.
        wav_dir = os.path.dirname(os.path.abspath(wav_path))
        print(f"\n  Default output directory: {wav_dir}")
        answer = input(
            "  Press Enter to save there, or provide a directory / file path: "
        ).strip()
        output_path = answer if answer else wav_dir

    # ── Run decoding ──────────────────────────────────────────────────────────
    try:
        decode_file(wav_path, output_path)
    except (DecoderError, FileNotFoundError, ValueError) as exc:
        print(f"\n  Error: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        print(f"\n  Unexpected error: {exc}\n")
        raise


if __name__ == "__main__":
    main()
