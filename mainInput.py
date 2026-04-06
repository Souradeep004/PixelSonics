"""
PixelSonics - mainInput.py
==========================
Encodes any supported file (TXT, CSV, PDF, image) into a WAV audio file
using the full PixelSonics pipeline:

    Input File → Compress (zstd) → ECC (Reed-Solomon) → Frame → FSK Modulate → WAV

Usage
-----
    python mainInput.py                          # interactive prompts
    python mainInput.py path/to/file.pdf         # file given as argument
    python mainInput.py file.csv output.wav      # file + custom output path

Supported input formats:  .txt  .csv  .pdf  .png  .jpg  .jpeg  .bmp  .gif
Output:  a .wav file that can be stored on disk and fed into mainOutput.py
"""

import sys
import os

# ── Make sure PixelSonics root is on sys.path regardless of where this
#    script lives or how it is invoked. ────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import wave
import time

from encoder.encoder import Encoder

# ─── Supported extensions → human label ──────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".txt":  "Text file",
    ".csv":  "CSV file",
    ".pdf":  "PDF file",
    ".png":  "PNG image",
    ".jpg":  "JPEG image",
    ".jpeg": "JPEG image",
    ".bmp":  "BMP image",
    ".gif":  "GIF image",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_duration(seconds: float) -> str:
    """Human-readable audio duration."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _default_output_path(input_path: str) -> str:
    """Derive default WAV output path from the input filename."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    directory = os.path.dirname(os.path.abspath(input_path))
    return os.path.join(directory, f"{base}_encoded.wav")


def _validate_input(path: str) -> str:
    """Return the resolved path or raise with a helpful message."""
    path = path.strip()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path!r}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext!r}\n"
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return path


def _wav_info(wav_path: str) -> tuple[int, float]:
    """Return (file_size_bytes, duration_seconds) for a WAV file."""
    size = os.path.getsize(wav_path)
    with wave.open(wav_path, "r") as wf:
        duration = wf.getnframes() / wf.getframerate()
    return size, duration


# ─── Core encode function ─────────────────────────────────────────────────────

def encode_file(input_path: str, output_path: str) -> None:
    """
    Run the full PixelSonics encoding pipeline on *input_path* and write
    the resulting WAV file to *output_path*.
    """
    enc = Encoder()

    input_size = os.path.getsize(input_path)
    ext = os.path.splitext(input_path)[1].lower()
    label = SUPPORTED_EXTENSIONS[ext]

    print()
    print("=" * 56)
    print("  PixelSonics Encoder")
    print("=" * 56)
    print(f"  Input  : {input_path}")
    print(f"  Type   : {label}")
    print(f"  Size   : {_fmt_bytes(input_size)}")
    print(f"  Output : {output_path}")
    print("=" * 56)
    print()
    print("  [1/4] Reading and normalising input ...")

    # Step 1 — normalise (Encoder handles type detection from extension)
    raw_bytes, media_type = enc.normalise(input_path)
    print(f"         Media type : {media_type}")

    print("  [2/4] Compressing  (zstd) ...")
    with_header = enc.add_header(raw_bytes, media_type)
    compressed = enc.compress(with_header)
    ratio = len(with_header) / max(len(compressed), 1)
    print(f"         {_fmt_bytes(len(with_header))} → {_fmt_bytes(len(compressed))}  "
          f"(ratio {ratio:.2f}x)")

    print("  [3/4] Applying Reed-Solomon ECC ...")
    ecc_encoded = enc.ecc_encode(compressed)
    overhead = len(ecc_encoded) - len(compressed)
    print(f"         ECC overhead : +{_fmt_bytes(overhead)}")

    print("  [4/4] Framing + FSK modulation → WAV ...")
    t0 = time.time()
    packets = enc.frame(ecc_encoded)
    waveform = enc.modulate_packets(packets)
    elapsed = time.time() - t0

    # Write WAV
    import numpy as np
    import wave as _wave
    pcm = (waveform * 32767).astype(np.int16)
    with _wave.open(output_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(enc.sample_rate)
        wf.writeframes(pcm.tobytes())

    wav_size, duration = _wav_info(output_path)

    print()
    print("=" * 56)
    print("  Encoding complete!")
    print(f"  WAV size    : {_fmt_bytes(wav_size)}")
    print(f"  Duration    : {_fmt_duration(duration)}  ({duration:.1f} s)")
    print(f"  Packets     : {len(packets)}")
    print(f"  Time taken  : {elapsed:.2f} s")
    print(f"  Saved to    : {output_path}")
    print("=" * 56)
    print()
    print("  Feed this WAV file into mainOutput.py to reconstruct the")
    print("  original file.")
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    # ── Resolve input path ────────────────────────────────────────────────────
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    else:
        print()
        print("  PixelSonics Encoder — Interactive Mode")
        print()
        print(f"  Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        print()
        input_path = input("  Enter path to input file: ").strip()
        if not input_path:
            print("  No path entered. Exiting.")
            sys.exit(1)

    try:
        input_path = _validate_input(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  Error: {exc}\n")
        sys.exit(1)

    # ── Resolve output path ───────────────────────────────────────────────────
    if len(sys.argv) >= 3:
        output_path = sys.argv[2].strip()
    else:
        suggested = _default_output_path(input_path)
        print(f"\n  Default output path: {suggested}")
        answer = input("  Press Enter to accept, or type a new path: ").strip()
        output_path = answer if answer else suggested

    if not output_path.lower().endswith(".wav"):
        output_path += ".wav"

    # ── Ensure output directory exists ────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    # ── Run encoding ──────────────────────────────────────────────────────────
    try:
        encode_file(input_path, output_path)
    except Exception as exc:
        print(f"\n  Encoding failed: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
