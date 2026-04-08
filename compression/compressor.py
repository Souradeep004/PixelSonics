"""
PixelSonics - Compression Layer
Uses Zstandard (zstd) for lossless compression/decompression of byte streams.

Smart compression
─────────────────
PDFs and images (JPEG, PNG, GIF …) are already compressed internally.
Feeding them through zstd produces output that is the *same size or larger*,
wasting transmission bandwidth with no benefit.

smart_compress() solves this by:
  1. Attempting zstd compression.
  2. Comparing compressed size to the original.
  3. Keeping whichever is smaller.
  4. Prepending a 1-byte flag so smart_decompress() knows what to do:
       0x01  → payload is zstd-compressed  (decompress before use)
       0x00  → payload is raw / uncompressed (pass through as-is)

This flag adds just 1 byte of overhead but makes the codec fully self-describing.
"""

import zstandard as zstd

# Default compression level (1=fast, 22=max). Level 3 is a good balance.
DEFAULT_LEVEL = 3

# 1-byte flags stored as the first byte of smart_compress() output
_FLAG_RAW = b"\x00"  # data was NOT compressed (already dense)
_FLAG_COMPRESSED = b"\x01"  # data IS zstd-compressed


# ─── Low-level zstd wrappers ──────────────────────────────────────────────────


def compress(data: bytes, level: int = DEFAULT_LEVEL) -> bytes:
    """
    Compress raw bytes using Zstandard (unconditionally).

    Args:
        data:  The raw byte stream to compress.
        level: Compression level (1–22). Higher = smaller output, slower.

    Returns:
        Compressed bytes (zstd frame, no flag byte).

    Raises:
        TypeError: If data is not bytes.
    """
    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, got {type(data).__name__}")
    return zstd.ZstdCompressor(level=level).compress(data)


def decompress(data: bytes) -> bytes:
    """
    Decompress a Zstandard-compressed byte stream (unconditionally).

    Args:
        data: Compressed bytes (must be a valid zstd frame, no flag byte).

    Returns:
        Original uncompressed bytes.

    Raises:
        TypeError: If data is not bytes.
        zstd.ZstdError: If decompression fails.
    """
    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, got {type(data).__name__}")
    return zstd.ZstdDecompressor().decompress(data)


# ─── Smart compress / decompress (recommended for pipeline use) ───────────────


def smart_compress(data: bytes, level: int = DEFAULT_LEVEL) -> tuple[bytes, bool]:
    """
    Compress *data* only if doing so actually reduces its size.

    Tries zstd compression and keeps whichever result (compressed or raw)
    is smaller.  Prepends a 1-byte flag so smart_decompress() is
    self-describing — no out-of-band signalling needed.

    Args:
        data:  Raw bytes to (maybe) compress.
        level: zstd compression level (1–22).

    Returns:
        (encoded_bytes, was_compressed)
          encoded_bytes  — flag byte + payload, ready to pass to smart_decompress()
          was_compressed — True if zstd compression was applied, False if raw pass-through

    Example::
        encoded, compressed = smart_compress(pdf_bytes)
        # compressed == False for a typical PDF
        original = smart_decompress(encoded)
        assert original == pdf_bytes
    """
    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, got {type(data).__name__}")

    compressed = zstd.ZstdCompressor(level=level).compress(data)

    if len(compressed) < len(data):
        # Compression helped — use it
        return _FLAG_COMPRESSED + compressed, True
    else:
        # Compression made it the same size or larger — skip it
        return _FLAG_RAW + data, False


def smart_decompress(data: bytes) -> bytes:
    """
    Reverse of smart_compress().  Reads the 1-byte flag and either
    decompresses (zstd) or strips the flag and returns the raw payload.

    Args:
        data: Output of smart_compress() — flag byte + payload.

    Returns:
        Original uncompressed bytes.

    Raises:
        ValueError: If the flag byte is unrecognised.
        zstd.ZstdError: If decompression of a compressed payload fails.
    """
    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, got {type(data).__name__}")
    if len(data) < 1:
        raise ValueError("smart_decompress: data is empty (no flag byte).")

    flag = data[:1]
    payload = data[1:]

    if flag == _FLAG_COMPRESSED:
        return zstd.ZstdDecompressor().decompress(payload)
    elif flag == _FLAG_RAW:
        return payload
    else:
        raise ValueError(
            f"smart_decompress: unknown flag byte 0x{flag.hex().upper()}. "
            "Was this data produced by smart_compress()?"
        )


# ─── Utility ──────────────────────────────────────────────────────────────────


def compression_ratio(original: bytes, encoded: bytes) -> float:
    """
    Calculate the effective compression ratio.

    Works with both compress() and smart_compress() output.
    For smart_compress() the 1-byte flag is included in *encoded*.

    Returns:
        Float. >1.0 means the output is smaller than the input (good).
        ~1.0 means no size change. <1.0 means the output is larger (bad).
    """
    if len(encoded) == 0:
        raise ValueError("Encoded data has zero length.")
    return len(original) / len(encoded)


def compress_stream(input_stream, output_stream, level: int = DEFAULT_LEVEL) -> int:
    """
    Compress a file-like input stream into a file-like output stream (unconditional zstd).

    Returns:
        Number of *input* bytes processed.
    """
    compressor = zstd.ZstdCompressor(level=level)
    bytes_written = 0
    with compressor.stream_writer(output_stream) as writer:
        while True:
            chunk = input_stream.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            bytes_written += len(chunk)
    return bytes_written


def decompress_stream(input_stream, output_stream) -> int:
    """
    Decompress a zstd-compressed input stream into an output stream (unconditional).

    Returns:
        Number of bytes written to output_stream.
    """
    decompressor = zstd.ZstdDecompressor()
    bytes_written = 0
    with decompressor.stream_reader(input_stream) as reader:
        while True:
            chunk = reader.read(65536)
            if not chunk:
                break
            output_stream.write(chunk)
            bytes_written += len(chunk)
    return bytes_written
