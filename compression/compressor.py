"""
PixelSonics - Compression Layer
Uses Zstandard (zstd) for lossless compression/decompression of byte streams.
"""

import zstandard as zstd


# Default compression level (1=fast, 22=max). Level 3 is a good balance.
DEFAULT_LEVEL = 3


def compress(data: bytes, level: int = DEFAULT_LEVEL) -> bytes:
    """
    Compress raw bytes using Zstandard.

    Args:
        data:  The raw byte stream to compress.
        level: Compression level (1–22). Higher = smaller output, slower.

    Returns:
        Compressed bytes.

    Raises:
        TypeError: If data is not bytes.
        zstd.ZstdError: If compression fails.
    """
    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, got {type(data).__name__}")

    compressor = zstd.ZstdCompressor(level=level)
    compressed = compressor.compress(data)
    return compressed


def decompress(data: bytes) -> bytes:
    """
    Decompress a Zstandard-compressed byte stream.

    Args:
        data: Compressed bytes (must have been compressed with zstd).

    Returns:
        Original uncompressed bytes.

    Raises:
        TypeError: If data is not bytes.
        zstd.ZstdError: If decompression fails (e.g. corrupted data).
    """
    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, got {type(data).__name__}")

    decompressor = zstd.ZstdDecompressor()
    decompressed = decompressor.decompress(data)
    return decompressed


def compression_ratio(original: bytes, compressed: bytes) -> float:
    """
    Calculate the compression ratio.

    Returns:
        Float >= 1.0. A ratio of 2.0 means the compressed data is half the size.
    """
    if len(compressed) == 0:
        raise ValueError("Compressed data has zero length.")
    return len(original) / len(compressed)


def compress_stream(input_stream, output_stream, level: int = DEFAULT_LEVEL) -> int:
    """
    Compress data from a file-like input stream into a file-like output stream.

    Args:
        input_stream:  Readable binary stream (e.g. open file, BytesIO).
        output_stream: Writable binary stream.
        level:         Compression level.

    Returns:
        Number of bytes written to output_stream.
    """
    compressor = zstd.ZstdCompressor(level=level)
    bytes_written = 0
    with compressor.stream_writer(output_stream) as writer:
        while True:
            chunk = input_stream.read(65536)  # 64 KB chunks
            if not chunk:
                break
            writer.write(chunk)
            bytes_written += len(chunk)
    return bytes_written


def decompress_stream(input_stream, output_stream) -> int:
    """
    Decompress a zstd-compressed input stream into an output stream.

    Args:
        input_stream:  Readable binary stream containing compressed data.
        output_stream: Writable binary stream.

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
