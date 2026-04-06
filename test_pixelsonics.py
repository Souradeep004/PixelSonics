import os
import sys

import unittest

import numpy as np

from compression import compress, compression_ratio, decompress
from decoder import Decoder
from ecc import ECCError
from ecc import decode as ecc_decode
from ecc import encode as ecc_encode
from ecc.ecc import _RS_AVAILABLE, ReedSolomonECC, xor_decode, xor_encode
from encoder import Encoder
from framing import FramingError, deframe, frame
from modulation import demodulate, modulate

"""
PixelSonics - Test Suite

Run with:  python -m pytest test/ -v
Or:        python test/test_pixelsonics.py
"""


# Ensure the PixelSonics project root (parent of test/) is on sys.path.
# os.path.abspath + dirname twice works on Windows and Unix alike.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # …/PixelSonics/test
_ROOT_DIR = os.path.dirname(_TEST_DIR)  # …/PixelSonics
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Compression Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCompression(unittest.TestCase):
    def test_roundtrip_text(self):
        """Compress and decompress plain text."""
        original = b"Hello, PixelSonics! " * 100
        compressed = compress(original)
        self.assertLess(
            len(compressed), len(original), "Compression should reduce size"
        )
        recovered = decompress(compressed)
        self.assertEqual(recovered, original)

    def test_roundtrip_binary(self):
        """Compress and decompress random binary data."""
        original = os.urandom(1024)
        compressed = compress(original)
        recovered = decompress(compressed)
        self.assertEqual(recovered, original)

    def test_empty_bytes(self):
        """Compress empty byte string."""
        compressed = compress(b"")
        recovered = decompress(compressed)
        self.assertEqual(recovered, b"")

    def test_compression_ratio(self):
        """Repetitive data should compress well."""
        original = b"AAAA" * 1000
        compressed = compress(original)
        ratio = compression_ratio(original, compressed)
        self.assertGreater(
            ratio, 5.0, "Repetitive data should achieve > 5x compression"
        )

    def test_type_error(self):
        """Non-bytes input should raise TypeError."""
        with self.assertRaises(TypeError):
            compress("not bytes")

    def test_invalid_decompress(self):
        """Corrupted data should raise an error on decompress."""
        with self.assertRaises(Exception):
            decompress(b"\x00\x01\x02\x03\x04")

    def test_high_compression_level(self):
        """Level 22 should give better compression than level 1 on text."""
        data = b"redundant data " * 500
        c1 = compress(data, level=1)
        c22 = compress(data, level=22)
        self.assertLessEqual(len(c22), len(c1))
        # Both decompress correctly
        self.assertEqual(decompress(c1), data)
        self.assertEqual(decompress(c22), data)


# ─────────────────────────────────────────────────────────────────────────────
# Framing Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFraming(unittest.TestCase):
    def test_roundtrip_small(self):
        """Frame and deframe small payload."""
        data = b"Short payload."
        packets = frame(data, chunk_size=8)
        recovered = deframe(packets)
        self.assertEqual(recovered, data)

    def test_roundtrip_exact_chunk(self):
        """Payload exactly one chunk size."""
        data = b"A" * 64
        packets = frame(data, chunk_size=64)
        self.assertEqual(len(packets), 1)
        recovered = deframe(packets)
        self.assertEqual(recovered, data)

    def test_roundtrip_multi_packet(self):
        """Payload splits across multiple packets."""
        data = bytes(range(256))
        packets = frame(data, chunk_size=32)
        self.assertEqual(len(packets), 8)
        recovered = deframe(packets)
        self.assertEqual(recovered, data)

    def test_out_of_order(self):
        """deframe() should handle out-of-order packets."""
        data = b"ABCDEFGH" * 10
        packets = frame(data, chunk_size=16)
        shuffled = list(reversed(packets))
        recovered = deframe(shuffled)
        self.assertEqual(recovered, data)

    def test_empty_payload(self):
        """Empty input should produce one empty packet."""
        packets = frame(b"", chunk_size=64)
        self.assertEqual(len(packets), 1)
        recovered = deframe(packets)
        self.assertEqual(recovered, b"")

    def test_crc_corruption_detected(self):
        """Flip a byte in the data area — CRC should catch it."""
        data = b"Sensitive data!"
        packets = frame(data, chunk_size=64)
        raw = bytearray(packets[0])
        raw[20] ^= 0xFF  # corrupt a byte
        with self.assertRaises(FramingError):
            deframe([bytes(raw)])

    def test_missing_packet(self):
        """Missing packet sequence should raise FramingError."""
        data = bytes(range(200))
        packets = frame(data, chunk_size=40)
        self.assertGreater(len(packets), 2)
        with self.assertRaises(FramingError):
            deframe(packets[:-1])  # drop last packet

    def test_duplicate_packet(self):
        """Duplicate packet should raise FramingError."""
        data = b"duplicate test" * 10
        packets = frame(data, chunk_size=16)
        with self.assertRaises(FramingError):
            deframe([packets[0], packets[0]] + packets[1:])


# ─────────────────────────────────────────────────────────────────────────────
# ECC Tests  (Reed-Solomon via reedsolo)
# ─────────────────────────────────────────────────────────────────────────────


class TestECC(unittest.TestCase):
    def test_rs_available(self):
        """reedsolo must be installed for these tests to be meaningful."""
        self.assertTrue(
            _RS_AVAILABLE, "reedsolo not installed — run: pip install reedsolo"
        )

    def test_roundtrip_short(self):
        """RS encode/decode short payload."""
        data = b"Hello Reed-Solomon!"
        encoded = ecc_encode(data)
        self.assertEqual(ecc_decode(encoded), data)

    def test_roundtrip_long(self):
        """RS encode/decode payload spanning multiple blocks (>245 bytes)."""
        data = bytes(range(256)) * 3  # 768 bytes → 4 blocks
        encoded = ecc_encode(data)
        self.assertEqual(ecc_decode(encoded), data)

    def test_roundtrip_empty(self):
        """RS encode/decode empty bytes."""
        encoded = ecc_encode(b"")
        self.assertEqual(ecc_decode(encoded), b"")

    def test_encoded_larger_than_original(self):
        """Encoded output must be larger than input."""
        data = b"X" * 100
        encoded = ecc_encode(data)
        self.assertGreater(len(encoded), len(data))

    def test_error_correction(self):
        """RS must correct up to nsym//2 corrupted bytes per block."""
        nsym = 10
        rs = ReedSolomonECC(nsym=nsym)
        data = b"Correctable error test! " * 5
        encoded = bytearray(rs.encode(data))

        # Corrupt nsym//2 = 5 bytes in the first block
        for i in range(5):
            encoded[i] ^= 0xFF

        recovered = rs.decode(bytes(encoded))
        self.assertEqual(recovered, data)

    def test_too_many_errors_raises(self):
        """More than nsym//2 errors per block should raise ECCError."""
        nsym = 10
        rs = ReedSolomonECC(nsym=nsym)
        data = b"Uncorrectable!" * 20
        encoded = bytearray(rs.encode(data))

        # Corrupt nsym//2 + 1 = 6 bytes — beyond correction capacity
        for i in range(6):
            encoded[i] ^= 0xFF

        with self.assertRaises(ECCError):
            rs.decode(bytes(encoded))

    def test_different_nsym_values(self):
        """RS roundtrip works for various nsym values."""
        data = b"nsym variation test " * 10
        for nsym in (4, 10, 20, 32):
            rs = ReedSolomonECC(nsym=nsym)
            encoded = rs.encode(data)
            self.assertEqual(rs.decode(encoded), data, f"Failed for nsym={nsym}")

    def test_overhead_bytes(self):
        """overhead_bytes() returns a value larger than the input size."""
        rs = ReedSolomonECC(nsym=10)
        self.assertGreater(rs.overhead_bytes(100), 0)

    def test_max_correctable_errors(self):
        """max_correctable_errors() == nsym // 2."""
        for nsym in (4, 10, 20):
            rs = ReedSolomonECC(nsym=nsym)
            self.assertEqual(rs.max_correctable_errors(), nsym // 2)

    def test_xor_fallback_roundtrip(self):
        """XOR parity fallback still works independently."""
        data = bytes(range(128))
        encoded = xor_encode(data)
        self.assertEqual(xor_decode(encoded), data)

    def test_xor_fallback_error_detected(self):
        """XOR fallback detects a flipped byte."""
        data = b"XOR parity test data!!!!!!!!!!!"
        encoded = bytearray(xor_encode(data))
        encoded[0] ^= 0x01
        with self.assertRaises(ECCError):
            xor_decode(bytes(encoded))


# ─────────────────────────────────────────────────────────────────────────────
# Modulation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModulation(unittest.TestCase):
    def test_waveform_type(self):
        """modulate() should return a float32 numpy array."""
        waveform = modulate(b"\xab")
        self.assertIsInstance(waveform, np.ndarray)
        self.assertEqual(waveform.dtype, np.float32)

    def test_waveform_amplitude(self):
        """Waveform should stay within [-1, 1]."""
        waveform = modulate(b"\x00\xff")
        self.assertLessEqual(np.max(np.abs(waveform)), 1.0 + 1e-6)

    def test_roundtrip_single_byte(self):
        """Modulate and demodulate a single byte."""
        for byte_val in (0x00, 0xFF, 0xAA, 0x55):
            data = bytes([byte_val])
            waveform = modulate(data)
            recovered = demodulate(waveform)
            self.assertEqual(recovered, data, f"Failed for byte 0x{byte_val:02X}")

    def test_roundtrip_multi_byte(self):
        """Modulate and demodulate several bytes."""
        data = bytes(range(8))
        waveform = modulate(data)
        recovered = demodulate(waveform)
        self.assertEqual(recovered, data)

    def test_waveform_length(self):
        """Waveform length scales with data size."""
        from modulation import BIT_DURATION, PREAMBLE_BITS, SAMPLE_RATE

        data = b"\xde\xad"  # 2 bytes = 16 bits
        expected_samples = int(SAMPLE_RATE * BIT_DURATION) * (16 + PREAMBLE_BITS)
        waveform = modulate(data)
        self.assertEqual(len(waveform), expected_samples)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Pipeline Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def _roundtrip(self, source, input_type=None) -> bytes:
        """Run full encode→decode and return recovered bytes."""
        waveform = self.encoder.encode_to_waveform(source, input_type)
        result = self.decoder.decode_from_waveform(waveform)
        return result.data

    def test_text_roundtrip(self):
        """Short text survives the full pipeline."""
        text = "Hello from PixelSonics!"
        recovered = self._roundtrip(text, input_type="text")
        self.assertEqual(recovered.decode("utf-8"), text)

    def test_text_media_type(self):
        """Media type header is preserved through the pipeline."""
        waveform = self.encoder.encode_to_waveform("type test", input_type="text")
        result = self.decoder.decode_from_waveform(waveform)
        self.assertEqual(result.media_type, "text/plain")

    def test_binary_roundtrip(self):
        """Raw binary bytes survive the full pipeline."""
        data = bytes(range(64))
        recovered = self._roundtrip(data)
        self.assertEqual(recovered, data)

    def test_original_size_header(self):
        """original_size in the header matches the actual payload size."""
        data = b"size check " * 5
        waveform = self.encoder.encode_to_waveform(data)
        result = self.decoder.decode_from_waveform(waveform)
        self.assertEqual(result.original_size, len(data))

    def test_wav_file_roundtrip(self):
        """Encode to WAV → decode from WAV roundtrip."""
        import tempfile

        text = "WAV file roundtrip test!"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            self.encoder.encode_to_wav(text, wav_path, input_type="text")
            result = self.decoder.decode_from_wav(wav_path)
            self.assertEqual(result.data.decode("utf-8"), text)
        finally:
            os.unlink(wav_path)

    def test_csv_like_data(self):
        """CSV-style text roundtrip."""
        csv_data = "name,age,city\nAlice,30,NYC\nBob,25,LA\n"
        recovered = self._roundtrip(csv_data.encode("utf-8"))
        self.assertEqual(recovered.decode("utf-8"), csv_data)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
