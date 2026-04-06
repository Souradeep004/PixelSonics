"""
PixelSonics - Framing Layer
Packs a byte payload into structured packets:  [SYNC | HEADER | DATA | CRC32]
and unpacks them on the receiving side.

Packet layout (all multi-byte integers are big-endian):
┌──────────────┬──────────────────────────────────┬──────────────────┬──────────────┐
│  SYNC (4 B)  │         HEADER (10 B)            │   DATA (N B)     │  CRC32 (4 B) │
├──────────────┼────────────┬─────────┬───────────┼──────────────────┼──────────────┤
│ 0xDEADBEEF  │seq (4 B)   │total(2B)│ length(4B)│  payload bytes   │ CRC32 of all │
└──────────────┴────────────┴─────────┴───────────┴──────────────────┴──────────────┘

Fields:
  SYNC   – Magic number to detect packet start.
  seq    – Packet sequence number (0-based).
  total  – Total number of packets in the transmission.
  length – Number of payload bytes in this packet.
  DATA   – Raw payload chunk.
  CRC32  – CRC32 checksum over SYNC+HEADER+DATA.
"""

import struct
import zlib
from dataclasses import dataclass

# Magic sync word — 4 bytes
SYNC_MAGIC: bytes = b'\xDE\xAD\xBE\xEF'
SYNC_SIZE = 4
HEADER_SIZE = 10      # seq(4) + total(2) + length(4)
CRC_SIZE = 4
OVERHEAD = SYNC_SIZE + HEADER_SIZE + CRC_SIZE   # 18 bytes per packet

# Default maximum payload bytes per packet (tune based on modulation bandwidth)
DEFAULT_CHUNK_SIZE = 128


@dataclass
class Packet:
    seq: int          # Sequence number of this packet
    total: int        # Total packets in the stream
    payload: bytes    # Raw data chunk

    def __repr__(self) -> str:
        return (
            f"Packet(seq={self.seq}/{self.total - 1}, "
            f"payload_len={len(self.payload)} B)"
        )


# ─── Encoding ────────────────────────────────────────────────────────────────

def _crc32(data: bytes) -> int:
    """Return unsigned CRC32 of data."""
    return zlib.crc32(data) & 0xFFFFFFFF


def build_packet(seq: int, total: int, payload: bytes) -> bytes:
    """
    Serialize a single Packet into raw bytes.

    Returns:
        bytes: SYNC + HEADER + DATA + CRC32
    """
    header = struct.pack(">IHI", seq, total, len(payload))
    body = SYNC_MAGIC + header + payload
    crc = struct.pack(">I", _crc32(body))
    return body + crc


def frame(data: bytes, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[bytes]:
    """
    Split *data* into framed packets ready for modulation.

    Args:
        data:       Compressed byte stream to frame.
        chunk_size: Maximum payload bytes per packet.

    Returns:
        List of raw packet byte-strings.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    if not chunks:
        chunks = [b""]          # At least one (empty) packet

    total = len(chunks)
    return [build_packet(seq, total, chunk) for seq, chunk in enumerate(chunks)]


# ─── Decoding ────────────────────────────────────────────────────────────────

class FramingError(Exception):
    """Raised when a packet is malformed or its CRC fails."""


def parse_packet(raw: bytes) -> Packet:
    """
    Deserialize raw bytes back into a Packet, verifying CRC.

    Raises:
        FramingError: On sync mismatch, short packet, or CRC failure.
    """
    min_size = SYNC_SIZE + HEADER_SIZE + CRC_SIZE
    if len(raw) < min_size:
        raise FramingError(
            f"Packet too short: {len(raw)} B (minimum {min_size} B)"
        )

    # Verify SYNC
    if raw[:SYNC_SIZE] != SYNC_MAGIC:
        raise FramingError(
            f"Bad sync word: {raw[:SYNC_SIZE].hex()!r}"
        )

    # Parse header
    seq, total, length = struct.unpack(">IHI", raw[SYNC_SIZE:SYNC_SIZE + HEADER_SIZE])

    expected_size = SYNC_SIZE + HEADER_SIZE + length + CRC_SIZE
    if len(raw) < expected_size:
        raise FramingError(
            f"Packet data truncated: expected {expected_size} B, got {len(raw)} B"
        )

    payload = raw[SYNC_SIZE + HEADER_SIZE: SYNC_SIZE + HEADER_SIZE + length]
    body = raw[:SYNC_SIZE + HEADER_SIZE + length]
    stored_crc, = struct.unpack(">I", raw[SYNC_SIZE + HEADER_SIZE + length:expected_size])

    computed_crc = _crc32(body)
    if computed_crc != stored_crc:
        raise FramingError(
            f"CRC mismatch on packet {seq}: "
            f"expected 0x{stored_crc:08X}, computed 0x{computed_crc:08X}"
        )

    return Packet(seq=seq, total=total, payload=payload)


def deframe(packets: list[bytes]) -> bytes:
    """
    Parse and reassemble a list of raw packet bytes into the original payload.

    Args:
        packets: Raw packet byte strings (order does not need to be sorted).

    Returns:
        Reassembled byte stream.

    Raises:
        FramingError: On parse error or missing packets.
    """
    parsed: dict[int, Packet] = {}
    total = None

    for raw in packets:
        pkt = parse_packet(raw)
        if total is None:
            total = pkt.total
        elif pkt.total != total:
            raise FramingError(
                f"Inconsistent total count: saw {pkt.total}, expected {total}"
            )
        if pkt.seq in parsed:
            raise FramingError(f"Duplicate packet seq={pkt.seq}")
        parsed[pkt.seq] = pkt

    if total is None:
        raise FramingError("No packets provided.")

    missing = [i for i in range(total) if i not in parsed]
    if missing:
        raise FramingError(f"Missing packets: {missing}")

    return b"".join(parsed[i].payload for i in range(total))
