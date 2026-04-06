# PixelSonics 🔊

An acoustic data communication stack — transmit any file over audio (speaker → microphone).

## Architecture

```
Input (text/image/CSV/PDF/bytes)
        ↓
  Byte Serialisation  ←── encoder/encoder.py
        ↓
  Compression (zstd)  ←── compression/compressor.py
        ↓
  ECC Encode          ←── ecc/ecc.py
        ↓
  Framing             ←── framing/framer.py   [SYNC|HEADER|DATA|CRC32]
        ↓
  FSK Modulation      ←── modulation/fsk.py
        ↓
  Audio Signal (WAV / numpy array)
════════════════════════════════════
  Microphone / WAV file
        ↓
  FSK Demodulation
        ↓
  Deframing
        ↓
  ECC Decode
        ↓
  Decompression (zstd)
        ↓
  Header Parsing
        ↓
  Original File  ←── decoder/decoder.py
```

## Project Layout

```
PixelSonics/
├── encoder/       # Encoder class — full TX pipeline
├── decoder/       # Decoder class — full RX pipeline
├── modulation/    # FSK modulate / demodulate
├── compression/   # zstd compress / decompress
├── framing/       # Packet framer / deframer
├── ecc/           # XOR parity (Phase 1) + RS stub (Phase 2)
└── test/          # Unit + integration tests
```

## Quick Start

### Installation

```bash
pip install zstandard numpy
```

### Transmit Text

```python
import sys
sys.path.insert(0, ".")          # run from PixelSonics/

from encoder import Encoder
from decoder import Decoder

enc = Encoder()
dec = Decoder()

# Encode text → WAV
enc.encode_to_wav("Hello, world!", "hello.wav", input_type="text")

# Decode WAV → text
result = dec.decode_from_wav("hello.wav")
print(result.as_text())          # Hello, world!
print(result.media_type)         # text/plain
```

### In-Memory Round-Trip

```python
waveform = enc.encode_to_waveform("Test message", input_type="text")
result = dec.decode_from_waveform(waveform)
print(result.data)
```

### Transmit a File

```python
enc.encode_to_wav("data/report.csv", "report_tx.wav")
result = dec.decode_from_wav("report_tx.wav")
result.save("data/report_recovered.csv")
```

## Running Tests

```bash
cd PixelSonics
python test/test_pixelsonics.py
# or
python -m pytest test/ -v
```

## Modulation Parameters

| Parameter      | Default   | Description                        |
|---------------|-----------|------------------------------------|
| `FREQ_0`      | 18 000 Hz | Frequency for bit 0                |
| `FREQ_1`      | 19 000 Hz | Frequency for bit 1                |
| `SAMPLE_RATE` | 44 100 Hz | Audio sample rate                  |
| `BIT_DURATION`| 5 ms      | Duration per symbol (~200 bps raw) |
| `PREAMBLE`    | 32 bits   | Sync preamble symbols              |
 
