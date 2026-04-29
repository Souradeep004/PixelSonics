# PixelSonics — Dockerfile
# Builds a lightweight image for the PixelSonics acoustic data communication stack.

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency list first (leverages Docker layer cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command: show usage help for both entry points
CMD ["python", "-c", "print('PixelSonics container is ready.'); print(); print('Encode a file:  python mainInput.py  <input_file> [output.wav]'); print('Decode a WAV:   python mainOutput.py <encoded.wav> [output_dir]')"]
