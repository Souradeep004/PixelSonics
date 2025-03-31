from flask import Flask, render_template, request, jsonify
import numpy as np
import sounddevice as sd
from scipy.signal import hilbert
import speech_recognition as sr

app = Flask(__name__)

# Parameters for FSK
SAMPLE_RATE = 44100
DURATION = 0.02
FREQ_0 = 2000
FREQ_1 = 4000

# Convert text to binary
def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

# Generate FSK-modulated waveform
def generate_fsk_wave(binary_data):
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    waveform = np.concatenate([
        0.5 * np.sin(2 * np.pi * (FREQ_1 if bit == '1' else FREQ_0) * t)
        for bit in binary_data
    ])
    return waveform

# Play sound
@app.route('/encode', methods=['POST'])
def encode_sound():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    binary_data = text_to_binary(text)
    waveform = generate_fsk_wave(binary_data)
    
    sd.play(waveform, samplerate=SAMPLE_RATE)
    sd.wait()

    return jsonify({"message": "Sound played successfully"})

# Decode FSK waveform back to binary
def fsk_wave_to_binary(waveform):
    samples_per_bit = int(SAMPLE_RATE * DURATION)
    binary_data = ""

    for i in range(0, len(waveform), samples_per_bit):
        segment = waveform[i:i+samples_per_bit]
        zero_crossings = np.where(np.diff(np.sign(segment)))[0]
        frequency = len(zero_crossings) / (2 * DURATION)

        if abs(frequency - FREQ_1) < abs(frequency - FREQ_0):
            binary_data += "1"
        else:
            binary_data += "0"

    return binary_data

# Convert binary back to text
def binary_to_text(binary_data):
    chars = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    return ''.join(chr(int(char, 2)) for char in chars if len(char) == 8)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/decode', methods=['POST'])
def decode_sound():
    waveform = request.json.get("waveform", [])
    if not waveform:
        return jsonify({"error": "No waveform data received"}), 400

    binary_data = fsk_wave_to_binary(np.array(waveform))
    decoded_text = binary_to_text(binary_data)

    return jsonify({"decoded_text": decoded_text})

if __name__ == "__main__":
    app.run(debug=True)