from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
import librosa
import soundfile as sf
import torch
import os
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Image to Sound Encoding
def encode_image_to_sound(image_path, output_filename, sample_rate=44100, min_freq=15000, max_freq=20000, duration_per_pixel=1/16):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))  # Resize for uniformity
    pixel_values = image.flatten()
    frequencies = np.interp(pixel_values, [0, 255], [min_freq, max_freq])
    
    audio_wave = []
    for freq in frequencies:
        t = np.linspace(0, duration_per_pixel, int(sample_rate * duration_per_pixel), False)
        wave = 0.5 * np.sin(2 * np.pi * freq * t)
        audio_wave.extend(wave)
    
    audio_wave = np.array(audio_wave) * 32767
    audio_wave = audio_wave.astype(np.int16)
    
    sf.write(output_filename, audio_wave, sample_rate)
    return output_filename

@app.route('/encode', methods=['POST'])
def encode():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    sound_path = os.path.join(OUTPUT_FOLDER, "encoded_sound.wav")

    image_file.save(image_path)
    encode_image_to_sound(image_path, sound_path)
    
    return send_file(sound_path, as_attachment=True)

# Sound to Image Decoding
def decode_sound_to_image(input_filename, sample_rate=44100, min_freq=15000, max_freq=20000, duration_per_pixel=1/16):
    audio_wave, _ = sf.read(input_filename)
    num_pixels = int(len(audio_wave) / (sample_rate * duration_per_pixel))
    pixel_values = []
    
    for i in range(num_pixels):
        start_idx = int(i * sample_rate * duration_per_pixel)
        end_idx = int((i + 1) * sample_rate * duration_per_pixel)
        if end_idx > len(audio_wave):
            break
        
        segment = audio_wave[start_idx:end_idx]
        spectrum = np.fft.fft(segment)
        freqs = np.fft.fftfreq(len(segment), d=1/sample_rate)
        peak_idx = np.argmax(np.abs(spectrum[:len(spectrum)//2]))
        dominant_freq = freqs[peak_idx]
        
        pixel_value = np.interp(dominant_freq, [min_freq, max_freq], [0, 255])
        pixel_values.append(pixel_value)
    
    size = int(np.sqrt(len(pixel_values)))
    image_reconstructed = np.array(pixel_values[:size*size], dtype=np.uint8).reshape((size, size))
    
    output_path = os.path.join(OUTPUT_FOLDER, "reconstructed_image.jpg")
    cv2.imwrite(output_path, image_reconstructed)
    
    return output_path

@app.route('/decode', methods=['POST'])
def decode():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(audio_path)
    
    image_path = decode_sound_to_image(audio_path)
    
    return send_file(image_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)