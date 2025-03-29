# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import numpy as np
# import wave
# import io
# import cv2
# import base64

# app = Flask(__name__)
# CORS(app)

# # Text Encoding Parameters
# SAMPLE_RATE = 44100  
# DURATION = 0.02  
# FREQ_0 = 500  
# FREQ_1 = 1000  

# # Image Encoding Parameters
# IMAGE_RESOLUTION = (64, 64)  
# MIN_FREQ = 500
# MAX_FREQ = 1000
# DURATION_PER_PIXEL = 1/20  

# def text_to_binary(text):
#     """ Convert text into binary representation """
#     try:
#         binary = ''.join(format(ord(char), '08b') for char in text)
#         print(f"✅ Converted text to binary: {binary[:64]}...")  # Log first 64 bits
#         return binary
#     except Exception as e:
#         print(f"❌ Error converting text to binary: {str(e)}")
#         raise ValueError("Text encoding failed.")

# def generate_fsk_wave(binary_data):
#     """ Generate FSK modulated wave from binary data """
#     try:
#         t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
#         waveform = np.concatenate([
#             0.5 * np.sin(2 * np.pi * (FREQ_1 if bit == '1' else FREQ_0) * t)
#             for bit in binary_data
#         ])
#         print("✅ FSK waveform generated successfully")
#         return waveform.astype(np.float32)
#     except Exception as e:
#         print(f"❌ Error generating FSK wave: {str(e)}")
#         raise ValueError("Waveform generation failed.")

# def encode_image_to_sound(image_bytes):
#     """ Convert an image into a sound wave """
#     try:
#         print("🔹 Decoding image...")
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

#         if image is None:
#             raise ValueError("❌ Could not decode image. Invalid format.")

#         print(f"✅ Image shape before resizing: {image.shape}")

#         image = cv2.resize(image, IMAGE_RESOLUTION)
#         print(f"✅ Image resized to: {IMAGE_RESOLUTION}")

#         pixel_values = image.flatten()
#         frequencies = np.interp(pixel_values, [0, 255], [MIN_FREQ, MAX_FREQ])
        
#         audio_wave = []
#         for freq in frequencies:
#             t = np.linspace(0, DURATION_PER_PIXEL, int(SAMPLE_RATE * DURATION_PER_PIXEL), False)
#             wave = 0.5 * np.sin(2 * np.pi * freq * t)
#             audio_wave.extend(wave)
        
#         audio_wave = np.array(audio_wave) * 32767
#         print("✅ Image successfully converted to sound wave")
#         return audio_wave.astype(np.int16)

#     except Exception as e:
#         print(f"❌ Error encoding image to sound: {str(e)}")
#         raise ValueError("Image encoding failed.")

# @app.route('/generate_sound', methods=['POST'])
# def generate_sound():
#     try:
#         print("🔹 Received request for sound generation")
#         data = request.json

#         if not data:
#             print("❌ No data received")
#             return jsonify({"error": "No data received"}), 400

#         if 'text' in data and data['text'].strip():
#             print("🔹 Processing text...")
#             binary_data = text_to_binary(data['text'])
#             waveform = generate_fsk_wave(binary_data)
#         elif 'image' in data and data['image']:
#             print("🔹 Processing image...")
#             image_data = base64.b64decode(data['image'].split(',')[1])
#             waveform = encode_image_to_sound(image_data)
#         else:
#             print("❌ Invalid input")
#             return jsonify({"error": "Invalid input"}), 400

#         buffer = io.BytesIO()
#         with wave.open(buffer, 'wb') as wf:
#             wf.setnchannels(1)
#             wf.setsampwidth(2)
#             wf.setframerate(SAMPLE_RATE)
#             wf.writeframes((waveform * 32767).astype(np.int16).tobytes())

#         buffer.seek(0)
#         print("✅ Sound file generated successfully")
#         return send_file(buffer, mimetype="audio/wav")

#     except Exception as e:
#         print(f"❌ Backend Error: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import wave
import io
import cv2
import base64

app = Flask(__name__)
CORS(app)

# Encoding parameters
SAMPLE_RATE = 44100  
DURATION = 0.02  
FREQ_0 = 500  
FREQ_1 = 1000  

# Image Encoding
IMAGE_RESOLUTION = (64, 64)  
MIN_FREQ = 500
MAX_FREQ = 1000
DURATION_PER_PIXEL = 1/20  

def text_to_binary(text):
    """ Convert text into binary representation """
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary

def generate_fsk_wave(binary_data):
    """ Generate FSK modulated wave from binary data """
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    waveform = np.concatenate([
        0.5 * np.sin(2 * np.pi * (FREQ_1 if bit == '1' else FREQ_0) * t)
        for bit in binary_data
    ])
    return waveform.astype(np.float32)

def encode_image_to_sound(image_bytes):
    """ Convert an image into a sound wave """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Invalid image format")

    image = cv2.resize(image, IMAGE_RESOLUTION)
    pixel_values = image.flatten()
    frequencies = np.interp(pixel_values, [0, 255], [MIN_FREQ, MAX_FREQ])
    
    audio_wave = []
    for freq in frequencies:
        t = np.linspace(0, DURATION_PER_PIXEL, int(SAMPLE_RATE * DURATION_PER_PIXEL), False)
        wave = 0.5 * np.sin(2 * np.pi * freq * t)
        audio_wave.extend(wave)
    
    return np.array(audio_wave).astype(np.float32)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route('/generate_sound', methods=['POST'])
def generate_sound():
    """ Handle encoding text or image into sound """
    data = request.json

    if not data:
        return jsonify({"error": "No data received"}), 400

    if 'text' in data and data['text'].strip():
        binary_data = text_to_binary(data['text'])
        waveform = generate_fsk_wave(binary_data)
    elif 'image' in data and data['image']:
        image_data = base64.b64decode(data['image'].split(',')[1])
        waveform = encode_image_to_sound(image_data)
    else:
        return jsonify({"error": "Invalid input"}), 400

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((waveform * 32767).astype(np.int16).tobytes())

    buffer.seek(0)
    return send_file(buffer, mimetype="audio/wav", as_attachment=True, download_name="encoded_sound.wav")

def decode_fsk_wave(audio_bytes):
    """ Decode FSK modulated audio back to binary and text """
    with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
        raw_audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    binary_data = []
    segment_size = int(SAMPLE_RATE * DURATION)
    
    for i in range(0, len(raw_audio), segment_size):
        segment = raw_audio[i:i + segment_size]
        fft_freqs = np.fft.fftfreq(len(segment), 1/SAMPLE_RATE)
        dominant_freq = abs(fft_freqs[np.argmax(np.abs(np.fft.fft(segment)))])
        
        if abs(dominant_freq - FREQ_1) < 100:
            binary_data.append('1')
        elif abs(dominant_freq - FREQ_0) < 100:
            binary_data.append('0')

    binary_string = ''.join(binary_data)
    decoded_text = ''.join(chr(int(binary_string[i:i+8], 2)) for i in range(0, len(binary_string), 8))
    return decoded_text.strip()

def decode_image_from_sound(audio_bytes):
    """ Convert sound wave back into an image """
    with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
        raw_audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    segment_size = int(SAMPLE_RATE * DURATION_PER_PIXEL)
    num_pixels = len(raw_audio) // segment_size
    frequencies = []

    for i in range(num_pixels):
        segment = raw_audio[i * segment_size:(i + 1) * segment_size]
        if len(segment) == 0:
            continue
        fft_freqs = np.fft.fftfreq(len(segment), 1/SAMPLE_RATE)
        dominant_freq = abs(fft_freqs[np.argmax(np.abs(np.fft.fft(segment)))])
        frequencies.append(dominant_freq)

    pixel_values = np.interp(frequencies, [MIN_FREQ, MAX_FREQ], [0, 255]).astype(np.uint8)

    if len(pixel_values) != IMAGE_RESOLUTION[0] * IMAGE_RESOLUTION[1]:
        return None  # Incomplete data

    image = pixel_values.reshape(IMAGE_RESOLUTION)
    
    _, img_encoded = cv2.imencode('.png', image)
    return img_encoded.tobytes()

@app.route('/decode_sound', methods=['POST'])
def decode_sound():
    """ Handle decoding of sound into text or image """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio'].read()

    # Try decoding the image first
    decoded_image = decode_image_from_sound(audio_file)
    if decoded_image:
        image_path = "static/decoded_image.png"
        with open(image_path, "wb") as f:
            f.write(decoded_image)

        # Send Image as a downloadable file
        return send_file(image_path, mimetype="image/png")

    # If no image found, try text decoding
    decoded_text = decode_fsk_wave(audio_file)
    if decoded_text.strip():
        return jsonify({"text": decoded_text})

    return jsonify({"error": "Decoding failed"}), 400

if __name__ == '__main__':
    app.run(debug=True)
