# PixelSonics
 PixelSonic is an innovative module that transforms images and text into sound and then decodes that sound back into meaningful output. The fusion of sound and vision, Pixel Sonic opens up a world of possibilities! This project explores an esoteric way to transfer complex data like image through sound. This is a prototype of our endeavour.
# Project Details
Audio Frequency Modulation can enable image transmission through sound by converting image data into audio signals and then decoding them back into an image. First, the image is compressed and converted into a binary stream. This binary data is then modulated using Frequency Shift Keying (FSK) or Phase Shift Keying (PSK), encoding the bits into distinct sound frequencies. A speaker emits these modulated signals, which can be transmitted through air or a medium. On the receiving end, a microphone captures the sound, and a Fast Fourier Transform (FFT) or Goertzel algorithm demodulates the audio back into a binary stream. The binary data is then reconstructed into an image, ensuring minimal loss with error correction techniques like Reed-Solomon coding. This method allows offline, short-range, contactless image transmission without the need for internet connectivity, making it useful for secure data sharing and environments with limited wireless infrastructure.
## Key Technologies Used
<p>✅ FSK/PSK for Modulation</p>
<p>✅ FFT/Goertzel for Demodulation</p>
<p>✅ Error Correction for Data Integrity</p>
<p>✅ Works in Offline & No-Wireless Environments</p>

## Libraries used
1. Numpy
2. OpenCV
3. Soundfile
4. Scipy
5. Matplotlib
