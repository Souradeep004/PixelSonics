const canvas = document.getElementById("waveCanvas");
const ctx = canvas.getContext("2d");
const assistantText = document.getElementById("assistantText");

canvas.width = 250;
canvas.height = 250;

let audioContext, analyser, microphone;
let waveOffset = 0;
let frequencyData = new Uint8Array(256);
let animationId;

function animateWave() {
    cancelAnimationFrame(animationId);
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw center circle
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const baseRadius = 30;
    
    ctx.beginPath();
    ctx.arc(centerX, centerY, baseRadius, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1a1a';
    ctx.fill();
    
    // Draw animated wave
    ctx.strokeStyle = '#4F959D';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const points = 100;
    const amplitude = 20;
    
    for (let i = 0; i <= points; i++) {
        const angle = (i / points) * Math.PI * 2;
        const radius = baseRadius + Math.sin(angle * 5 + waveOffset) * amplitude;
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.closePath();
    ctx.stroke();
    
    waveOffset += 0.1;
    animationId = requestAnimationFrame(animateWave);
}

function visualizeFrequency() {
    cancelAnimationFrame(animationId);
    
    if (!analyser) return;
    
    analyser.getByteFrequencyData(frequencyData);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const baseRadius = 30;
    
    // Draw center circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, baseRadius, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1a1a';
    ctx.fill();
    
    // Draw frequency bars
    const barCount = 32;
    const maxBarLength = 80;
    
    for (let i = 0; i < barCount; i++) {
        const freqIndex = Math.floor(i * (frequencyData.length / barCount));
        const energy = frequencyData[freqIndex] / 255;
        const barLength = energy * maxBarLength;
        const angle = (i / barCount) * Math.PI * 2;
        
        const x1 = centerX + Math.cos(angle) * baseRadius;
        const y1 = centerY + Math.sin(angle) * baseRadius;
        const x2 = centerX + Math.cos(angle) * (baseRadius + barLength);
        const y2 = centerY + Math.sin(angle) * (baseRadius + barLength);
        
        ctx.strokeStyle = `hsl(${i * 360 / barCount}, 100%, 50%)`;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    }
    
    animationId = requestAnimationFrame(visualizeFrequency);
}

async function encodeMessage() {
    let text = document.getElementById("encodeInput").value;
    if (text.trim() === "") {
        alert("Please enter text to encode!");
        return;
    }

    assistantText.innerText = "Encoding and transmitting sound...";
    
    try {
        const response = await fetch('/encode', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text })
        });

        const data = await response.json();
        if (data.error) {
            alert(data.error);
            return;
        }

        playWaveform(data.waveform);
    } catch (error) {
        console.error("Encoding error:", error);
        alert("Failed to encode text");
    }
}

function playWaveform(waveformBase64) {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    const binaryString = atob(waveformBase64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    const waveform = new Float32Array(bytes.buffer);

    const audioBuffer = audioContext.createBuffer(1, waveform.length, 44100);
    audioBuffer.getChannelData(0).set(waveform);
    
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    analyser.connect(audioContext.destination);
    
    visualizeFrequency();
    source.start();
    
    source.onended = () => {
        assistantText.innerText = "Transmission complete!";
        setTimeout(() => animateWave(), 1000);
    };
}

async function startDecoding() {
    assistantText.innerText = "Listening for sound waves...";
    
    try {
        const response = await fetch('/start_recording', {
            method: 'POST'
        });

        const data = await response.json();
        if (data.status === 'recording_complete') {
            assistantText.innerText = "Recording complete! Decoding...";
            decodeAudio();
        }
    } catch (error) {
        console.error("Recording error:", error);
        assistantText.innerText = "Recording failed";
    }
}

async function decodeAudio() {
    try {
        const response = await fetch('/decode');
        const data = await response.json();

        if (data.error) {
            assistantText.innerText = data.error;
            return;
        }

        document.getElementById("decodeOutput").value = data.text;
        assistantText.innerText = "Decoding complete!";
    } catch (error) {
        console.error("Decoding error:", error);
        assistantText.innerText = "Decoding failed";
    } finally {
        setTimeout(() => animateWave(), 500);
    }
}

// Initialize animation
animateWave();

// Event listeners
document.querySelector("button[onclick='encodeMessage()']").addEventListener('click', encodeMessage);
document.querySelector("button[onclick='startDecoding()']").addEventListener('click', startDecoding);



