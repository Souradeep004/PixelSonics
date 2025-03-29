const canvas = document.getElementById("waveCanvas");
const ctx = canvas.getContext("2d");

canvas.width = 250;
canvas.height = 250;

let waveOffset = 0;

function animateWave() {
    requestAnimationFrame(animateWave);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();

    let centerY = canvas.height / 2;
    let amplitude = 20;
    ctx.strokeStyle = "rgb(0, 122, 204)";
    ctx.lineWidth = 3;

    for (let x = 0; x < canvas.width; x++) {
        let y = centerY + Math.sin((x + waveOffset) * 0.12) * amplitude;
        ctx.lineTo(x, y);
    }

    ctx.stroke();
    waveOffset += 2;
}

animateWave();

function encodeMessage() {
    let text = document.getElementById("encodeInput").value;
    if (text.trim() === "") {
        alert("Please enter text to encode!");
        return;
    }

    let wave = ggwave.encode({ payload: text, protocolId: 0, volume: 1.0 });
    let audio = new Audio(wave);
    audio.play();
}

function startDecoding() {
    ggwave.init({ onDecoded: (message) => {
        document.getElementById("decodeOutput").value = message;
    }});
}