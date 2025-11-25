// Connect to backend server
const socket = io();

// DOM Elements
const video = document.getElementById("video");
const predictionDisplay = document.getElementById("prediction");
const stablePredictionDisplay = document.getElementById("stablePrediction");
const sentenceDisplay = document.getElementById("sentence");
const cameraBtn = document.getElementById("cameraBtn");

let cameraActive = false;

// Start camera function
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
        cameraActive = true;
        sendFrames();
        console.log("Camera started!");
    } catch (error) {
        console.error("Camera not accessible:", error);
    }
}

// Convert frame to Base64 and emit to server
function sendFrames() {
    if (!cameraActive) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
        if (blob) {
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64Image = reader.result.split(",")[1];
                socket.emit("frame", { image: base64Image });
                console.log("Frame sent");
            };
            reader.readAsDataURL(blob);
        }
    }, "image/jpeg");

    setTimeout(sendFrames, 150); // ~6.6 FPS
}

// Handle predictions from server
socket.on("prediction", (data) => {
    console.log("Prediction received:", data);
    predictionDisplay.innerText = data.prediction || "None";
});

// Handle stable prediction
socket.on("stable_prediction", (data) => {
    console.log("Stable prediction received:", data);
    stablePredictionDisplay.innerText = data.stable_prediction || "None";
});

// Handle sentence update
socket.on("sentence_update", (data) => {
    console.log("Sentence received:", data);
    sentenceDisplay.innerText = data.sentence || "";
});

// Connect message
socket.on("connect", () => {
    console.log("Connected to server!");
});

// Disconnect message
socket.on("disconnect", () => {
    console.log("Disconnected from server!");
});

// Button Event Listener
cameraBtn.addEventListener("click", () => {
    if (!cameraActive) {
        startCamera();
    }
});
