let activeStream = null;

export async function startCamera(videoElement, width, height) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: width }, height: { ideal: height }, facingMode: 'user' }
        });
        videoElement.srcObject = stream;
        activeStream = stream;
        return true;
    } catch (err) {
        console.error('Camera access denied:', err);
        return false;
    }
}

export function stopCamera(videoElement) {
    if (activeStream) {
        activeStream.getTracks().forEach(t => t.stop());
        activeStream = null;
    }
    if (videoElement) {
        videoElement.srcObject = null;
    }
}

export function captureFrame(videoElement, canvasElement, width, height) {
    if (!videoElement || videoElement.readyState < 2) return null;
    canvasElement.width = width;
    canvasElement.height = height;
    const ctx = canvasElement.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, width, height);
    return canvasElement.toDataURL('image/jpeg', 0.9);
}
