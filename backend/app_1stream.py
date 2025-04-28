import cv2
from fastapi.websockets import WebSocketState
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import multiprocessing as mp
import asyncio
import threading
import uvicorn
import time
from backend.cv_all import process_stream  # Your video processing function

app = FastAPI()
NUM_STREAMS = 1  # Adjust if you need multiple streams
TARGET_FPS = 25  # Throttle frame rate
RESOLUTION = (640, 480)  # Downscale frames to 640x480

mp_queues = []
processes = []
async_queues = []

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_running_loop()
    for stream_id in range(NUM_STREAMS):
        # Use smaller queues to prevent overload
        mp_queue = mp.Queue(maxsize=5)
        mp_queues.append(mp_queue)
        
        # Start video processing process
        process = mp.Process(
            target=process_stream,
            args=(f'output_5min.mp4', stream_id, mp_queue),
            daemon=True
        )
        process.start()
        processes.append(process)
        
        async_queue = asyncio.Queue(maxsize=5)
        async_queues.append(async_queue)
        
        # Bridge between multiprocessing and asyncio
        threading.Thread(
            target=queue_bridge,
            args=(mp_queue, async_queue, loop),
            daemon=True
        ).start()

def queue_bridge(mp_queue, async_queue, loop):
    while True:
        try:
            frame_data = mp_queue.get()
            if not async_queue.full():
                asyncio.run_coroutine_threadsafe(
                    async_queue.put(frame_data), 
                    loop
                )
            else:
                print("Async queue full - dropping frame")
        except Exception as e:
            print(f"Queue bridge error: {e}")

@app.websocket("/ws/{stream_id}")
async def video_stream(websocket: WebSocket, stream_id: int):
    await websocket.accept()
    last_frame_time = time.time()
    try:
        async_queue = async_queues[stream_id]
        while True:
            try:
                # Throttle frame rate
                await asyncio.sleep(max(0, 1/TARGET_FPS - (time.time() - last_frame_time)))
                frame_data = await async_queue.get()
                
                await websocket.send_bytes(frame_data)
                last_frame_time = time.time()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)  # Small sleep if queue is empty
                continue
    except Exception as e:
        print(f"Stream {stream_id} error: {str(e)}")
        if not websocket.client_state == WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except RuntimeError:
                pass  # Already closed
    finally:
        if not websocket.client_state == WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except RuntimeError:
                pass  # Already closed
@app.get("/")
async def get_html():
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_ping_interval=5, ws_ping_timeout=5)


html_content="""
<!DOCTYPE html>
<html>
<head>
    <title>Video Stream</title>
    <style>
        .video-container {
            position: relative;
            width: 100%;
            max-width: 800px;
            margin: 20px auto;
            background: #000;
        }
        .video-img {
            width: 100%;
            height: auto;
            object-fit: contain;
        }
    </style>
</head>
<body>
    <div class="video-container">
        <img class="video-img" id="stream-0" decoding="async">
    </div>
    <script>
        const img = document.getElementById('stream-0');
        const stream_id = 0;  // Change if multiple streams
        const ws = new WebSocket("ws://localhost:8000/ws/0");
        ws.binaryType = 'arraybuffer';
        
        // Memory management for blob URLs
        let currentBlobUrl = null;
        
        ws.onmessage = function(event) {
            if (currentBlobUrl) {
                URL.revokeObjectURL(currentBlobUrl);
            }
            const blob = new Blob([event.data], {type: 'image/jpeg'});
            currentBlobUrl = URL.createObjectURL(blob);
            img.src = currentBlobUrl;
        };
        
        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
        };
        
        ws.onclose = () => {
            console.log("WebSocket disconnected");
        };
    </script>
</body>
</html>"""