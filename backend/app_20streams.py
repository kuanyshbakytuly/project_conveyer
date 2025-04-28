import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import multiprocessing as mp
import asyncio
import threading
import uvicorn
import time
from backend.cv import process_stream

app = FastAPI()
NUM_STREAMS = 20
TARGET_FPS = 25
RESOLUTION = (320, 180)  # Reduced resolution for better performance

mp_queues = []
processes = []
async_queues = []

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_running_loop()
    for stream_id in range(NUM_STREAMS):
        # Use smaller queues to prevent memory buildup
        mp_queue = mp.Queue(maxsize=30)  # Reduced from 200
        mp_queues.append(mp_queue)
        
        process = mp.Process(
            target=process_stream,
            args=(f'output_5min.mp4', stream_id, mp_queue),
            daemon=True
        )
        process.start()
        processes.append(process)
        
        async_queue = asyncio.Queue(maxsize=30)  # Reduced from 200
        async_queues.append(async_queue)
        
        threading.Thread(
            target=queue_bridge,
            args=(mp_queue, async_queue, loop),
            daemon=True
        ).start()

def queue_bridge(mp_queue, async_queue, loop):
    while True:
        try:
            frame_data = mp_queue.get()
            # Skip frames if queue is full to maintain real-time performance
            if not async_queue.full():
                asyncio.run_coroutine_threadsafe(
                    async_queue.put(frame_data), 
                    loop
                )
            else:
                pass  # Drop frame to prevent backlog
        except Exception as e:
            print(f"Queue bridge error: {e}")

@app.websocket("/ws/{stream_id}")
async def video_stream(websocket: WebSocket, stream_id: int):
    await websocket.accept()
    last_frame_time = time.time()
    try:
        async_queue = async_queues[stream_id]
        while True:
            # Throttle frame rate
            await asyncio.sleep(max(0, 1/TARGET_FPS - (time.time() - last_frame_time)))
            frame_data = await async_queue.get()
            await websocket.send_bytes(frame_data)
            last_frame_time = time.time()
    except Exception as e:
        print(f"Stream {stream_id} error: {str(e)}")
    finally:
        try:
            await websocket.close()
        except:
            pass

@app.get("/")
async def get_html():
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_ping_interval=5, ws_ping_timeout=5)
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>20 Stream Monitor</title>
    <style>
        .video-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            padding: 10px;
        }
        .video-container {
            position: relative;
            padding-bottom: 56.25%; /* 16:9 aspect ratio */
            background: #000;
        }
        .video-img {
            position: absolute;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
    </style>
</head>
<body>
    <div class="video-grid" id="videoGrid"></div>
    <script>
        // Create video elements
        const grid = document.getElementById('videoGrid');
        for (let i = 0; i < 20; i++) {
            const container = document.createElement('div');
            container.className = 'video-container';
            
            const img = document.createElement('img');
            img.className = 'video-img';
            img.id = `stream-${i}`;
            
            container.appendChild(img);
            grid.appendChild(container);
            
            // Connect WebSocket
            const ws = new WebSocket("ws://localhost:8000/ws/0");
            ws.binaryType = 'arraybuffer';
            
            ws.onmessage = function(event) {
                const blob = new Blob([event.data], {type: 'image/jpeg'});
                img.src = URL.createObjectURL(blob);
            };
        }
    </script>
</body>
</html>
"""