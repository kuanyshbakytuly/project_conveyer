import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
import multiprocessing as mp
import asyncio
import threading
import uvicorn
import time

app = FastAPI()
NUM_STREAMS = 20
TARGET_FPS = 25
RESOLUTION = (320, 180)

class StreamProcessor:
    def __init__(self):
        self.frame_buffers = [mp.Queue(maxsize=2) for _ in range(NUM_STREAMS)]  # Tiny buffers
        self.processes = []
        
    def start(self):
        for stream_id in range(NUM_STREAMS):
            p = mp.Process(
                target=self._process_stream,
                args=(f'output_5min.mp4', stream_id, self.frame_buffers[stream_id]),
                daemon=True
            )
            p.start()
            self.processes.append(p)
    
    def _process_stream(self, stream, stream_id, output_queue):
        cap = cv2.VideoCapture(stream)  # Use CPU decoding
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        
        last_frame_time = time.time()
        frame_interval = 1/TARGET_FPS
        
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, RESOLUTION)
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
            
            # Non-blocking frame dispatch
            if not output_queue.full():
                output_queue.put(jpeg.tobytes(), block=False)
            
            # Precise FPS control
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)
            
            if int(time.time()) > int(last_frame_time):
                print(f'{stream_id}: 1 sec')
                last_frame_time = time.time()

processor = StreamProcessor()

@app.on_event("startup")
async def startup():
    processor.start()

@app.websocket("/ws/{stream_id}")
async def video_feed(websocket: WebSocket, stream_id: int):
    await websocket.accept()
    buffer = processor.frame_buffers[stream_id]
    
    try:
        while True:
            start_time = time.time()
            if not buffer.empty():
                frame = buffer.get_nowait()
                await websocket.send_bytes(frame)
            
            # Maintain FPS on sender side
            elapsed = time.time() - start_time
            await asyncio.sleep(max(0, 1/TARGET_FPS - elapsed))
    except Exception as e:
        print(f"Stream {stream_id} error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)