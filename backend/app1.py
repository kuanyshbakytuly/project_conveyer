import ffmpegcv
import multiprocessing as mp
import torch
from pynvml import *
import cv2
import psutil
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from kafka import KafkaProducer, KafkaConsumer
import json
import asyncio
import threading
import base64
import uuid
import time
from kafka.errors import NoBrokersAvailable

app = FastAPI()

# Kafka configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
INPUT_TOPIC = "video_input"
OUTPUT_TOPIC = "video_output"
KAFKA_MAX_RETRIES = 5
KAFKA_RETRY_DELAY = 3

class ConnectionManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()

def get_kafka_producer():
    retries = 0
    while retries < KAFKA_MAX_RETRIES:
        try:
            return KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                api_version=(2, 0, 2),
                acks='all',
                retries=3
            )
        except NoBrokersAvailable:
            retries += 1
            print(f"Waiting for Kafka broker (attempt {retries}/{KAFKA_MAX_RETRIES})...")
            time.sleep(KAFKA_RETRY_DELAY)
    raise Exception("Failed to connect to Kafka broker after multiple attempts")

def get_kafka_consumer():
    retries = 0
    while retries < KAFKA_MAX_RETRIES:
        try:
            return KafkaConsumer(
                OUTPUT_TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                auto_offset_reset='earliest',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                api_version=(2, 0, 2),
                consumer_timeout_ms=1000,
                group_id='video-consumer-group'
            )
        except NoBrokersAvailable:
            retries += 1
            print(f"Waiting for Kafka broker (attempt {retries}/{KAFKA_MAX_RETRIES})...")
            time.sleep(KAFKA_RETRY_DELAY)
    raise Exception("Failed to connect to Kafka broker after multiple attempts")

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Video Stream</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            #video { background: #000; border: 1px solid #ccc; }
            .controls { margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Real-time Video Stream</h1>
        <div class="controls">
            <input type="text" id="streamUrl" placeholder="Enter video URL" value="output_5min.mp4">
            <button onclick="startStream()">Start Stream</button>
            <button onclick="stopStream()">Stop Stream</button>
        </div>
        <img id="video" width="640" height="480">
        <script>
            const video = document.getElementById('video');
            const streamUrl = document.getElementById('streamUrl');
            let ws = null;
            let streamId = null;
            
            function startStream() {
                if (ws) return;
                
                ws = new WebSocket('ws://' + window.location.host + '/ws/video');
                
                ws.onopen = function() {
                    const url = streamUrl.value || 'output_5min.mp4';
                    ws.send(JSON.stringify({action: 'start', url: url}));
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.stream_id) {
                        streamId = data.stream_id;
                    }
                    if (data.frame) {
                        video.src = 'data:image/jpeg;base64,' + data.frame;
                    }
                };
                
                ws.onclose = function() {
                    ws = null;
                    streamId = null;
                };
            }
            
            function stopStream() {
                if (ws && streamId) {
                    ws.send(JSON.stringify({action: 'stop', stream_id: streamId}));
                }
            }
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    stream_id = None
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['action'] == 'start':
                stream_url = message['url']
                stream_id = str(uuid.uuid4())
                await websocket.send_json({"stream_id": stream_id})
                
                # Start processing in background
                process = mp.Process(
                    target=process_stream,
                    args=(stream_url, stream_id, OUTPUT_TOPIC)
                )
                process.start()
                
            elif message['action'] == 'stop' and message['stream_id']:
                # Implement your stop logic here
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

def process_stream(stream_url, stream_id, output_topic):
    try:
        # Initialize CUDA if available
        if torch.cuda.is_available():
            torch.cuda.init()
            nvmlInit()

        producer = get_kafka_producer()
        
        # Use ffmpegcv for better hardware acceleration
        cap = ffmpegcv.VideoCapture(stream_url, pix_fmt='bgr24', resize=(640, 480))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Stream {stream_id} ended")
                break

            # Process frame
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send to Kafka
            try:
                producer.send(output_topic, value={
                    'stream_id': stream_id,
                    'frame': frame_base64,
                    'timestamp': time.time()
                })
            except Exception as e:
                print(f"Kafka send error: {str(e)}")
                producer = get_kafka_producer()
                continue
                
            # Control frame rate
            time.sleep(1/30)  # ~30 FPS

    except Exception as e:
        print(f"Process {stream_id} error: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        if 'producer' in locals():
            producer.close()
        if torch.cuda.is_available():
            nvmlShutdown()

async def consume_and_broadcast():
    consumer = get_kafka_consumer()
    try:
        for message in consumer:
            data = message.value
            for connection in manager.active_connections.copy():
                try:
                    await connection.send_json(data)
                except Exception as e:
                    print(f"Error sending to WebSocket: {str(e)}")
                    manager.disconnect(connection)
    except Exception as e:
        print(f"Kafka consumer error: {str(e)}")
    finally:
        consumer.close()

@app.on_event("startup")
async def startup_event():
    # Start Kafka consumer in background
    asyncio.create_task(consume_and_broadcast())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)