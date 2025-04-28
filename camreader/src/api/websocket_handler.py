import cv2
import json
import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.utils.logger import logger
from src.api.routers import camera_router
from src.services.stream_reader import frame_streams
from src.utils.batch_annotation import batch_annotation
from src.services.kafka_consumer import KafkaMessageConsumer


clients = {i: [] for i in frame_streams.keys()}
consumer = KafkaMessageConsumer(topic='predictions', bootstrap_servers='kafka:9092')

async def get_predictions_from_kafka(index):
    async for message in consumer.consumer:
        if message.value['index'] == index:
            return message.value['predictions']

@camera_router.websocket("/camera_stream/{stream_index}")
async def stream_handler(websocket: WebSocket, stream_index: int):
    await websocket.accept()
    params = ["Masks", "Visors", "Phones", "Jewels", "Smoke_face_eat"]
    logger.info(f"Client connected to stream {stream_index}")
    
    if stream_index not in frame_streams:
        await websocket.send_text("Invalid stream index")
        logger.warning(f"Invalid stream index {stream_index} requested")
        return

    clients[stream_index].append(websocket)
    
    async def send_frames():
        try:
            while True:
                if frame_streams[stream_index]:
                    frame_data = frame_streams[stream_index]
                    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    _, buffer = cv2.imencode('.jpg', frame)
                    await websocket.send_bytes(buffer.tobytes())
                await asyncio.sleep(0.1)  # Отправляем кадры с интервалом
        except asyncio.CancelledError:
            logger.info("Frame sending task cancelled")
        except Exception as e:
            logger.error(f"Error sending frames: {e}")

    async def receive_messages():
        try:
            while True:
                message = await websocket.receive_text()
                params = json.loads(message)
                logger.info(f"Received parameters for batch annotation: {params}")
        except WebSocketDisconnect:
            clients[stream_index].remove(websocket)
            logger.info(f"Client disconnected from stream {stream_index}")
            raise
        except asyncio.CancelledError:
            logger.info("Message receiving task cancelled")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    send_task = asyncio.create_task(send_frames())
    receive_task = asyncio.create_task(receive_messages())
    
    try:
        await asyncio.gather(send_task, receive_task)
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        receive_task.cancel()
        await websocket.close()
        clients[stream_index].remove(websocket)
        logger.info(f"Client fully disconnected from stream {stream_index}")