import cv2
import asyncio
import concurrent.futures
import yaml
from src.utils.logger import logger
from src.services.kafka_producer import KafkaMessageProducer

RTSP_STREAMS = []
frame_streams = {}

try:
    with open('configs/streams.yaml', 'r') as file:
        config = yaml.safe_load(file)
        RTSP_STREAMS = config['streams']
        frame_streams = {index: None for index in range(len(RTSP_STREAMS))}
        logger.info(f"Loaded streams: {RTSP_STREAMS}")
        logger.info(f"Initialized frame streams: {frame_streams}")
except Exception as e:
    logger.error(f"Failed to load streams from YAML: {e}")

def read_stream(index, rtsp_url):
    producer = KafkaMessageProducer()
    logger.info(f"Starting stream {index} from {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # Увеличить размер буфера

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error(f"Failed to read frame from stream {index}. Stopping stream.")
            break
        
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            producer.send_message('frames', {'index': index, 'frame_data': frame_data, 'frame_count': frame_count, 'rtsp_url': rtsp_url})
            frame_streams[index] = frame_data
        except Exception as e:
            logger.error(f"Error sending frame to Kafka: {e}")
            frame_streams[index] = None

        frame_count += 1

async def start_streams():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(read_stream, index, url) for index, url in enumerate(RTSP_STREAMS)]
        await asyncio.gather(*[asyncio.wrap_future(f) for f in futures])
    logger.info("All RTSP streams started")