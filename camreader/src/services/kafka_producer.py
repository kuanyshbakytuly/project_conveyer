from kafka import KafkaProducer
import json
from src.utils.logger import logger

class KafkaMessageProducer:
    def __init__(self, bootstrap_servers='kafka:9092'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def send_message(self, topic, message):
        try:
            self.producer.send(topic, message)
            self.producer.flush()
            logger.info(f"Sent message to topic {topic}: {message}")
        except Exception as e:
            logger.error(f"Failed to send message to topic {topic}: {e}")