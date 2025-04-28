from kafka import KafkaConsumer
import json
from src.utils.logger import logger

class KafkaMessageConsumer:
    def __init__(self, topic, bootstrap_servers='kafka:9092'):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

    def consume_messages(self, process_message):
        try:
            for message in self.consumer:
                logger.info(f"Received message: {message.value}")
                process_message(message.value)
        except Exception as e:
            logger.error(f"Error consuming messages: {e}")