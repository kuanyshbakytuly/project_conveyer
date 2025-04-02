from kafka import KafkaProducer
import json

# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092', 
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    api_version=(2, 6, 0)
)

# Example sending data to Kafka
data = {"camera_id": 1, "potato_id": 2, "size": "30-40mm"}
producer.send('potato_data', value=data)

# Flush and close the producer
producer.flush()
producer.close()

