sudo docker run --gpus=all -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:25.03-py3

pip install opencv-python
apt-get update
apt-get install -y libgl1
apt-get install -y libglvnd-dev
ldconfig -p | grep libGL

tritonserver --model-repository=/models