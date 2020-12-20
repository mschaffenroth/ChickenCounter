# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM debian:stable

RUN apt update && apt install --no-install-recommends -y python3-pip libgl1-mesa-glx python3-setuptools build-essential autoconf libtool pkg-config python3-dev && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install gsutil

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# SOURCE: 0 1 X for webcam, or file/folder 
ENV SOURCE 0 
# MAX_FPS: do you want to restrict the fps?
ENV MAX_FPS 100 
ENV BROKER_TOPIC v1/devices/chicken_counter/telemetry
ENV BROKER_URL localhost
ENV BROKER_PORT 1883
ENV USERNAME chicken_counter
ENV PASSWORD none

# Copy weights
RUN python3 -c "from models.experimental import attempt_load; \
attempt_load('weights/yolov5s.pt'); \
attempt_load('weights/yolov5m.pt'); \
attempt_load('weights/yolov5l.pt')"

ENTRYPOINT python3 server3.py --source $SOURCE --max-fps $MAX_FPS --broker-url $BROKER_URL --broker-topic $BROKER_TOPIC --broker-port $BROKER_PORT --broker-username $USERNAME --broker-password $PASSWORD --max-fps $MAX_FPS

# ---------------------------------------------------  Extras Below  ---------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest && sudo docker build -t $t . && sudo docker push $t
# for v in {300..303}; do t=ultralytics/coco:v$v && sudo docker build -t $t . && sudo docker push $t; done

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/coco:/usr/src/coco $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -a -q --filter ancestor=ultralytics/yolov5:latest)

# Bash into running container
# sudo docker container exec -it ba65811811ab bash

# Bash into stopped container
# sudo docker commit 092b16b25c5b usr/resume && sudo docker run -it --gpus all --ipc=host -v "$(pwd)"/coco:/usr/src/coco --entrypoint=sh usr/resume

# Send weights to GCP
# python -c "from utils.general import *; strip_optimizer('runs/train/exp0_*/weights/best.pt', 'tmp.pt')" && gsutil cp tmp.pt gs://*.pt

# Clean up
# docker system prune -a --volumes
