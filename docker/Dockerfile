FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl gnupg2 lsb-release \
    python3 python3-pip \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    portaudio19-dev libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ROS2 Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y ros-humble-ros-base python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install onnxruntime opencv-python-headless numpy \
    sounddevice soundfile librosa speechbrain openai-whisper TTS ollama

WORKDIR /workspace
CMD ["/bin/bash"]
