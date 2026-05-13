#!/bin/bash
# deploy/setup_nano.sh
# Run once on Jetson Orin Nano while connected to internet.
# Sets up all dependencies for offline operation.

set -e

echo "=== Verifying JetPack ==="
cat /etc/nv_tegra_release
nvcc --version
dpkg -l | grep tensorrt | head -5

echo "=== Installing ROS2 Humble ==="
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu \
    $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list

sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

echo "=== Installing Orbbec SDK ==="
git clone https://github.com/orbbec/OrbbecSDK_ROS2.git ~/ros2_ws/src/orbbec_sdk
sudo apt install -y libusb-1.0-0-dev libudev-dev
cd ~/ros2_ws/src/orbbec_sdk
sudo ./scripts/install_udev_rules.sh
sudo udevadm control --reload && sudo udevadm trigger

cd ~/ros2_ws
colcon build --packages-select orbbec_camera
source install/setup.bash

echo "=== Installing Ollama + Llama 3.2 3B ==="
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
sleep 5
ollama pull llama3.2:3b
ollama run llama3.2:3b "Hello" && echo "LLM OK"

echo "=== Installing Python dependencies ==="
pip3 install onnxruntime opencv-python-headless numpy
pip3 install speechbrain openai-whisper
pip3 install TTS sounddevice soundfile librosa
pip3 install ollama

echo "=== Setting max performance mode ==="
sudo nvpmodel -m 0
sudo jetson_clocks

echo "=== Setup complete. Run TensorRT conversion next. ==="
echo "trtexec --onnx=models/emotion_b2_sim.onnx --saveEngine=models/emotion_b2.trt --fp16 --workspace=2048"
echo "trtexec --onnx=models/retinaface_sim.onnx --saveEngine=models/retinaface.trt --fp16 --workspace=1024"
