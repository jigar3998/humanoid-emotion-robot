# stt_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import whisper
import numpy as np


class STTNode(Node):
    """
    Speech-to-Text using OpenAI Whisper — fully offline.
    Subscribes to audio chunks, publishes transcribed text.

    Model size vs speed on Orin Nano:
      tiny  (~75MB)  — fastest, lower accuracy
      base  (~140MB) — recommended balance
      small (~460MB) — higher accuracy, 2x slower
    """
    def __init__(self):
        super().__init__('stt_node')
        self.declare_parameter('model_size', 'base')
        model_size = self.get_parameter('model_size').value

        self.get_logger().info(f'Loading Whisper {model_size}...')
        self.model = whisper.load_model(model_size)
        self.get_logger().info('Whisper ready')

        self.sub = self.create_subscription(
            Float32MultiArray, '/humanoid/audio_chunk',
            self.transcribe_callback, 10
        )
        self.pub = self.create_publisher(String, '/humanoid/speech_text', 10)

    def transcribe_callback(self, msg):
        audio = np.array(msg.data, dtype=np.float32)

        # Skip if mostly silence
        if np.abs(audio).mean() < 0.002:
            return

        result = self.model.transcribe(audio, language='en', fp16=False)
        text   = result['text'].strip()

        if text:
            self.get_logger().info(f'Heard: "{text}"')
            self.pub.publish(String(data=text))


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(STTNode())
    rclpy.shutdown()
