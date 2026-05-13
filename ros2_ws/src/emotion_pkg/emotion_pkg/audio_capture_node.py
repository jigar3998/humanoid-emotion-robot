# audio_capture_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import sounddevice as sd
import threading

SAMPLE_RATE   = 16000
CHUNK_SECONDS = 3
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


class AudioCaptureNode(Node):
    """
    Continuously captures microphone audio.
    Publishes 3-second chunks to /humanoid/audio_chunk.
    """
    def __init__(self):
        super().__init__('audio_capture_node')
        self.pub    = self.create_publisher(Float32MultiArray, '/humanoid/audio_chunk', 10)
        self.buffer = []
        self.lock   = threading.Lock()

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype='float32',
            callback=self.audio_callback
        )
        self.stream.start()
        self.create_timer(CHUNK_SECONDS, self.publish_chunk)
        self.get_logger().info('Audio capture running')

    def audio_callback(self, indata, frames, time, status):
        with self.lock:
            self.buffer.extend(indata[:, 0].tolist())

    def publish_chunk(self):
        with self.lock:
            if len(self.buffer) < CHUNK_SAMPLES:
                return
            chunk       = self.buffer[:CHUNK_SAMPLES]
            self.buffer = self.buffer[CHUNK_SAMPLES:]
        msg      = Float32MultiArray()
        msg.data = chunk
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(AudioCaptureNode())
    rclpy.shutdown()
