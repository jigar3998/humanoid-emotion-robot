# tts_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from TTS.api import TTS
import sounddevice as sd
import soundfile as sf
import tempfile
import threading
import os


class TTSNode(Node):
    """
    Converts LLM text response to speech and plays through speaker.
    Uses Coqui TTS — fully offline.
    """
    def __init__(self):
        super().__init__('tts_node')
        self.get_logger().info('Loading TTS model...')
        self.tts = TTS('tts_models/en/ljspeech/tacotron2-DDC')
        self.get_logger().info('TTS ready')

        self.sub = self.create_subscription(
            String, '/humanoid/llm_response', self.speak_callback, 10
        )
        self.speaking = False
        self.lock = threading.Lock()

    def speak_callback(self, msg):
        with self.lock:
            if self.speaking:
                return
            self.speaking = True

        thread = threading.Thread(
            target=self.speak, args=(msg.data,), daemon=True
        )
        thread.start()

    def speak(self, text):
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                tmp = f.name
            self.tts.tts_to_file(text=text, file_path=tmp)
            data, sr = sf.read(tmp)
            sd.play(data, sr)
            sd.wait()
            os.unlink(tmp)
        except Exception as e:
            self.get_logger().error(f'TTS error: {e}')
        finally:
            with self.lock:
                self.speaking = False


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TTSNode())
    rclpy.shutdown()
