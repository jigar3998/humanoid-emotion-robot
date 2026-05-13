# voice_emotion_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import numpy as np
import json
import threading
import tempfile
import soundfile as sf


class VoiceEmotionNode(Node):
    """
    Detects emotion from voice tone and prosody.
    Uses SpeechBrain wav2vec2 (IEMOCAP).
    Publishes emotion + prosodic features for crying detection.

    Published to /humanoid/voice_emotion:
        {"emotion": "sad", "confidence": 0.82}

    Published to /humanoid/audio_features:
        {"pitch_variance": 12.4, "energy": 0.031}
    """
    def __init__(self):
        super().__init__('voice_emotion_node')
        self.declare_parameter('model_dir', 'models/speechbrain_emotion')
        model_dir = self.get_parameter('model_dir').value

        self.get_logger().info('Loading voice emotion model...')
        from speechbrain.pretrained import EmotionRecognizer
        self.classifier = EmotionRecognizer.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir=model_dir
        )
        self.get_logger().info('Voice emotion model ready')

        self.sub = self.create_subscription(
            Float32MultiArray, '/humanoid/audio_chunk',
            self.on_audio_chunk, 10
        )
        self.pub_emotion   = self.create_publisher(String, '/humanoid/voice_emotion', 10)
        self.pub_features  = self.create_publisher(String, '/humanoid/audio_features', 10)
        self.processing    = False
        self.lock          = threading.Lock()

    def on_audio_chunk(self, msg):
        with self.lock:
            if self.processing:
                return
            self.processing = True
        audio = np.array(msg.data, dtype=np.float32)
        thread = threading.Thread(target=self.process, args=(audio,), daemon=True)
        thread.start()

    def process(self, audio):
        try:
            # Emotion from tone
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, audio, 16000)
                out_prob, score, index, text_lab = self.classifier.classify_file(f.name)

            self.pub_emotion.publish(String(data=json.dumps({
                'emotion': text_lab[0], 'confidence': float(score)
            })))

            # Prosodic features for crying detection
            import librosa
            f0, _, _ = librosa.pyin(audio, fmin=80, fmax=400)
            pitch_variance = float(np.nanstd(f0)) if not np.all(np.isnan(f0)) else 0.0
            energy = float(librosa.feature.rms(y=audio).mean())

            self.pub_features.publish(String(data=json.dumps({
                'pitch_variance': pitch_variance, 'energy': energy
            })))

        except Exception as e:
            self.get_logger().error(f'Voice emotion error: {e}')
        finally:
            with self.lock:
                self.processing = False


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(VoiceEmotionNode())
    rclpy.shutdown()
