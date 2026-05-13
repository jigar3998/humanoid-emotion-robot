# emotion_fusion_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
from collections import deque

VOICE_MAP = {'hap': 'happy', 'sad': 'sad', 'ang': 'angry', 'neu': 'neutral'}
FACE_CONF_THRESHOLD  = 0.65
VOICE_CONF_THRESHOLD = 0.60


class EmotionFusionNode(Node):
    """
    Combines face, voice, and speech signals.
    Detects crying via multi-signal confirmation.
    Publishes unified emotion state to /humanoid/emotion_state.

    Published JSON format:
    {
        "face_emotion":  "sad",
        "voice_emotion": "sad",
        "speech_text":   "I failed my exam",
        "dominant":      "sad",
        "is_crying":     false,
        "confidence":    0.82
    }
    """
    def __init__(self):
        super().__init__('emotion_fusion_node')
        self.face_emotion  = 'neutral'
        self.face_conf     = 0.0
        self.voice_emotion = 'neutral'
        self.voice_conf    = 0.0
        self.speech_text   = ''
        self.pitch_history = deque(maxlen=10)

        self.create_subscription(String, '/humanoid/face_emotion', self.on_face, 10)
        self.create_subscription(String, '/humanoid/voice_emotion', self.on_voice, 10)
        self.create_subscription(String, '/humanoid/speech_text', self.on_speech, 10)
        self.create_subscription(String, '/humanoid/audio_features',
                                 self.on_audio_features, 10)

        self.pub = self.create_publisher(String, '/humanoid/emotion_state', 10)
        self.create_timer(0.5, self.publish_state)

    def on_face(self, msg):
        data = json.loads(msg.data)
        self.face_emotion = data['emotion']
        self.face_conf    = data['confidence']

    def on_voice(self, msg):
        data = json.loads(msg.data)
        self.voice_emotion = VOICE_MAP.get(data['emotion'], data['emotion'])
        self.voice_conf    = data['confidence']

    def on_speech(self, msg):
        self.speech_text = msg.data

    def on_audio_features(self, msg):
        features = json.loads(msg.data)
        self.pitch_history.append(features.get('pitch_variance', 0))

    def detect_crying(self):
        """
        Requires at least 2 of 3 independent signals to confirm crying.
        Prevents false positives from single-signal misclassification.
        """
        face_signal  = self.face_emotion in ['sad', 'fear'] and self.face_conf > 0.6
        avg_pitch    = (sum(self.pitch_history) / len(self.pitch_history)
                        if self.pitch_history else 0)
        voice_signal = avg_pitch > 30.0
        crying_words = ['crying', "can't stop", 'tears', 'sobbing',
                        "won't stop", 'breaking', 'falling apart']
        text_signal  = any(w in self.speech_text.lower() for w in crying_words)
        return sum([face_signal, voice_signal, text_signal]) >= 2

    def determine_dominant(self):
        """
        Priority: crying > voice (if confident) > face (if confident) > neutral
        Voice is more reliable than face for genuine emotional state.
        """
        if self.detect_crying():
            return 'crying', 1.0
        if self.voice_emotion != 'neutral' and self.voice_conf > VOICE_CONF_THRESHOLD:
            return self.voice_emotion, self.voice_conf
        if self.face_conf > FACE_CONF_THRESHOLD:
            return self.face_emotion, self.face_conf
        return 'neutral', 0.5

    def publish_state(self):
        dominant, confidence = self.determine_dominant()
        state = {
            'face_emotion':  self.face_emotion,
            'voice_emotion': self.voice_emotion,
            'speech_text':   self.speech_text,
            'dominant':      dominant,
            'is_crying':     dominant == 'crying',
            'confidence':    round(confidence, 3)
        }
        self.pub.publish(String(data=json.dumps(state)))


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(EmotionFusionNode())
    rclpy.shutdown()
