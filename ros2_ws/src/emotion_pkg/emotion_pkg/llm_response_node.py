# llm_response_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import ollama
import json
import threading


RESPONSE_STYLE = {
    'happy':    'match their energy, be joyful and celebratory with them',
    'sad':      'validate their feelings first before any encouragement. Never rush to solutions',
    'angry':    'stay calm, acknowledge their frustration without arguing or dismissing',
    'fear':     'be a steady, reassuring presence. Speak calmly and clearly',
    'disgust':  'acknowledge what bothered them, then gently redirect',
    'surprise': 'engage their curiosity, ask what happened with genuine interest',
    'neutral':  'be warm and engaging, ask an open-ended question to connect',
    'contempt': 'be non-confrontational, try gentle re-engagement without judgment',
    'crying':   'do NOT offer advice or solutions. Just be present. Comfort first, everything else later',
}

FORBIDDEN_PHRASES = [
    "I understand how you feel", "I'm here for you",
    "That must be really hard", "Everything happens for a reason",
    "Look on the bright side", "It could be worse",
    "Stay positive", "I know exactly how you feel",
]


class LLMResponseNode(Node):
    """
    Receives fused emotion state.
    Generates empathetic response via Llama 3.2 3B (Ollama, offline).
    Publishes to /humanoid/llm_response.
    """
    def __init__(self):
        super().__init__('llm_response_node')
        self.declare_parameter('model', 'llama3.2:3b')
        self.model = self.get_parameter('model').value

        self.sub = self.create_subscription(
            String, '/humanoid/emotion_state', self.on_emotion_state, 10
        )
        self.pub = self.create_publisher(String, '/humanoid/llm_response', 10)

        self.processing = False
        self.lock = threading.Lock()
        self.get_logger().info(f'LLM node ready (model: {self.model})')

    def on_emotion_state(self, msg):
        with self.lock:
            if self.processing:
                return
            self.processing = True

        thread = threading.Thread(
            target=self.generate_response, args=(msg.data,), daemon=True
        )
        thread.start()

    def generate_response(self, state_json):
        try:
            state = json.loads(state_json)
            face_emotion  = state.get('face_emotion', 'neutral')
            voice_emotion = state.get('voice_emotion', 'neutral')
            speech_text   = state.get('speech_text', '')
            is_crying     = state.get('is_crying', False)

            if is_crying:
                dominant = 'crying'
            elif voice_emotion != 'neutral' and state.get('confidence', 0) > 0.6:
                dominant = voice_emotion
            else:
                dominant = face_emotion

            style = RESPONSE_STYLE.get(dominant, RESPONSE_STYLE['neutral'])
            prompt = self.build_prompt(
                face_emotion, voice_emotion, speech_text, dominant, style, is_crying
            )

            self.get_logger().info(f'Generating response for: {dominant}')
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            text = response['message']['content'].strip()
            self.get_logger().info(f'Response: {text}')
            self.pub.publish(String(data=text))

        except Exception as e:
            self.get_logger().error(f'LLM error: {e}')
        finally:
            with self.lock:
                self.processing = False

    def build_prompt(self, face, voice, speech, dominant, style, crying):
        person_said = f'"{speech}"' if speech else '[said nothing]'
        crying_note = '\nIMPORTANT: Person appears to be crying.' if crying else ''
        forbidden   = ', '.join(FORBIDDEN_PHRASES)

        return f"""You are a compassionate humanoid robot companion providing genuine emotional support.{crying_note}

What you observed:
- Face expression: {face}
- Voice tone: {voice}
- Person said: {person_said}
- Primary emotional state: {dominant}

Response approach for this situation: {style}

Rules:
1. Respond in 2-3 natural sentences maximum
2. Never start with "I understand" — show understanding through words instead
3. Never use these phrases: {forbidden}
4. If crying: offer presence and comfort ONLY. No advice, no silver linings, no "it will get better"
5. If sad: validate the feeling before any encouragement
6. If happy: genuinely celebrate with them
7. Match emotional intensity — do not be cheerful when they are devastated
8. Speak as a warm, caring presence — not a therapist reading from a script

Respond now:"""


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LLMResponseNode())
    rclpy.shutdown()
