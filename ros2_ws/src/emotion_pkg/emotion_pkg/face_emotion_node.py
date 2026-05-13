# face_emotion_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import json

from .face_detector import FaceDetector
from .emotion_classifier import EmotionClassifier


class FaceEmotionNode(Node):
    """
    Subscribes to Orbbec RGB stream.
    Detects faces, classifies emotion.
    Publishes to /humanoid/face_emotion as JSON.
    """
    def __init__(self):
        super().__init__('face_emotion_node')
        self.declare_parameter('use_trt',        False)
        self.declare_parameter('emotion_model',  '../models/emotion_b2.onnx')
        self.declare_parameter('face_model',     '../models/retinaface_sim.onnx')
        self.declare_parameter('conf_threshold', 0.7)

        use_trt  = self.get_parameter('use_trt').value
        em_model = self.get_parameter('emotion_model').value
        fd_model = self.get_parameter('face_model').value
        conf     = self.get_parameter('conf_threshold').value

        self.face_detector = FaceDetector(fd_model, use_trt=use_trt)
        self.classifier    = EmotionClassifier(em_model, use_trt=use_trt)
        self.conf          = conf
        self.bridge        = CvBridge()

        self.sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.on_frame, 10
        )
        self.pub = self.create_publisher(String, '/humanoid/face_emotion', 10)
        self.get_logger().info('Face emotion node ready')

    def on_frame(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Frame error: {e}')
            return

        boxes = self.face_detector.detect(frame, self.conf)
        if not boxes:
            return

        box  = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        face = self.face_detector.align_face(frame, box)
        emotion, confidence = self.classifier.predict(face)

        self.pub.publish(String(data=json.dumps({
            'emotion': emotion, 'confidence': confidence
        })))


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(FaceEmotionNode())
    rclpy.shutdown()
