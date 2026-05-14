from launch import LaunchDescription
from launch_ros.actions import Node

MODEL_DIR = '/home/user/humanoid_vision/models'


def generate_launch_description():
    return LaunchDescription([

        # Camera
        Node(
            package='orbbec_camera',
            executable='orbbec_camera_node',
            name='camera',
            parameters=[{
                'color_width': 1280, 'color_height': 720, 'color_fps': 30,
                'depth_width': 640,  'depth_height': 480,
                'enable_color': True, 'enable_depth': True,
            }]
        ),

        # Face emotion — HSEmotions enet_b2_8 + InsightFace detection
        # On Nano: set use_trt=True and provide trt engine paths
        Node(
            package='emotion_pkg',
            executable='face_emotion_node',
            parameters=[{
                'use_trt':           True,
                'trt_face_model':    f'{MODEL_DIR}/retinaface.trt',
                'trt_emotion_model': f'{MODEL_DIR}/enet_b2_8.trt',
                'conf_threshold':    0.5,
            }]
        ),

        # Audio capture
        Node(package='emotion_pkg', executable='audio_capture_node'),

        # Speech-to-text
        Node(
            package='emotion_pkg',
            executable='stt_node',
            parameters=[{'model_size': 'base'}]
        ),

        # Voice emotion
        Node(package='emotion_pkg', executable='voice_emotion_node'),

        # Emotion fusion
        Node(package='emotion_pkg', executable='emotion_fusion'),

        # LLM response
        Node(
            package='emotion_pkg',
            executable='llm_response_node',
            parameters=[{'model': 'llama3.2:3b'}]
        ),

        # Text-to-speech
        Node(package='emotion_pkg', executable='tts_node'),
    ])
