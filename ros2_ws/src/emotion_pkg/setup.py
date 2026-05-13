from setuptools import setup
from glob import glob

package_name = 'emotion_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'face_emotion_node  = emotion_pkg.face_emotion_node:main',
            'audio_capture_node = emotion_pkg.audio_capture_node:main',
            'stt_node           = emotion_pkg.stt_node:main',
            'voice_emotion_node = emotion_pkg.voice_emotion_node:main',
            'emotion_fusion     = emotion_pkg.emotion_fusion_node:main',
            'llm_response_node  = emotion_pkg.llm_response_node:main',
            'tts_node           = emotion_pkg.tts_node:main',
        ],
    },
)
