# Socially Intelligent Humanoid Robot
## Complete Build Guide: Emotion Perception + Emotional Support System
### Mac (Development) → Jetson Orin Nano (Deployment)

---

> **Goal:** A humanoid that perceives human emotions through face, voice, and speech — then responds with genuine, contextual emotional support entirely offline.
>
> **Hardware:** Orbbec Gemini 335/335L + Jetson Orin Nano 8GB + Microphone
>
> **Dev Machine:** Apple Mac (M-series)
>
> **Key constraint:** 100% offline after initial setup. No API calls, no cloud, no internet required at runtime.

---

## Table of Contents

1. [Full System Architecture](#1-full-system-architecture)
2. [Component Decisions & Rationale](#2-component-decisions--rationale)
3. [RAM & Resource Budget](#3-ram--resource-budget)
4. [Mac Environment Setup](#4-mac-environment-setup)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Face Emotion Model — Training on Mac](#6-face-emotion-model--training-on-mac)
7. [ONNX Export & Validation](#7-onnx-export--validation)
8. [Voice Emotion Detection](#8-voice-emotion-detection)
9. [Speech-to-Text (Whisper)](#9-speech-to-text-whisper)
10. [Local LLM — Emotional Understanding & Response](#10-local-llm--emotional-understanding--response)
11. [Text-to-Speech — Emotional Voice Output](#11-text-to-speech--emotional-voice-output)
12. [ROS2 Node Development on Mac](#12-ros2-node-development-on-mac)
13. [Docker Testing on Mac](#13-docker-testing-on-mac)
14. [Transferring to Orin Nano](#14-transferring-to-orin-nano)
15. [Orin Nano Setup](#15-orin-nano-setup)
16. [TensorRT Conversion on Nano](#16-tensorrt-conversion-on-nano)
17. [Orbbec Camera Integration](#17-orbbec-camera-integration)
18. [Running the Full Pipeline](#18-running-the-full-pipeline)
19. [Emotion to Response Logic](#19-emotion-to-response-logic)
20. [Crying Detection](#20-crying-detection)
21. [LLM Prompt Engineering](#21-llm-prompt-engineering)
22. [Troubleshooting](#22-troubleshooting)
23. [Project Folder Structure](#23-project-folder-structure)
24. [Build Timeline](#24-build-timeline)

---

## 1. Full System Architecture

### The Emotional AI Loop

```
PERCEIVE
  Gemini 335L (RGB)          Microphone
       |                          |
  Face Detection             Audio Stream
  (RetinaFace TRT)               |
       |                  +------+------+
  Face Emotion         Voice Emotion   Speech-to-Text
  (EfficientNet-B2)    (SpeechBrain)   (Whisper offline)

         |
UNDERSTAND
  Emotion Fusion Node
  - Combines face + voice + speech signals
  - Detects crying (multi-signal)
  - Determines dominant emotional state
       |
  Local LLM (Llama 3.2 3B via Ollama)
  - Receives: emotion state + what person said
  - Understands WHY they feel that way
  - Generates empathetic, contextual response

         |
RESPOND
  Text-to-Speech (Coqui TTS)
  - Speaks response with warm tone

  Gesture Controller
  - Triggers appropriate body language

  Expression Controller (if robot has face/screen)
  - Mirrors appropriate facial state
```

### ROS2 Topic Map

```
/camera/color/image_raw      --> face_emotion_node --> /humanoid/face_emotion
/microphone/audio_chunk      --> voice_emotion_node --> /humanoid/voice_emotion
                             --> stt_node           --> /humanoid/speech_text
                             --> audio_features     --> /humanoid/audio_features

All four above --> emotion_fusion_node --> /humanoid/emotion_state

/humanoid/emotion_state --> llm_response_node --> /humanoid/llm_response

/humanoid/llm_response --> tts_node      --> speaker output
                       --> gesture_node  --> /humanoid/gesture_cmd
```

---

## 2. Component Decisions & Rationale

| Layer | Component | Why |
|---|---|---|
| Camera | Orbbec Gemini 335/335L | On-chip depth compute — zero GPU load on Nano. 335L for outdoor/IP65 |
| Face detection | RetinaFace (MobileNet backbone) | Best face detection accuracy/speed. Pretrained ONNX available free |
| Face emotion | EfficientNet-B2 + TensorRT FP16 | ~78-80% accuracy, ~20fps on Nano, fits in memory |
| Voice emotion | SpeechBrain wav2vec2 | Detects emotion from tone/prosody, offline, ARM64 compatible |
| Speech-to-text | Whisper base (offline) | Understands what person said — critical for LLM context |
| Understanding | Llama 3.2 3B via Ollama | Fits in 8GB Nano RAM, generates empathetic responses, fully offline |
| TTS | Coqui TTS | Emotional voice output, free, offline, runs on Nano |
| Inference runtime | TensorRT FP16 | Halves memory, doubles inference speed on Jetson |
| Robot framework | ROS2 Humble | Industry standard robotics middleware, works offline |
| Training GPU | Mac MPS (Metal) | M-series Mac GPU, 3-5x faster than CPU for PyTorch |

### Why Voice + Speech Matters as Much as Face

A face-only system misses critical signals:
- People can smile while internally devastated
- Crying with head down — no face visible
- Anger expressed quietly through word choice
- Fear disguised as calm demeanor

Voice tone reveals emotional intensity. Speech content reveals the cause. The LLM combines all three signals to produce a genuinely contextual response — not a canned reply.

---

## 3. RAM & Resource Budget

Jetson Orin Nano has 8GB unified RAM shared between CPU and GPU.

| Component | RAM Usage |
|---|---|
| EfficientNet-B2 TRT FP16 | ~200 MB |
| RetinaFace TRT FP16 | ~80 MB |
| Whisper base | ~140 MB |
| SpeechBrain voice emotion | ~300 MB |
| Llama 3.2 3B via Ollama | ~2.0 GB |
| Coqui TTS | ~200 MB |
| ROS2 + system processes | ~1.5 GB |
| **Total** | **~4.4 GB** |

Headroom: ~3.6 GB — sufficient for navigation, motion control, other robot systems.

**If using 4GB Nano:** Replace Llama 3.2 3B with Llama 3.2 1B (~1.0 GB). Quality drops slightly but remains functional.

---

## 4. Mac Environment Setup

### 4.1 Prerequisites

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# System dependencies
brew install python@3.10 cmake git wget portaudio ffmpeg
```

portaudio and ffmpeg are required for audio capture and processing.

### 4.2 Python Virtual Environment

```bash
# Create isolated environment
python3.10 -m venv ~/humanoid_env

# Activate — run this every new terminal session
source ~/humanoid_env/bin/activate

# Verify
python --version  # must show 3.10.x
```

### 4.3 Install All Python Packages

```bash
pip install --upgrade pip

# Core ML
pip install torch torchvision torchaudio
pip install onnx onnxruntime
pip install efficientnet_pytorch

# Vision
pip install opencv-python
pip install mediapipe
pip install Pillow

# Training
pip install albumentations
pip install scikit-learn
pip install matplotlib
pip install tqdm

# Audio and Speech
pip install speechbrain
pip install openai-whisper
pip install pyaudio
pip install sounddevice
pip install soundfile
pip install librosa

# LLM
pip install ollama

# TTS
pip install TTS

# Optional: experiment tracking
pip install wandb
```

### 4.4 Install Ollama on Mac (LLM development)

```bash
brew install ollama

# Start Ollama service
ollama serve &

# Pull Llama 3.2 3B — same model that will run on Nano
ollama pull llama3.2:3b

# Test
ollama run llama3.2:3b "Say hello warmly to someone who is sad"
```

### 4.5 Verify Mac GPU (MPS)

Create `scripts/test_mps.py`:

```python
import torch
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

x = torch.randn(1000, 1000).to(device)
y = torch.matmul(x, x)
print("MPS compute: OK" if device.type == "mps" else "CPU only — training will be slow")
```

```bash
python scripts/test_mps.py
# Expected: Device: mps
```

---

## 5. Dataset Preparation

### 5.1 Emotion Classes

The system recognizes 8 emotions (7 Ekman basics + contempt from AffectNet):

| Index | Emotion | Humanoid Response Strategy |
|---|---|---|
| 0 | angry | Stay calm, acknowledge frustration, do not argue |
| 1 | contempt | Gentle re-engagement, non-confrontational |
| 2 | disgust | Acknowledge, redirect positively |
| 3 | fear | Reassure, steady and calm presence |
| 4 | happy | Match energy, celebrate with them |
| 5 | neutral | Light engagement, open-ended question |
| 6 | sad | Validate first, then gently encourage |
| 7 | surprise | Engage curiosity, ask what happened |

Plus crying — detected separately via multi-signal fusion (see Section 20).

### 5.2 FER2013 Dataset

Primary training dataset. 35,887 grayscale face images, 7 emotion classes.

Steps:
1. Create account at kaggle.com
2. Download: https://www.kaggle.com/datasets/msambare/fer2013
3. Extract:

```bash
unzip archive.zip -d data/fer2013/
```

Expected structure:
```
data/fer2013/
├── train/
│   ├── angry/        (3,995 images)
│   ├── disgust/      (436 images)
│   ├── fear/         (4,097 images)
│   ├── happy/        (7,215 images)
│   ├── neutral/      (4,965 images)
│   ├── sad/          (4,830 images)
│   └── surprise/     (3,171 images)
└── test/
    └── (same structure)
```

Note: disgust is severely underrepresented (436 vs 7215 for happy). Training script handles this with class weighting.

### 5.3 AffectNet Dataset (Recommended)

450,000 real-world face images. Adds contempt as 8th class. Produces models that generalize far better than FER2013 alone.

1. Register at http://mohammadmahoormoradi.net/affectnet.htm (free for research)
2. Download on Mac with good internet — takes 2-4 hours
3. Store at `data/affectnet/`

Training strategy: pretrain on AffectNet (large, diverse) then fine-tune on FER2013.

### 5.4 Dataset Loader

Create `training/dataset.py`:

```python
# training/dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

EMOTIONS = ['angry', 'contempt', 'disgust', 'fear',
            'happy', 'neutral', 'sad', 'surprise']
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
NUM_CLASSES = len(EMOTIONS)


class EmotionDataset(Dataset):
    """
    Loads FER2013 or AffectNet from folder structure:
        root/{split}/{emotion_class}/image.jpg
    Automatically converts grayscale to RGB for EfficientNet.
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.transform = transform
        self.samples = []
        split_dir = os.path.join(root_dir, split)

        for emotion in EMOTIONS:
            emotion_dir = os.path.join(split_dir, emotion)
            if not os.path.exists(emotion_dir):
                continue
            for fname in os.listdir(emotion_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(emotion_dir, fname),
                        EMOTION_TO_IDX[emotion]
                    ))

        print(f"  [{split}] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label


def compute_class_weights(dataset):
    """
    Inverse-frequency weights for imbalanced classes.
    Passed to CrossEntropyLoss to prevent bias toward majority classes.
    """
    counts = [0] * NUM_CLASSES
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (NUM_CLASSES * c) if c > 0 else 0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def get_transforms(split='train'):
    if split == 'train':
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=0.5),
            A.GaussNoise(p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.CoarseDropout(max_holes=4, max_height=35,
                            max_width=35, p=0.25),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def get_dataloaders(data_dir, batch_size=32):
    train_ds = EmotionDataset(data_dir, 'train', get_transforms('train'))
    val_ds   = EmotionDataset(data_dir, 'test',  get_transforms('val'))
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    weights = compute_class_weights(train_ds)
    return train_loader, val_loader, weights
```

---

## 6. Face Emotion Model — Training on Mac

### 6.1 Why EfficientNet-B2

| Model | Accuracy FER2013 | Speed on Nano | Size |
|---|---|---|---|
| MobileNetV2 | ~72% | ~35fps | 14MB |
| EfficientNet-B0 | ~75% | ~25fps | 20MB |
| EfficientNet-B2 | ~78-80% | ~20fps | 29MB |
| EfficientNet-B4 | ~82% | ~8fps | 75MB |
| ResNet-50 | ~79% | ~10fps | 98MB |

EfficientNet-B2 is the precision/speed sweet spot for Orin Nano.

### 6.2 Training Script

Create `training/train.py`:

```python
# training/train.py
import os
import json
import torch
import torch.nn as nn
from torch import optim
from efficientnet_pytorch import EfficientNet
from dataset import get_dataloaders, NUM_CLASSES, EMOTIONS
from tqdm import tqdm

DATA_DIR   = '../data/fer2013'
SAVE_DIR   = '../models'
EPOCHS     = 60
BATCH_SIZE = 32
LR         = 1e-4

os.makedirs(SAVE_DIR, exist_ok=True)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

model = EfficientNet.from_pretrained('efficientnet-b2')
model._fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model._fc.in_features, NUM_CLASSES)
)
model = model.to(device)

train_loader, val_loader, class_weights = get_dataloaders(DATA_DIR, BATCH_SIZE)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)

best_acc = 0
history  = {'train_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += model(imgs).argmax(1).eq(labels).sum().item()
            total   += labels.size(0)

    val_acc = correct / total * 100
    print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    scheduler.step(val_acc)
    history['train_loss'].append(avg_loss)
    history['val_acc'].append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f'{SAVE_DIR}/emotion_b2_best.pth')
        print(f"  Saved best ({val_acc:.2f}%)")

torch.save(model.state_dict(), f'{SAVE_DIR}/emotion_b2_final.pth')
with open(f'{SAVE_DIR}/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"Done. Best accuracy: {best_acc:.2f}%")
```

Run:
```bash
cd training
python train.py
```

Expected training time on Mac M-series: M1 ~4-5 hours, M2/M3 ~2-3 hours, M4 ~1.5-2 hours.

### 6.3 Evaluate Per-Class Accuracy

Create `training/evaluate.py`:

```python
# training/evaluate.py
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report
from dataset import get_dataloaders, EMOTIONS, NUM_CLASSES

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = EfficientNet.from_pretrained('efficientnet-b2')
model._fc = nn.Sequential(nn.Dropout(0.4),
                           nn.Linear(model._fc.in_features, NUM_CLASSES))
model.load_state_dict(torch.load('../models/emotion_b2_best.pth',
                                  map_location='cpu'))
model = model.to(device).eval()

_, val_loader, _ = get_dataloaders('../data/fer2013', batch_size=64)

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        preds = model(imgs.to(device)).argmax(dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds, target_names=EMOTIONS))
```

---

## 7. ONNX Export & Validation

ONNX is the bridge between Mac training and Nano deployment. TensorRT engines must be built on the Nano — they are hardware-specific. ONNX is the portable intermediate format.

### 7.1 Export Script

Create `export/export_onnx.py`:

```python
# export/export_onnx.py
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import onnx
import onnxruntime as ort
import numpy as np

NUM_CLASSES = 8
MODEL_PATH  = '../models/emotion_b2_best.pth'
OUTPUT_PATH = '../models/emotion_b2.onnx'

model = EfficientNet.from_pretrained('efficientnet-b2')
model._fc = nn.Sequential(nn.Dropout(0.4),
                           nn.Linear(model._fc.in_features, NUM_CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval().cpu()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, dummy, OUTPUT_PATH,
    opset_version=17,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"Exported: {OUTPUT_PATH}")

onnx.checker.check_model(onnx.load(OUTPUT_PATH))
print("ONNX structure valid")

sess = ort.InferenceSession(OUTPUT_PATH)
out  = sess.run(None, {'input': dummy.numpy()})[0]
emotions = ['angry','contempt','disgust','fear','happy','neutral','sad','surprise']
print(f"Output shape: {out.shape}, predicted: {emotions[out.argmax()]}")
```

```bash
cd export && python export_onnx.py
```

### 7.2 Download Pretrained RetinaFace

```bash
pip install insightface
python -c "
import insightface
from insightface.model_zoo import get_model
det = get_model('retinaface_r50_v1')
det.prepare(ctx_id=-1)
print('RetinaFace cached')
"
cp ~/.insightface/models/retinaface_r50_v1/*.onnx ../models/retinaface.onnx
```

### 7.3 Simplify ONNX for Better TRT Conversion

```bash
pip install onnx-simplifier
python -m onnxsim ../models/emotion_b2.onnx ../models/emotion_b2_sim.onnx
python -m onnxsim ../models/retinaface.onnx ../models/retinaface_sim.onnx
```

Use the _sim.onnx versions for TensorRT conversion on Nano.

---

## 8. Voice Emotion Detection

Tone of voice reveals what the face often hides. This component captures emotional intensity and genuineness from audio prosody.

### 8.1 Voice Emotion Detector

Create `audio/voice_emotion.py`:

```python
# audio/voice_emotion.py
import numpy as np
import tempfile
import soundfile as sf
from speechbrain.pretrained import EmotionRecognizer


class VoiceEmotionDetector:
    """
    Detects emotion from audio tone and prosody.
    Uses wav2vec2 pretrained on IEMOCAP dataset.
    Detects: happy, sad, angry, neutral.
    Runs fully offline after first model download.
    """
    def __init__(self, save_dir='models/speechbrain_emotion'):
        print("Loading voice emotion model...")
        self.classifier = EmotionRecognizer.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir=save_dir
        )
        print("Voice emotion model ready")

    def predict_from_array(self, audio_array, sample_rate=16000):
        """
        Predict emotion from float32 numpy audio array at 16kHz mono.
        Returns (emotion_label, confidence_score).
        """
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio_array, sample_rate)
            out_prob, score, index, text_lab = \
                self.classifier.classify_file(f.name)
        return text_lab[0], float(score)

    def extract_audio_features(self, audio_array, sample_rate=16000):
        """
        Extract prosodic features for crying detection.
        Returns pitch variance and energy level.
        """
        import librosa
        f0, _, _ = librosa.pyin(audio_array, fmin=80, fmax=400)
        pitch_variance = float(np.nanstd(f0)) if not np.all(np.isnan(f0)) else 0.0
        energy = float(librosa.feature.rms(y=audio_array).mean())
        return {'pitch_variance': pitch_variance, 'energy': energy}
```

---

## 9. Speech-to-Text (Whisper)

Knowing what the person said is critical for the LLM to generate a meaningful response. "I failed my exam" vs "I'm just tired" vs "I don't know what to do anymore" all show a sad face but require very different responses.

### 9.1 STT Node

Create `ros2_ws/src/emotion_pkg/emotion_pkg/stt_node.py`:

```python
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
```

---

## 10. Local LLM — Emotional Understanding & Response

This is the brain of the system. The LLM receives all perception signals and generates a genuinely contextual, empathetic response. Runs 100% offline via Ollama.

### 10.1 Install Ollama on Orin Nano

```bash
# Run on Nano while online, one time only
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model (~2GB download)
ollama pull llama3.2:3b

# Verify it responds
ollama run llama3.2:3b "Respond warmly to someone who is feeling sad"
```

### 10.2 LLM Response Node

Create `ros2_ws/src/emotion_pkg/emotion_pkg/llm_response_node.py`:

```python
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
```

### 10.3 Test LLM Prompts on Mac

Create `scripts/test_llm_prompts.py`:

```python
# scripts/test_llm_prompts.py
import ollama

test_cases = [
    {'label': 'Person crying after breakup',
     'face': 'sad', 'voice': 'sad', 'crying': True,
     'speech': "We broke up after 3 years. I don't know what to do."},
    {'label': 'Failed exam',
     'face': 'sad', 'voice': 'neutral', 'crying': False,
     'speech': "I failed my exam again."},
    {'label': 'Got promoted',
     'face': 'happy', 'voice': 'happy', 'crying': False,
     'speech': "I just got promoted!"},
    {'label': 'Anxious about presentation',
     'face': 'fear', 'voice': 'neutral', 'crying': False,
     'speech': "I have a big presentation tomorrow and I'm terrified."},
    {'label': 'Boss took credit',
     'face': 'angry', 'voice': 'angry', 'crying': False,
     'speech': "My boss took credit for my entire project."},
    {'label': 'Silent but sad',
     'face': 'sad', 'voice': 'sad', 'crying': False,
     'speech': ''},
]

for case in test_cases:
    print(f"\n{'='*55}")
    print(f"Scenario: {case['label']}")
    print(f"Said: {case['speech'] or '(nothing)'}")

    prompt = f"""You are a compassionate humanoid robot.
Face shows: {case['face']}, voice tone: {case['voice']}.
{'Person is crying.' if case['crying'] else ''}
Person said: "{case['speech']}"
Respond with genuine empathy in 2-3 sentences.
Never use generic phrases like "I understand" or "I'm here for you"."""

    resp = ollama.chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    print(f"Response: {resp['message']['content']}")
```

```bash
python scripts/test_llm_prompts.py
```

Iterate on prompts until every scenario produces a genuinely warm, contextual response.

---

## 11. Text-to-Speech — Emotional Voice Output

The response must sound warm and human — not robotic.

### 11.1 TTS Node

Create `ros2_ws/src/emotion_pkg/emotion_pkg/tts_node.py`:

```python
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
```

---

## 12. ROS2 Node Development on Mac

### 12.1 Audio Capture Node

Create `ros2_ws/src/emotion_pkg/emotion_pkg/audio_capture_node.py`:

```python
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
```

### 12.2 Face Detector

Create `ros2_ws/src/emotion_pkg/emotion_pkg/face_detector.py`:

```python
# face_detector.py
import cv2
import numpy as np
import onnxruntime as ort


class FaceDetector:
    """
    RetinaFace face detector.
    ONNX Runtime on Mac for testing.
    TensorRT on Nano for production.
    """
    def __init__(self, model_path, use_trt=False):
        self.use_trt = use_trt
        if not use_trt:
            self.session    = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
        else:
            self._load_trt(model_path)

    def _load_trt(self, engine_path):
        import tensorrt as trt
        import pycuda.autoinit
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def preprocess(self, image, size=(640, 640)):
        h, w  = image.shape[:2]
        img   = cv2.resize(image, size).astype(np.float32)
        img  -= np.array([104, 117, 123])
        img   = img.transpose(2, 0, 1)[np.newaxis]
        return img, (w, h)

    def detect(self, image, conf_thresh=0.7):
        inp, orig = self.preprocess(image)
        if not self.use_trt:
            outs = self.session.run(None, {self.input_name: inp})
        else:
            outs = self._trt_infer(inp)
        return self._parse_boxes(outs, orig, conf_thresh)

    def _parse_boxes(self, outs, orig_size, conf_thresh):
        # Parse RetinaFace output — implementation depends on ONNX model variant
        return []

    def align_face(self, image, box, size=224):
        x1, y1, x2, y2 = [int(v) for v in box]
        pad = int((x2 - x1) * 0.1)
        x1  = max(0, x1 - pad)
        y1  = max(0, y1 - pad)
        x2  = min(image.shape[1], x2 + pad)
        y2  = min(image.shape[0], y2 + pad)
        return cv2.resize(image[y1:y2, x1:x2], (size, size))
```

### 12.3 Emotion Classifier

Create `ros2_ws/src/emotion_pkg/emotion_pkg/emotion_classifier.py`:

```python
# emotion_classifier.py
import numpy as np
import onnxruntime as ort
import cv2

EMOTIONS = ['angry', 'contempt', 'disgust', 'fear',
            'happy', 'neutral', 'sad', 'surprise']
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EmotionClassifier:
    """
    EfficientNet-B2 emotion classifier.
    ONNX Runtime on Mac. TensorRT FP16 on Nano.
    """
    def __init__(self, model_path, use_trt=False):
        self.use_trt = use_trt
        if not use_trt:
            self.session    = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
        else:
            self._load_trt(model_path)

    def _load_trt(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = \
            self._allocate_buffers(cuda)

    def _allocate_buffers(self, cuda):
        import tensorrt as trt
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in self.engine:
            size     = trt.volume(self.engine.get_binding_shape(binding))
            dtype    = trt.nptype(self.engine.get_binding_dtype(binding))
            host     = cuda.pagelocked_empty(size, dtype)
            device   = cuda.mem_alloc(host.nbytes)
            bindings.append(int(device))
            entry    = {'host': host, 'device': device}
            (inputs if self.engine.binding_is_input(binding) else outputs).append(entry)
        return inputs, outputs, bindings, stream

    def preprocess(self, face_bgr):
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        face = (face - MEAN) / STD
        return face.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def predict(self, face_bgr):
        inp = self.preprocess(face_bgr)
        if not self.use_trt:
            out = self.session.run(None, {self.input_name: inp})[0][0]
        else:
            out = self._trt_infer(inp)
        exp   = np.exp(out - out.max())
        probs = exp / exp.sum()
        idx   = probs.argmax()
        return EMOTIONS[idx], float(probs[idx])

    def _trt_infer(self, inp):
        import pycuda.driver as cuda
        np.copyto(self.inputs[0]['host'], inp.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'],
                               self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'],
                               self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].copy()
```

### 12.4 Face Emotion Node

Create `ros2_ws/src/emotion_pkg/emotion_pkg/face_emotion_node.py`:

```python
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
```

### 12.5 Emotion Fusion Node

Create `ros2_ws/src/emotion_pkg/emotion_pkg/emotion_fusion_node.py`:

```python
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
```

### 12.6 Package Configuration

Create `ros2_ws/src/emotion_pkg/setup.py`:

```python
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
```

Create `ros2_ws/src/emotion_pkg/package.xml`:

```xml
<?xml version="1.0"?>
<package format="3">
  <name>emotion_pkg</name>
  <version>0.1.0</version>
  <description>Socially intelligent humanoid emotion and support system</description>
  <maintainer email="you@example.com">Your Name</maintainer>
  <license>MIT</license>
  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>std_msgs</depend>
  <depend>cv_bridge</depend>
</package>
```

---

## 13. Docker Testing on Mac

### 13.1 Dockerfile

Create `docker/Dockerfile`:

```dockerfile
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl gnupg2 lsb-release \
    python3 python3-pip \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    portaudio19-dev libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ROS2 Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y ros-humble-ros-base python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install onnxruntime opencv-python-headless numpy \
    sounddevice soundfile librosa speechbrain openai-whisper TTS ollama

WORKDIR /workspace
CMD ["/bin/bash"]
```

### 13.2 Build and Test

```bash
cd docker
docker build -t humanoid-ros2 .

docker run -it --platform linux/arm64 \
    -v $(pwd)/../ros2_ws:/workspace/ros2_ws \
    -v $(pwd)/../models:/workspace/models \
    humanoid-ros2

# Inside container:
source /opt/ros/humble/setup.bash
cd /workspace/ros2_ws
colcon build
source install/setup.bash
ros2 run emotion_pkg face_emotion_node
```

---

## 14. Transferring to Orin Nano

### 14.1 Via SCP (WiFi)

```bash
# Find Nano IP — on Nano: ip addr show

rsync -avz --progress \
    ~/humanoid_vision/ \
    user@192.168.1.100:/home/user/humanoid_vision/
```

### 14.2 Via USB Drive (Offline)

```bash
# Mac to USB
cp -r ~/humanoid_vision/ /Volumes/USB_DRIVE/

# USB to Nano (on Nano)
cp -r /media/user/USB_DRIVE/humanoid_vision/ ~/
```

---

## 15. Orin Nano Setup

Run all steps once while connected to internet.

### 15.1 Verify JetPack 6

```bash
cat /etc/nv_tegra_release     # JetPack version
nvcc --version                # CUDA 12.x expected
dpkg -l | grep tensorrt       # TensorRT packages
tegrastats                    # real-time RAM/GPU monitor
```

### 15.2 Install ROS2 Humble

```bash
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
source ~/.bashrc
```

### 15.3 Install Orbbec SDK

```bash
git clone https://github.com/orbbec/OrbbecSDK_ROS2.git ~/ros2_ws/src/orbbec_sdk
sudo apt install -y libusb-1.0-0-dev libudev-dev

cd ~/ros2_ws/src/orbbec_sdk
sudo ./scripts/install_udev_rules.sh
sudo udevadm control --reload && sudo udevadm trigger

cd ~/ros2_ws
colcon build --packages-select orbbec_camera
source install/setup.bash
```

### 15.4 Install Ollama + LLM

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama3.2:3b
ollama run llama3.2:3b "Hello"   # verify works
```

### 15.5 Install Python Dependencies

```bash
pip3 install onnxruntime opencv-python-headless numpy
pip3 install speechbrain openai-whisper
pip3 install TTS sounddevice soundfile librosa
pip3 install ollama
```

### 15.6 Set Performance Mode

```bash
sudo nvpmodel -m 0     # max power mode
sudo jetson_clocks     # lock clocks at maximum
```

---

## 16. TensorRT Conversion on Nano

Must run on Nano. TRT engines are hardware-specific and cannot be built elsewhere.

### 16.1 Convert Emotion Model

```bash
trtexec \
    --onnx=~/humanoid_vision/models/emotion_b2_sim.onnx \
    --saveEngine=~/humanoid_vision/models/emotion_b2.trt \
    --fp16 \
    --workspace=2048 \
    --verbose
# Takes 5-15 minutes. Run once only.
```

### 16.2 Convert Face Detector

```bash
trtexec \
    --onnx=~/humanoid_vision/models/retinaface_sim.onnx \
    --saveEngine=~/humanoid_vision/models/retinaface.trt \
    --fp16 \
    --workspace=1024 \
    --verbose
```

### 16.3 Verify TRT Engines

```bash
python3 -c "
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open('models/emotion_b2.trt', 'rb') as f:
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
print(f'Engine OK. Bindings: {engine.num_bindings}')
"
```

---

## 17. Orbbec Camera Integration

### 17.1 Verify Camera

```bash
lsusb | grep 2bc5       # Orbbec vendor ID — must appear
ros2 launch orbbec_camera gemini_330_series.launch.py

# In second terminal:
ros2 topic hz /camera/color/image_raw    # must show ~30 Hz
ros2 topic hz /camera/depth/image_raw    # must show ~30 Hz
```

### 17.2 Master Launch File

Create `ros2_ws/src/emotion_pkg/launch/humanoid_emotion.launch.py`:

```python
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

        # Face emotion
        Node(
            package='emotion_pkg',
            executable='face_emotion_node',
            parameters=[{
                'use_trt':        True,
                'emotion_model':  f'{MODEL_DIR}/emotion_b2.trt',
                'face_model':     f'{MODEL_DIR}/retinaface.trt',
                'conf_threshold': 0.7,
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
```

---

## 18. Running the Full Pipeline

### 18.1 Start Ollama First

```bash
# Terminal 1 — keep running in background
ollama serve
```

### 18.2 Launch Everything

```bash
# Terminal 2
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch emotion_pkg humanoid_emotion.launch.py
```

### 18.3 Monitor Topics

```bash
ros2 topic echo /humanoid/emotion_state   # fused state
ros2 topic echo /humanoid/llm_response    # what robot will say
ros2 topic echo /humanoid/speech_text     # what person said
```

### 18.4 Offline Verification

```bash
sudo nmcli networking off
ros2 launch emotion_pkg humanoid_emotion.launch.py
# Must work completely without internet
sudo nmcli networking on
```

---

## 19. Emotion to Response Logic

The LLM handles nuance automatically. These are the general behavioral rules built into every prompt:

| Situation | System Behavior |
|---|---|
| Person is crying | Comfort ONLY. No advice, no motivation, no silver linings |
| Person failed something | Validate first, then reframe, then gently encourage |
| Person is angry | Stay calm, acknowledge frustration, do not escalate |
| Person is scared | Steady, reassuring, calm tone |
| Person is happy | Match energy, celebrate genuinely |
| Person says nothing, looks sad | Ask one soft open-ended question |
| Face says fine, voice says sad | Trust voice — it is harder to fake |
| Mixed conflicting signals | Dominant signal wins: crying > voice > face |

---

## 20. Crying Detection

Crying requires multi-signal confirmation to avoid false positives.

```python
def detect_crying(self):
    """
    Requires at least 2 of 3 independent signals.

    Signal 1 — Face: sad or fear expression with confidence > 0.6
    Signal 2 — Voice: high pitch variance (voice cracking, breaking)
                      threshold: average pitch variance > 30.0
    Signal 3 — Speech: text contains crying-related words

    2+ signals = crying confirmed
    """
    face_signal = (
        self.face_emotion in ['sad', 'fear'] and
        self.face_conf > 0.6
    )

    avg_pitch    = (sum(self.pitch_history) / len(self.pitch_history)
                    if self.pitch_history else 0)
    voice_signal = avg_pitch > 30.0

    crying_words = ['crying', "can't stop", 'tears', 'sobbing',
                    "won't stop", 'breaking', 'falling apart']
    text_signal  = any(w in self.speech_text.lower() for w in crying_words)

    return sum([face_signal, voice_signal, text_signal]) >= 2
```

When crying is confirmed:
- LLM receives crying-mode instruction: comfort only, no solutions
- Response is presence-focused: "I'm right here with you"
- Do not offer advice, silver linings, or encouragement

---

## 21. LLM Prompt Engineering

Prompts are the most important tunable parameter. Iterate extensively before deployment.

### Testing Checklist

Test every scenario before deploying:

- Person crying after a loss
- Person happy about an achievement
- Person angry at someone else (not the robot)
- Person afraid of something upcoming
- Person says nothing — silent but visibly sad
- Mixed conflicting signals (happy face, sad voice)
- Background noise causing wrong transcription
- Very short responses ("I'm fine" when clearly not fine)

### Anti-Patterns to Avoid

These phrases make the robot feel robotic and clinical. Instruct the LLM to never use them:

```python
FORBIDDEN_PHRASES = [
    "I understand how you feel",
    "I'm here for you",
    "That must be really hard",
    "Everything happens for a reason",
    "Look on the bright side",
    "It could be worse",
    "Stay positive",
    "I know exactly how you feel",
]
```

### Response Length Guidelines

```
Crying or crisis:   1-2 short sentences. Less is more.
Sad:                2-3 sentences. Validate then one gentle thought.
Angry:              2 sentences. Acknowledge then one grounding question.
Happy:              1-2 sentences. Match their energy.
Neutral:            1 sentence plus one open question.
```

---

## 22. Troubleshooting

### Camera not detected
```bash
lsusb | grep 2bc5
sudo udevadm control --reload && sudo udevadm trigger
# Replug USB cable
```

### TensorRT conversion fails
```bash
# Verify ONNX opset is 17 or lower
python3 -c "import onnx; m=onnx.load('emotion_b2.onnx'); print(m.opset_import)"

# Simplify ONNX first
python3 -m onnxsim emotion_b2.onnx emotion_b2_sim.onnx

# Watch memory during conversion
tegrastats
```

### LLM response too slow
```bash
# Verify Ollama is using GPU (not CPU)
ollama run llama3.2:3b "test" --verbose

# Ensure max performance mode
sudo nvpmodel -m 0 && sudo jetson_clocks
```

### Whisper not transcribing
```bash
# Increase chunk size for better accuracy
# In audio_capture_node.py: CHUNK_SECONDS = 5

# Check microphone is detected
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

### Out of RAM on Nano
```bash
tegrastats   # identify which process consumes most memory

# Option 1: smaller LLM
ollama pull llama3.2:1b   # ~1GB vs 2GB

# Option 2: smaller Whisper
# Change 'base' to 'tiny' in stt_node parameters

# Option 3: add swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### MPS error on Mac during training
```python
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# Set this BEFORE importing torch in every script
```

### Voice emotion wrong confidence
```python
# Raise threshold in emotion_fusion_node.py
VOICE_CONF_THRESHOLD = 0.75   # was 0.60
```

---

## 23. Project Folder Structure

```
humanoid_vision/
│
├── training/
│   ├── train.py                      # EfficientNet-B2 training
│   ├── dataset.py                    # FER2013/AffectNet data loader
│   └── evaluate.py                   # per-class accuracy report
│
├── export/
│   └── export_onnx.py                # PyTorch to ONNX export
│
├── audio/
│   └── voice_emotion.py              # SpeechBrain wrapper
│
├── scripts/
│   ├── test_mps.py                   # verify Mac GPU
│   └── test_llm_prompts.py           # iterate LLM responses on Mac
│
├── data/
│   ├── fer2013/                      # from Kaggle
│   │   ├── train/
│   │   └── test/
│   └── affectnet/                    # from mohammadmahoormoradi.net
│
├── models/
│   ├── emotion_b2_best.pth           # PyTorch weights — Mac output
│   ├── emotion_b2_final.pth          # final epoch weights
│   ├── emotion_b2.onnx               # ONNX export — Mac
│   ├── emotion_b2_sim.onnx           # simplified ONNX — Mac
│   ├── retinaface.onnx               # pretrained face detector — Mac
│   ├── retinaface_sim.onnx           # simplified — Mac
│   ├── emotion_b2.trt                # TensorRT engine — Nano ONLY
│   ├── retinaface.trt                # TensorRT engine — Nano ONLY
│   └── speechbrain_emotion/          # SpeechBrain cached model
│
├── ros2_ws/
│   └── src/
│       └── emotion_pkg/
│           ├── emotion_pkg/
│           │   ├── __init__.py
│           │   ├── face_detector.py
│           │   ├── emotion_classifier.py
│           │   ├── face_emotion_node.py
│           │   ├── audio_capture_node.py
│           │   ├── stt_node.py
│           │   ├── voice_emotion_node.py
│           │   ├── emotion_fusion_node.py
│           │   ├── llm_response_node.py
│           │   └── tts_node.py
│           ├── launch/
│           │   └── humanoid_emotion.launch.py
│           ├── resource/
│           │   └── emotion_pkg
│           ├── setup.py
│           └── package.xml
│
├── docker/
│   └── Dockerfile                    # ARM64 Ubuntu for Mac testing
│
└── deploy/
    └── setup_nano.sh                 # run once on Nano while online
```

---

## 24. Build Timeline

### Mac Work — Days 1 to 7

| Day | Tasks |
|---|---|
| 1 | Environment setup, dataset download, verify MPS |
| 2 | Train face emotion model — runs overnight if needed |
| 3 | Evaluate model, export ONNX, download RetinaFace, simplify |
| 4 | Set up Ollama on Mac, write and test all LLM prompts exhaustively |
| 5 | Write all 8 ROS2 nodes |
| 6 | Docker testing — verify node logic in ARM64 container |
| 7 | Package everything, transfer to Nano |

### Nano Work — Days 8 to 10

| Day | Tasks |
|---|---|
| 8 | JetPack verify, ROS2 install, Orbbec SDK, Ollama + Llama 3B |
| 9 | TRT conversion, install Python deps, build ROS2 workspace |
| 10 | End-to-end test, offline verification, tune LLM prompts with real interactions |

### Week 3 and Beyond

- Refine LLM prompts based on real interactions
- Handle additional edge cases as discovered
- Add gesture and expression responses
- Improve crying detection thresholds based on testing

---

## Summary: Mac vs Nano Split

| Task | Mac | Nano |
|---|---|---|
| Python environment | Yes | No |
| Dataset download | Yes | No |
| Face emotion training | Yes MPS | No |
| ONNX export and simplification | Yes | No |
| LLM prompt development | Yes Ollama | No |
| All ROS2 node code | Yes | No |
| Docker ARM64 testing | Yes | No |
| JetPack setup | No | Yes once |
| ROS2 Humble install | No | Yes once |
| Orbbec SDK install | No | Yes once |
| Ollama + Llama 3.2 3B | No | Yes once |
| TRT engine conversion | Not possible | Yes once |
| SpeechBrain + Whisper | No | Yes once |
| Full offline runtime | No | Yes always |

---

*This document is the single source of truth for Claude Code to build the complete system.
All file paths, class names, ROS2 topic names, parameter names, and node entry points
defined here are authoritative and must be followed exactly.*
