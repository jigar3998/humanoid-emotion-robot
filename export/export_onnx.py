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
