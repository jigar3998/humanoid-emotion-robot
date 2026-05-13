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
