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
