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
