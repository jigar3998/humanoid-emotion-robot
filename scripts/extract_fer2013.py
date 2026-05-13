# scripts/extract_fer2013.py
# Converts fer2013.csv to folder structure:
#   data/fer2013/train/{emotion}/*.png
#   data/fer2013/test/{emotion}/*.png
#
# FER2013 emotion labels:
#   0=angry  1=disgust  2=fear  3=happy  4=sad  5=surprise  6=neutral

import os
import csv
import numpy as np
from PIL import Image

CSV_PATH  = '../data/fer2013/fer2013.csv'
OUT_DIR   = '../data/fer2013'

EMOTION_MAP = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral',
}

SPLIT_MAP = {
    'Training':    'train',
    'PublicTest':  'test',
    'PrivateTest': 'test',
}

# Create dirs
for split in ('train', 'test'):
    for emotion in EMOTION_MAP.values():
        os.makedirs(os.path.join(OUT_DIR, split, emotion), exist_ok=True)

counts = {}
with open(CSV_PATH, newline='') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        emotion_idx = int(row['emotion'])
        pixels      = list(map(int, row['pixels'].split()))
        usage       = row['Usage']

        emotion = EMOTION_MAP[emotion_idx]
        split   = SPLIT_MAP.get(usage, 'test')

        img = Image.fromarray(
            np.array(pixels, dtype=np.uint8).reshape(48, 48)
        ).convert('RGB')

        fname = f'{i:06d}.png'
        img.save(os.path.join(OUT_DIR, split, emotion, fname))

        key = (split, emotion)
        counts[key] = counts.get(key, 0) + 1

        if i % 5000 == 0:
            print(f'  {i} rows processed...')

print('\nExtraction complete:')
for (split, emotion), count in sorted(counts.items()):
    print(f'  {split}/{emotion}: {count}')
