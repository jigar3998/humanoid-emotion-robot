# scripts/test_mps.py
import torch
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

x = torch.randn(1000, 1000).to(device)
y = torch.matmul(x, x)
print("MPS compute: OK" if device.type == "mps" else "CPU only — training will be slow")
