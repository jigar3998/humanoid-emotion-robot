# scripts/download_models.py
# One-time setup: downloads all pretrained models required for Mac dev.
# Run once after cloning the repo.
#
# Downloads:
#   HSEmotions enet_b2_8  → ~/.hsemotion/enet_b2_8.onnx
#   InsightFace buffalo_l → ~/.insightface/models/buffalo_l/

import os
import subprocess
import sys


def download_hsemotions():
    model_path = os.path.expanduser('~/.hsemotion/enet_b2_8.onnx')
    if os.path.isfile(model_path) and os.path.getsize(model_path) > 1_000_000:
        print(f'HSEmotions model already exists ({os.path.getsize(model_path)//1024//1024}MB)')
        return

    os.makedirs(os.path.expanduser('~/.hsemotion'), exist_ok=True)
    print('Downloading HSEmotions enet_b2_8 via git sparse checkout...')

    tmp = '/tmp/hsemotion_dl'
    os.makedirs(tmp, exist_ok=True)
    cmds = [
        f'git -C {tmp} init',
        f'git -C {tmp} remote add origin https://github.com/HSE-asavchenko/face-emotion-recognition.git',
        f'git -C {tmp} sparse-checkout init',
        f'git -C {tmp} sparse-checkout set models/affectnet_emotions/onnx',
        f'git -C {tmp} pull --depth=1 origin main',
    ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True, check=True)

    src = f'{tmp}/models/affectnet_emotions/onnx/enet_b2_8.onnx'
    if os.path.isfile(src):
        import shutil
        shutil.copy(src, model_path)
        print(f'Saved to {model_path} ({os.path.getsize(model_path)//1024//1024}MB)')
    else:
        print('ERROR: enet_b2_8.onnx not found after clone')
        sys.exit(1)


def download_insightface():
    det_path = os.path.expanduser('~/.insightface/models/buffalo_l/det_10g.onnx')
    if os.path.isfile(det_path) and os.path.getsize(det_path) > 1_000_000:
        print(f'InsightFace buffalo_l already exists')
        return

    print('Downloading InsightFace buffalo_l...')
    import insightface
    app = insightface.app.FaceAnalysis(name='buffalo_l', allowed_modules=['detection'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print('InsightFace models downloaded')


if __name__ == '__main__':
    download_hsemotions()
    download_insightface()
    print('\nAll models ready.')
    print('Verify with: python scripts/test_emotion_pipeline.py')
