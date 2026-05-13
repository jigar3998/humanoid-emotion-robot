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
