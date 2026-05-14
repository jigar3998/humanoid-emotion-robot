# face_detector.py
import cv2
import numpy as np


class FaceDetector:
    """
    Face detector using InsightFace buffalo_l (SCRFD det_10g).
    ONNX Runtime on Mac. TensorRT on Nano.
    Models auto-cached to ~/.insightface/models/buffalo_l/
    """
    def __init__(self, model_path=None, use_trt=False):
        self.use_trt = use_trt
        if not use_trt:
            import insightface
            self.app = insightface.app.FaceAnalysis(
                name='buffalo_l',
                allowed_modules=['detection']
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        else:
            self._load_trt(model_path)

    def _load_trt(self, engine_path):
        import tensorrt as trt
        import pycuda.autoinit
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def detect(self, image_bgr, conf_thresh=0.5):
        """Returns list of [x1, y1, x2, y2] bounding boxes."""
        if not self.use_trt:
            faces = self.app.get(image_bgr)
            return [face.bbox.tolist() for face in faces
                    if face.det_score >= conf_thresh]
        else:
            return self._trt_detect(image_bgr, conf_thresh)

    def _trt_detect(self, image_bgr, conf_thresh):
        # TRT SCRFD inference — implement on Nano after trtexec conversion
        raise NotImplementedError("TRT face detection: implement on Nano")

    def align_face(self, image, box, size=260):
        """Crop and resize face with 10% padding. Returns BGR image."""
        x1, y1, x2, y2 = [int(v) for v in box]
        pad = int((x2 - x1) * 0.1)
        x1  = max(0, x1 - pad)
        y1  = max(0, y1 - pad)
        x2  = min(image.shape[1], x2 + pad)
        y2  = min(image.shape[0], y2 + pad)
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return np.zeros((size, size, 3), dtype=np.uint8)
        return cv2.resize(face, (size, size))
