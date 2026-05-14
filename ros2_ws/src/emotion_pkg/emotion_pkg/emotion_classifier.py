# emotion_classifier.py
import numpy as np
import cv2

EMOTIONS = ['angry', 'contempt', 'disgust', 'fear',
            'happy', 'neutral', 'sad', 'surprise']

# HSEmotions label → our lowercase label
_LABEL_MAP = {
    'Anger':    'angry',
    'Contempt': 'contempt',
    'Disgust':  'disgust',
    'Fear':     'fear',
    'Happiness':'happy',
    'Neutral':  'neutral',
    'Sadness':  'sad',
    'Surprise': 'surprise',
}


class EmotionClassifier:
    """
    EfficientNet-B2 trained on AffectNet (8 emotions) via HSEmotions.
    ONNX Runtime on Mac. TensorRT FP16 on Nano.
    Model: enet_b2_8 — ~80% on real-world faces, 29MB.
    """
    def __init__(self, model_path=None, use_trt=False):
        self.use_trt = use_trt
        if not use_trt:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            self.recognizer = HSEmotionRecognizer(model_name='enet_b2_8')
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
            size   = trt.volume(self.engine.get_binding_shape(binding))
            dtype  = trt.nptype(self.engine.get_binding_dtype(binding))
            host   = cuda.pagelocked_empty(size, dtype)
            device = cuda.mem_alloc(host.nbytes)
            bindings.append(int(device))
            entry  = {'host': host, 'device': device}
            (inputs if self.engine.binding_is_input(binding) else outputs).append(entry)
        return inputs, outputs, bindings, stream

    def predict(self, face_bgr):
        """
        Args:
            face_bgr: cropped face image, any size (resized internally)
        Returns:
            (emotion_str, confidence_float)
        """
        if not self.use_trt:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            emotion_raw, scores = self.recognizer.predict_emotions(face_rgb, logits=False)
            emotion = _LABEL_MAP.get(emotion_raw, emotion_raw.lower())
            confidence = float(np.max(scores))
            return emotion, confidence
        else:
            return self._trt_predict(face_bgr)

    def _trt_predict(self, face_bgr):
        import pycuda.driver as cuda
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(face_rgb, (260, 260)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = ((x - mean) / std).transpose(2, 0, 1)[np.newaxis]
        np.copyto(self.inputs[0]['host'], x.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'],
                               self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'],
                               self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        logits = self.outputs[0]['host'].copy()
        exp    = np.exp(logits - logits.max())
        probs  = exp / exp.sum()
        idx    = probs.argmax()
        return EMOTIONS[idx], float(probs[idx])
