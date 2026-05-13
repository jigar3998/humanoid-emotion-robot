# emotion_classifier.py
import numpy as np
import onnxruntime as ort
import cv2

EMOTIONS = ['angry', 'contempt', 'disgust', 'fear',
            'happy', 'neutral', 'sad', 'surprise']
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EmotionClassifier:
    """
    EfficientNet-B2 emotion classifier.
    ONNX Runtime on Mac. TensorRT FP16 on Nano.
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
            size     = trt.volume(self.engine.get_binding_shape(binding))
            dtype    = trt.nptype(self.engine.get_binding_dtype(binding))
            host     = cuda.pagelocked_empty(size, dtype)
            device   = cuda.mem_alloc(host.nbytes)
            bindings.append(int(device))
            entry    = {'host': host, 'device': device}
            (inputs if self.engine.binding_is_input(binding) else outputs).append(entry)
        return inputs, outputs, bindings, stream

    def preprocess(self, face_bgr):
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        face = (face - MEAN) / STD
        return face.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def predict(self, face_bgr):
        inp = self.preprocess(face_bgr)
        if not self.use_trt:
            out = self.session.run(None, {self.input_name: inp})[0][0]
        else:
            out = self._trt_infer(inp)
        exp   = np.exp(out - out.max())
        probs = exp / exp.sum()
        idx   = probs.argmax()
        return EMOTIONS[idx], float(probs[idx])

    def _trt_infer(self, inp):
        import pycuda.driver as cuda
        np.copyto(self.inputs[0]['host'], inp.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'],
                               self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'],
                               self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host'].copy()
