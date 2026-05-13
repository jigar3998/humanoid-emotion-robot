# audio/voice_emotion.py
import numpy as np
import tempfile
import soundfile as sf
from speechbrain.pretrained import EmotionRecognizer


class VoiceEmotionDetector:
    """
    Detects emotion from audio tone and prosody.
    Uses wav2vec2 pretrained on IEMOCAP dataset.
    Detects: happy, sad, angry, neutral.
    Runs fully offline after first model download.
    """
    def __init__(self, save_dir='models/speechbrain_emotion'):
        print("Loading voice emotion model...")
        self.classifier = EmotionRecognizer.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir=save_dir
        )
        print("Voice emotion model ready")

    def predict_from_array(self, audio_array, sample_rate=16000):
        """
        Predict emotion from float32 numpy audio array at 16kHz mono.
        Returns (emotion_label, confidence_score).
        """
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio_array, sample_rate)
            out_prob, score, index, text_lab = \
                self.classifier.classify_file(f.name)
        return text_lab[0], float(score)

    def extract_audio_features(self, audio_array, sample_rate=16000):
        """
        Extract prosodic features for crying detection.
        Returns pitch variance and energy level.
        """
        import librosa
        f0, _, _ = librosa.pyin(audio_array, fmin=80, fmax=400)
        pitch_variance = float(np.nanstd(f0)) if not np.all(np.isnan(f0)) else 0.0
        energy = float(librosa.feature.rms(y=audio_array).mean())
        return {'pitch_variance': pitch_variance, 'energy': energy}
