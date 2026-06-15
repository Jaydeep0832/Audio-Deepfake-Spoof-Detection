import os
import io
import numpy as np
import librosa
import soundfile as sf
import torchaudio
import scipy.io.wavfile as wavfile
from pathlib import Path
from typing import Tuple, Dict, Any, Union


class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 4.0,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mfcc: int = 40
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

    def load_audio(self, source: Union[str, Path, bytes, io.BytesIO]) -> Tuple[np.ndarray, int]:
        if isinstance(source, (str, Path)):
            y, sr = librosa.load(str(source), sr=self.sample_rate)
            return y, sr

        raw_bytes = source if isinstance(source, bytes) else source.getvalue()

        # Multi-decoder fallback for browser streams
        try:
            buffer = io.BytesIO(raw_bytes)
            y, sr = librosa.load(buffer, sr=self.sample_rate)
            return y, sr
        except Exception:
            pass

        try:
            buffer = io.BytesIO(raw_bytes)
            data, sr = sf.read(buffer)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != self.sample_rate:
                data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)
            return data.astype(np.float32), self.sample_rate
        except Exception:
            pass

        try:
            buffer = io.BytesIO(raw_bytes)
            waveform, sr = torchaudio.load(buffer)
            y = waveform.mean(dim=0).numpy()
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            return y.astype(np.float32), self.sample_rate
        except Exception:
            pass

        try:
            buffer = io.BytesIO(raw_bytes)
            sr, data = wavfile.read(buffer)
            data = data.astype(np.float32)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if np.max(np.abs(data)) > 1.0:
                data = data / (np.max(np.abs(data)) + 1e-8)
            if sr != self.sample_rate:
                data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)
            return data, self.sample_rate
        except Exception as e:
            raise ValueError(f"Could not decode audio stream: {e}")

    def standardize_length(self, y: np.ndarray) -> np.ndarray:
        if len(y) < self.target_length:
            pad_width = self.target_length - len(y)
            y = np.pad(y, (0, pad_width), mode='constant')
        elif len(y) > self.target_length:
            y = y[:self.target_length]
        return y

    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=20,
            fmax=self.sample_rate / 2
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        norm_mel = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)
        return norm_mel.astype(np.float32)

    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([mfcc, delta1, delta2])

        norm_mfcc = (combined - np.mean(combined)) / (np.std(combined) + 1e-8)

        if norm_mfcc.shape[0] < self.n_mels:
            pad_h = self.n_mels - norm_mfcc.shape[0]
            norm_mfcc = np.pad(norm_mfcc, ((0, pad_h), (0, 0)), mode='constant')

        return norm_mfcc.astype(np.float32)

    def extract_dual_feature_tensor(self, y: np.ndarray) -> np.ndarray:
        mel = self.extract_mel_spectrogram(y)
        mfcc = self.extract_mfcc(y)

        min_t = min(mel.shape[1], mfcc.shape[1])
        mel = mel[:, :min_t]
        mfcc = mfcc[:self.n_mels, :min_t]

        dual_tensor = np.stack([mel, mfcc], axis=0)
        return dual_tensor.astype(np.float32)

    def apply_spec_augment(self, spec: np.ndarray, freq_mask_max: int = 15, time_mask_max: int = 20) -> np.ndarray:
        aug_spec = spec.copy()

        f = np.random.randint(0, freq_mask_max)
        f0 = np.random.randint(0, self.n_mels - f)
        aug_spec[:, f0:f0+f, :] = 0

        t_max = aug_spec.shape[2]
        t = np.random.randint(0, time_mask_max)
        t0 = np.random.randint(0, max(1, t_max - t))
        aug_spec[:, :, t0:t0+t] = 0

        return aug_spec

    def process(self, source: Union[str, Path, bytes], augment: bool = False) -> Dict[str, Any]:
        y, sr = self.load_audio(source)
        y_std = self.standardize_length(y)
        dual_tensor = self.extract_dual_feature_tensor(y_std)

        if augment:
            dual_tensor = self.apply_spec_augment(dual_tensor)

        return {
            "raw_audio": y_std,
            "sample_rate": self.sample_rate,
            "dual_tensor": dual_tensor,
            "mel_spectrogram": dual_tensor[0],
            "mfcc": dual_tensor[1],
            "shape": dual_tensor.shape
        }
