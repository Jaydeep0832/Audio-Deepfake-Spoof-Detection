import os
import tarfile
import urllib.request
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path

# OpenSLR LibriSpeech clean audio corpus URL
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"


def download_librispeech_sample(output_dir="data/temp", max_files=400):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tar_path = Path(output_dir) / "dev-clean.tar.gz"

    if not tar_path.exists():
        print("[*] Downloading LibriSpeech speech corpus (~300MB)...")
        urllib.request.urlretrieve(LIBRISPEECH_URL, tar_path)

    print("[*] Extracting audio files...")
    wav_files = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".flac"):
                f = tar.extractfile(member)
                if f:
                    y, sr = librosa.load(f, sr=16000)
                    wav_files.append(y)
                    if len(wav_files) >= max_files:
                        break
    return wav_files


def synthesize_ai_spoof_samples(bonafide_samples, num_spoofs=400):
    spoof_samples = []

    for i in range(num_spoofs):
        source_y = bonafide_samples[i % len(bonafide_samples)].copy()
        sr = 16000
        category = i % 4

        if category == 0:
            # Neural vocoder pitch shift
            n_steps = np.random.uniform(-4.0, 4.0)
            spoof_y = librosa.effects.pitch_shift(source_y, sr=sr, n_steps=n_steps)
        elif category == 1:
            # Ring modulation and sub-harmonics
            t = np.linspace(0, len(source_y) / sr, len(source_y))
            carrier = np.sin(2 * np.pi * np.random.uniform(50, 150) * t)
            spoof_y = source_y * carrier * 0.7 + source_y * 0.3
        elif category == 2:
            # Tremolo amplitude modulation
            t = np.linspace(0, len(source_y) / sr, len(source_y))
            tremolo = 0.5 + 0.5 * np.sin(2 * np.pi * np.random.uniform(3, 12) * t)
            spoof_y = source_y * tremolo
        else:
            # High-pass spectral filter
            spoof_y = librosa.effects.preemphasis(source_y)
            noise = np.random.normal(0, 0.005, len(spoof_y))
            spoof_y = spoof_y + noise

        spoof_samples.append(spoof_y.astype(np.float32))

    return spoof_samples


def build_asvspoof_dataset():
    audio_dir = Path("data/sample_audio")
    protocol_dir = Path("data/protocols")
    audio_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Fetching human speech clips...")
    bonafide = download_librispeech_sample(max_files=400)
    print(f"[*] Extracting {len(bonafide)} bonafide human voice clips...")

    metadata = []

    for idx, y in enumerate(bonafide):
        file_id = f"bonafide_human_{idx+1:03d}"
        file_path = audio_dir / f"{file_id}.wav"
        sf.write(file_path, y, 16000)
        metadata.append({"audio_file": file_id, "path": str(file_path), "label": "bonafide", "target": 0})

    print("[*] Generating synthetic AI voice spoof samples...")
    spoofs = synthesize_ai_spoof_samples(bonafide, num_spoofs=400)

    for idx, y in enumerate(spoofs):
        file_id = f"spoof_ai_{idx+1:03d}"
        file_path = audio_dir / f"{file_id}.wav"
        sf.write(file_path, y, 16000)
        metadata.append({"audio_file": file_id, "path": str(file_path), "label": "spoof", "target": 1})

    df = pd.DataFrame(metadata).sample(frac=1.0, random_state=42).reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * 0.6)
    n_dev = int(n_total * 0.2)

    df.iloc[:n_train].to_csv(protocol_dir / "train_protocol.csv", index=False)
    df.iloc[n_train:n_train+n_dev].to_csv(protocol_dir / "dev_protocol.csv", index=False)
    df.iloc[n_train+n_dev:].to_csv(protocol_dir / "eval_protocol.csv", index=False)

    print(f"[+] Dataset Ready! Total audio samples: {n_total}")


if __name__ == "__main__":
    build_asvspoof_dataset()
