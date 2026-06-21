import os
import sys
import joblib
import numpy as np
import torch
import lightgbm as lgb
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.audio_processor import AudioProcessor
    from src.models import ResNetSpoofDetector
    from src.dataset import create_dataloader
except ImportError:
    from audio_processor import AudioProcessor
    from models import ResNetSpoofDetector
    from dataset import create_dataloader


class ResNetLightGBMEnsemble:
    def __init__(
        self,
        resnet_model: Optional[ResNetSpoofDetector] = None,
        lgb_model: Optional[lgb.LGBMClassifier] = None,
        processor: Optional[AudioProcessor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.processor = processor if processor is not None else AudioProcessor()
        self.resnet = resnet_model if resnet_model is not None else ResNetSpoofDetector(in_channels=2)
        self.resnet.to(self.device)
        self.resnet.eval()
        self.lgb_model = lgb_model

    def extract_embeddings_from_dataloader(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        self.resnet.eval()
        embeddings_list, labels_list = [], []

        with torch.no_grad():
            for inputs, targets, _ in dataloader:
                inputs = inputs.to(self.device)
                feats = self.resnet.extract_features(inputs)
                embeddings_list.append(feats.cpu().numpy())
                labels_list.append(targets.numpy())

        X = np.vstack(embeddings_list)
        y = np.concatenate(labels_list)
        return X, y

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_save_path: str = "models/lightgbm_ensemble.pkl"
    ) -> lgb.LGBMClassifier:
        print(f"[*] Training LightGBM ensemble on {X_train.shape[0]} embeddings...")

        lgbm = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            random_state=42,
            verbose=-1
        )

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        lgbm.fit(X_train, y_train, eval_set=eval_set)
        self.lgb_model = lgbm

        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(lgbm, model_save_path)
        print(f"[+] Saved LightGBM model to: {model_save_path}")

        return lgbm

    def predict_audio(self, source: Union[str, Path, bytes]) -> Dict[str, Any]:
        processed = self.processor.process(source)
        dual_tensor = processed["dual_tensor"]

        input_tensor = torch.from_numpy(dual_tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            resnet_logits = self.resnet(input_tensor)
            resnet_probs = torch.softmax(resnet_logits, dim=1).cpu().numpy()[0]
            resnet_spoof_prob = float(resnet_probs[1])
            embedding = self.resnet.extract_features(input_tensor).cpu().numpy()

        if self.lgb_model is not None:
            lgb_probs = self.lgb_model.predict_proba(embedding)[0]
            spoof_prob = float(lgb_probs[1])
            method = "ResNet-18 + LightGBM Ensemble"
        else:
            spoof_prob = resnet_spoof_prob
            method = "ResNet-18 CNN Base"

        is_spoof = bool(spoof_prob >= 0.5)
        confidence = float(abs(spoof_prob - 0.5) * 2.0 * 100.0)

        return {
            "is_spoof": is_spoof,
            "label": "SPOOF (AI Voice)" if is_spoof else "BONAFIDE (Human)",
            "spoof_probability": round(spoof_prob, 4),
            "human_probability": round(1.0 - spoof_prob, 4),
            "confidence_percentage": round(confidence, 2),
            "method": method,
            "mel_spectrogram": dual_tensor[0]
        }

    def save_pipeline(self, resnet_path: str = "models/resnet_spoof_detector.pth", lgb_path: str = "models/lightgbm_ensemble.pkl"):
        Path(resnet_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.resnet.state_dict(), resnet_path)
        if self.lgb_model is not None:
            joblib.dump(self.lgb_model, lgb_path)

    def load_pipeline(self, resnet_path: str = "models/resnet_spoof_detector.pth", lgb_path: str = "models/lightgbm_ensemble.pkl"):
        if Path(resnet_path).exists():
            self.resnet.load_state_dict(torch.load(resnet_path, map_location=self.device))
            self.resnet.eval()
            print(f"[+] Loaded ResNet weights from {resnet_path}")
        if Path(lgb_path).exists():
            self.lgb_model = joblib.load(lgb_path)
            print(f"[+] Loaded LightGBM ensemble model from {lgb_path}")


def train_ensemble_pipeline(
    train_protocol: str = "data/protocols/train_protocol.csv",
    val_protocol: str = "data/protocols/dev_protocol.csv",
    resnet_weights: str = "models/resnet_spoof_detector.pth",
    lgb_save_path: str = "models/lightgbm_ensemble.pkl"
):
    ensemble = ResNetLightGBMEnsemble()
    ensemble.load_pipeline(resnet_path=resnet_weights)

    train_loader = create_dataloader(train_protocol, batch_size=16, shuffle=False)
    val_loader = create_dataloader(val_protocol, batch_size=16, shuffle=False) if Path(val_protocol).exists() else None

    print("[*] Extracting embeddings for Training set...")
    X_train, y_train = ensemble.extract_embeddings_from_dataloader(train_loader)

    X_val, y_val = None, None
    if val_loader is not None:
        print("[*] Extracting embeddings for Validation set...")
        X_val, y_val = ensemble.extract_embeddings_from_dataloader(val_loader)

    ensemble.train_lightgbm(X_train, y_train, X_val, y_val, model_save_path=lgb_save_path)
    print("\n[+] Ensemble Model Trained Successfully.\n")


if __name__ == "__main__":
    train_ensemble_pipeline()
