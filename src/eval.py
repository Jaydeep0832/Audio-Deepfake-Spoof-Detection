import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.ensemble import ResNetLightGBMEnsemble
    from src.dataset import create_dataloader
    from src.metrics import compute_eer, compute_tdcf
except ImportError:
    from ensemble import ResNetLightGBMEnsemble
    from dataset import create_dataloader
    from metrics import compute_eer, compute_tdcf


def evaluate_system(
    eval_protocol: str = "data/protocols/eval_protocol.csv",
    resnet_weights: str = "models/resnet_spoof_detector.pth",
    lgb_weights: str = "models/lightgbm_ensemble.pkl",
    report_dir: str = "reports"
):
    print(f"[*] Starting system evaluation on protocol: {eval_protocol}")
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    ensemble = ResNetLightGBMEnsemble()
    ensemble.load_pipeline(resnet_path=resnet_weights, lgb_path=lgb_weights)

    eval_loader = create_dataloader(eval_protocol, batch_size=16, shuffle=False)

    y_true, y_scores, y_preds = [], [], []

    for inputs, targets, _ in eval_loader:
        inputs = inputs.to(ensemble.device)
        embeddings = ensemble.resnet.extract_features(inputs).cpu().numpy()

        if ensemble.lgb_model is not None:
            probs = ensemble.lgb_model.predict_proba(embeddings)[:, 1]
            preds = (probs >= 0.5).astype(int)
        else:
            logits = ensemble.resnet(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)

        y_true.extend(targets.numpy())
        y_scores.extend(probs)
        y_preds.extend(preds)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_preds = np.array(y_preds)

    eer, opt_threshold = compute_eer(y_scores, y_true)
    tdcf = compute_tdcf(y_scores, y_true)
    acc = accuracy_score(y_true, y_preds) * 100.0
    cm = confusion_matrix(y_true, y_preds)

    print("\n" + "=" * 60)
    print("ASVSPOOF SYSTEM EVALUATION RESULTS")
    print("=" * 60)
    print(f" Equal Error Rate (EER):        {eer:.2f}%")
    print(f" Optimal EER Threshold:        {opt_threshold:.4f}")
    print(f" Tandem DCF (t-DCF):            {tdcf:.4f}")
    print(f" Classification Accuracy:      {acc:.2f}%")
    print(f" Confusion Matrix (TN, FP, FN, TP):\n{cm}")
    print("=" * 60 + "\n")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ASVspoof Deepfake Detection ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    save_path = Path(report_dir) / "roc_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved ROC curve plot to: {save_path}")


if __name__ == "__main__":
    evaluate_system()
