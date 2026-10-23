"""Train a 3D CNN directly from CT/MRI volumes for research experiments.

Supported input: NIfTI files (.nii/.nii.gz) and DICOM series directories.
Manifest columns: patient_id, image_path, stage; optional visit_idx.
Labels must be supplied by the dataset owner and use integer stages 0-3.
This script is research code, not a diagnostic tool.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Clip each scan to its own robust intensity percentiles and standardize.

    This is a generic starting point only. Modality-specific preprocessing
    (especially CT Hounsfield-unit windows) may be needed for a real study.
    """
    arr = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("Image volume contains no finite voxel values.")
    values = arr[finite]
    lo, hi = np.percentile(values, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip(np.nan_to_num(arr, nan=float(np.median(values)), posinf=hi, neginf=lo), lo, hi)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-6:
        std = 1.0
    return ((arr - mean) / std).astype(np.float32)


def volume_to_tensor(volume: np.ndarray, image_size: int = 64) -> torch.Tensor:
    """Convert a Z,Y,X array into a normalized [1,Z,Y,X] tensor of fixed size."""
    arr = normalize_volume(volume)
    tensor = torch.from_numpy(arr)[None, None, ...]
    tensor = F.interpolate(tensor, size=(image_size, image_size, image_size),
                           mode="trilinear", align_corners=False)
    return tensor[0].contiguous()


def read_medical_volume(path: str) -> np.ndarray:
    """Read a NIfTI file or a directory containing one DICOM series via SimpleITK."""
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "SimpleITK is required to read medical images. Install requirements.txt "
            "or run: python -m pip install SimpleITK"
        ) from exc

    image_path = Path(path).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    if image_path.is_dir():
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(image_path))
        if not series_ids:
            raise ValueError(f"No DICOM series found in directory: {image_path}")
        if len(series_ids) > 1:
            raise ValueError(
                f"Directory contains {len(series_ids)} DICOM series: {image_path}. "
                "Put one series in each directory and reference the desired series."
            )
        filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(image_path), series_ids[0])
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(filenames)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(image_path))
    volume = sitk.GetArrayFromImage(image)  # SimpleITK array order: Z,Y,X for 3D
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D scan; got array shape {volume.shape} from {image_path}")
    return volume


class RawVolumeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int = 64):
        self.frame = frame.reset_index(drop=True).copy()
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        volume = read_medical_volume(str(row["image_path"]))
        image = volume_to_tensor(volume, self.image_size)
        return image, int(row["stage"])


class Small3DCNN(nn.Module):
    """Compact 3D CNN baseline for direct volume-to-stage classification."""
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1), nn.BatchNorm3d(8), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def split_by_patient(frame: pd.DataFrame, seed: int = 42):
    patient_ids = np.array(sorted(frame["patient_id"].astype(str).unique()))
    if len(patient_ids) < 10:
        raise ValueError(
            f"Need at least 10 distinct patients for this default split; found {len(patient_ids)}. "
            "Use a carefully designed split appropriate to the dataset rather than splitting scans randomly."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    n_train = max(1, int(n * 0.70))
    n_valid = max(1, int(n * 0.15))
    if n_train + n_valid >= n:
        n_train, n_valid = n - 2, 1
    train_ids = set(patient_ids[:n_train])
    valid_ids = set(patient_ids[n_train:n_train + n_valid])
    test_ids = set(patient_ids[n_train + n_valid:])
    assert train_ids.isdisjoint(valid_ids) and train_ids.isdisjoint(test_ids) and valid_ids.isdisjoint(test_ids)
    return (
        frame[frame["patient_id"].astype(str).isin(train_ids)].copy(),
        frame[frame["patient_id"].astype(str).isin(valid_ids)].copy(),
        frame[frame["patient_id"].astype(str).isin(test_ids)].copy(),
    )


def validate_manifest(frame: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    required = {"patient_id", "image_path", "stage"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    if frame[list(required)].isna().any().any():
        raise ValueError("Manifest contains empty patient_id, image_path, or stage values.")
    frame["patient_id"] = frame["patient_id"].astype(str)
    numeric_stage = pd.to_numeric(frame["stage"], errors="raise")
    if not np.all(numeric_stage == numeric_stage.astype(int)):
        raise ValueError("stage labels must be integer values 0, 1, 2, or 3.")
    frame["stage"] = numeric_stage.astype(int)
    if not frame["stage"].isin([0, 1, 2, 3]).all():
        raise ValueError("stage labels must be integer values 0, 1, 2, or 3.")
    def resolve_path(value):
        path = Path(str(value)).expanduser()
        return str(path if path.is_absolute() else (base_dir / path).resolve())
    frame["image_path"] = frame["image_path"].map(resolve_path)
    if "visit_idx" in frame.columns:
        frame = frame.sort_values(["patient_id", "visit_idx"]).reset_index(drop=True)
    if frame.duplicated(["patient_id", "image_path"]).any():
        raise ValueError("Duplicate patient_id/image_path pairs found in manifest.")
    return frame


def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(dim=1)
            ys.extend(labels.cpu().tolist())
            preds.extend(pred.cpu().tolist())
    return {
        "accuracy": float(accuracy_score(ys, preds)),
        "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
        "classification_report": classification_report(
            ys, preds, labels=[0, 1, 2, 3],
            target_names=["stage_0", "stage_1", "stage_2", "stage_3"],
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(ys, preds, labels=[0, 1, 2, 3]).tolist(),
        "n_scans": len(ys),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="raw_images_manifest.csv", help="CSV with patient_id,image_path,stage")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="raw_image_results")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 16:
        raise ValueError("epochs/batch-size must be positive and image-size must be >=16.")
    set_seed(args.seed)
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Copy raw_images_manifest_template.csv "
            "to raw_images_manifest.csv and fill it with your authorised dataset paths and labels."
        )
    frame = validate_manifest(pd.read_csv(manifest_path), manifest_path.parent)
    for path in frame["image_path"]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Image path in manifest does not exist: {path}")
    train_df, valid_df, test_df = split_by_patient(frame, args.seed)
    if min(len(train_df), len(valid_df), len(test_df)) < 1:
        raise ValueError("Patient-level split produced an empty partition; review dataset size.")
    print(f"Scans: total={len(frame)}, train={len(train_df)}, validation={len(valid_df)}, test={len(test_df)}")
    print(f"Distinct patients: total={frame.patient_id.nunique()}, train={train_df.patient_id.nunique()}, "
          f"validation={valid_df.patient_id.nunique()}, test={test_df.patient_id.nunique()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader = DataLoader(RawVolumeDataset(train_df, args.image_size), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    valid_loader = DataLoader(RawVolumeDataset(valid_df, args.image_size), batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    test_loader = DataLoader(RawVolumeDataset(test_df, args.image_size), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    model = Small3DCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_loss, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                valid_losses.append(float(criterion(model(images), labels).item()))
        val_loss = float(np.mean(valid_losses))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss})
        print(f"Epoch {epoch:02d}: train_loss={np.mean(losses):.4f}, validation_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4:
                print("Early stopping after 4 epochs without validation-loss improvement.")
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    metrics = evaluate(model, test_loader, device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "raw_image_3dcnn.pt")
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    summary = {
        "task": "supervised 3D scan classification",
        "input": "raw 3D medical volumes (NIfTI files or one-series-per-directory DICOM)",
        "n_scans": len(frame), "n_patients": int(frame.patient_id.nunique()),
        "split_patients": {
            "train": int(train_df.patient_id.nunique()),
            "validation": int(valid_df.patient_id.nunique()),
            "test": int(test_df.patient_id.nunique()),
        },
        "test_metrics": {k: v for k, v in metrics.items() if k not in {"classification_report", "confusion_matrix"}},
        "classification_report": metrics["classification_report"],
        "confusion_matrix": metrics["confusion_matrix"],
        "limitations": [
            "This is research starter code, not a clinical tool.",
            "All labels must be supplied and verified by the dataset owner; this script does not create labels.",
            "A simple per-volume intensity normalization is used. CT-specific Hounsfield-unit windows and modality-specific preprocessing may be required.",
            "Resizing 3D scans to a fixed cube may discard spatial detail and is not a validated medical preprocessing protocol.",
            "Results are meaningful only if dataset labels, acquisition protocols, and patient-level splits are appropriate.",
            "This standalone raw-image CNN currently uses images and labels only; it is not yet fused with the existing clinical/wearable/treatment temporal model."
        ],
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nHeld-out test metrics (only for the supplied labelled dataset):")
    print(json.dumps(summary["test_metrics"], indent=2))
    print(f"Saved model and results under: {out.resolve()}")
    print("Do not interpret these metrics as clinical validation.")


if __name__ == "__main__":
    main()
