"""Train a raw-scan + clinical/wearable/treatment temporal classifier.

Each manifest row represents one patient visit and one 3D scan. The model encodes
raw image volumes with a 3D CNN, fuses those embeddings with available numeric
clinical/wearable/treatment features and their missingness masks, then applies a
causal Transformer over visits. Labels must be supplied and verified by the user.
This is research starter code, not a diagnostic tool.
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from train_raw_image_model import normalize_volume, read_medical_volume, fit_volume_to_cube

# Prefix convention for easy dataset substitution:
# C_ = clinical, T_ = treatment, M_ = medicine, W_ = wearable.
# Any numeric feature column using one of these prefixes is accepted.
FEATURE_GROUPS = {
    "clinical": ["C_lab1", "C_lab2", "C_vitals", "C_meds_count", "C_comorbidities_count",
                 "C_age", "C_sex_code", "C_bmi"],
    "treatment": ["T_treatment_flag", "T_time_since_treatment_start",
                  "T_treatment_duration_days", "T_treatment_change"],
    "medicine": ["M_medicine_type_code", "M_dose_per_day", "M_frequency_per_day",
                 "M_cumulative_exposure"],
    "wearable": ["W_steps", "W_sleep_hours", "W_heart_rate", "W_hrv", "W_activity"],
}
# Backward compatibility for existing manifests made with the earlier demo template.
LEGACY_FEATURES = [
    "lab1", "lab2", "vitals", "meds", "comorbidities", "age", "sex", "bmi",
    "treatment_flag", "time_since_treatment_start", "treatment_duration_days",
    "treatment_change", "medicine_type", "dose_per_day", "frequency_per_day",
    "cumulative_exposure", "steps", "sleep_hours", "heart_rate", "hrv", "activity",
]
ALL_FEATURES = [c for group in FEATURE_GROUPS.values() for c in group] + LEGACY_FEATURES
GROUP_PREFIXES = ("C_", "T_", "M_", "W_")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_manifest(frame: pd.DataFrame, base_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    required = {"patient_id", "image_path", "stage", "visit_idx"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    if frame.empty:
        raise ValueError("Manifest has no rows.")
    for col in ("patient_id", "image_path", "stage", "visit_idx"):
        if frame[col].isna().any():
            raise ValueError(f"Manifest contains empty values in required column {col!r}.")
    frame["patient_id"] = frame["patient_id"].astype(str).str.strip()
    frame["image_path"] = frame["image_path"].astype(str).str.strip()
    if (frame["patient_id"] == "").any() or (frame["image_path"] == "").any():
        raise ValueError("patient_id and image_path must not be blank.")
    stage = pd.to_numeric(frame["stage"], errors="raise")
    if not np.isfinite(stage.to_numpy(dtype=float)).all() or not np.equal(stage, np.floor(stage)).all():
        raise ValueError("stage labels must be finite integers 0, 1, 2, or 3.")
    frame["stage"] = stage.astype(int)
    if not frame["stage"].isin([0, 1, 2, 3]).all():
        raise ValueError("stage labels must be integer values 0, 1, 2, or 3.")
    visit_values = pd.to_numeric(frame["visit_idx"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(visit_values).all()
            or not np.equal(visit_values, np.floor(visit_values)).all()
            or (visit_values < 0).any()):
        raise ValueError("visit_idx must contain non-negative integers.")
    frame["visit_idx"] = visit_values.astype(np.int64)
    if "time_months" not in frame.columns:
        frame["time_months"] = frame["visit_idx"]
    time_values = pd.to_numeric(frame["time_months"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(time_values).all() or (time_values < 0).any():
        raise ValueError("time_months must contain finite, non-negative values.")
    frame["time_months"] = time_values
    if frame.duplicated(["patient_id", "visit_idx"]).any():
        raise ValueError("Each patient must have at most one scan row per visit_idx.")
    ordered_visits = frame.sort_values(["patient_id", "visit_idx"])
    for patient_id, patient_visits in ordered_visits.groupby("patient_id", sort=False):
        patient_times = patient_visits["time_months"].to_numpy(dtype=float)
        if (np.diff(patient_times) < 0).any():
            raise ValueError(
                f"time_months must not decrease across visits for patient {patient_id!r}."
            )
    feature_cols = [
        col for col in frame.columns
        if col in ALL_FEATURES or col.startswith(GROUP_PREFIXES)
    ]
    if not feature_cols:
        raise ValueError(
            "For multimodal fusion, include at least one supported numeric feature or categorical feature column named with "
            "C_ (clinical), T_ (treatment), M_ (medicine), or W_ (wearable), or use a legacy "
            "column from the earlier template. Use train_raw_image_model.py for image-only training."
        )
    for col in feature_cols:
        if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_string_dtype(frame[col]):
            values = frame[col].astype("string").str.strip()
            frame[col] = values.replace("", pd.NA)
        else:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
            vals = frame[col].dropna().to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                raise ValueError(f"Feature {col!r} contains infinite values.")
    def resolve_path(value):
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (base_dir / path).resolve())
    frame["image_path"] = frame["image_path"].map(resolve_path)
    for path in frame["image_path"]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Image path in manifest does not exist: {path}")
    return frame.sort_values(["patient_id", "visit_idx"]).reset_index(drop=True), feature_cols


def split_by_patient(frame: pd.DataFrame, seed: int = 42):
    ids = np.asarray(sorted(frame["patient_id"].unique()), dtype=object)
    if len(ids) < 10:
        raise ValueError(f"At least 10 distinct patients are required for this default split; found {len(ids)}.")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = max(1, int(len(ids) * 0.70))
    n_valid = max(1, int(len(ids) * 0.15))
    if n_train + n_valid >= len(ids):
        n_train, n_valid = len(ids) - 2, 1
    train_ids = set(ids[:n_train])
    valid_ids = set(ids[n_train:n_train + n_valid])
    test_ids = set(ids[n_train + n_valid:])
    assert train_ids.isdisjoint(valid_ids) and train_ids.isdisjoint(test_ids) and valid_ids.isdisjoint(test_ids)
    return tuple(frame[frame.patient_id.isin(ids_part)].copy() for ids_part in (train_ids, valid_ids, test_ids))


def fit_feature_schema(train_frame: pd.DataFrame, raw_feature_cols: list[str]) -> dict:
    """Learn categorical vocabularies from training patients only."""
    numeric_cols, categorical = [], {}
    for col in raw_feature_cols:
        if pd.api.types.is_object_dtype(train_frame[col]) or pd.api.types.is_string_dtype(train_frame[col]):
            values = train_frame[col].dropna().astype(str).str.strip()
            categorical[col] = sorted(v for v in values.unique().tolist() if v)
        else:
            numeric_cols.append(col)
    return {"numeric_columns": numeric_cols, "categorical_categories": categorical}


def encode_feature_frame(frame: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list[str]]:
    """Expand categorical text to one-hot columns; retain missingness and unseen-category handling."""
    encoded = frame[["patient_id", "image_path", "stage", "visit_idx", "time_months"]].copy()
    feature_cols = []
    for col in schema["numeric_columns"]:
        values = pd.to_numeric(frame[col] if col in frame else pd.Series(np.nan, index=frame.index), errors="raise")
        vals = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError(f"Feature {col!r} contains infinite values.")
        encoded[col] = values
        feature_cols.append(col)
    for col, categories in schema["categorical_categories"].items():
        raw = frame[col].astype("string").str.strip() if col in frame else pd.Series(pd.NA, index=frame.index, dtype="string")
        raw = raw.replace("", pd.NA)
        names = [f"{col}__category_{i}" for i in range(len(categories))]
        # Keep the zero-valued unknown column for checkpoint-schema compatibility.
        # Never activate it: no unknown examples occur in the training vocabulary,
        # so its weight is not learned. Unseen categories therefore map to all zeros.
        unknown_name = f"{col}__unknown"
        columns = names + [unknown_name]
        for name in columns:
            encoded[name] = 0.0
        for row_index, value in raw.items():
            if pd.isna(value):
                for name in columns:
                    encoded.at[row_index, name] = np.nan
            elif str(value) in categories:
                encoded.at[row_index, names[categories.index(str(value))]] = 1.0
            # Unseen values intentionally leave all category columns, including
            # the compatibility unknown column, at zero while remaining observed.
        feature_cols.extend(columns)
    return encoded, feature_cols


def fit_feature_stats(train_frame: pd.DataFrame, feature_cols: list[str]):
    stats = {}
    for col in feature_cols:
        values = train_frame[col].dropna().to_numpy(dtype=np.float64)
        mean = float(values.mean()) if len(values) else 0.0
        std = float(values.std()) if len(values) else 1.0
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def make_sequences(frame: pd.DataFrame, feature_cols: list[str], stats: dict):
    sequences = []
    for patient_id, group in frame.groupby("patient_id", sort=True):
        group = group.sort_values("visit_idx")
        values = group[feature_cols].to_numpy(dtype=np.float32)
        observed = np.isfinite(values)
        scaled = np.zeros_like(values, dtype=np.float32)
        for j, col in enumerate(feature_cols):
            mean, std = stats[col]["mean"], stats[col]["std"]
            scaled[observed[:, j], j] = (values[observed[:, j], j] - mean) / std
        sequences.append({
            "patient_id": str(patient_id),
            "paths": group["image_path"].tolist(),
            "features": scaled,
            "feature_mask": observed.astype(np.float32),
            "time": group["time_months"].to_numpy(dtype=np.float32),
            "labels": group["stage"].to_numpy(dtype=np.int64),
        })
    return sequences


class MultimodalVisitDataset(Dataset):
    def __init__(self, sequences, image_size, modality, ct_low, ct_high, spacing):
        self.sequences = sequences
        self.image_size = image_size
        self.modality = modality
        self.ct_low = ct_low
        self.ct_high = ct_high
        self.spacing = spacing

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        item = self.sequences[index]
        scans = []
        for path in item["paths"]:
            volume = read_medical_volume(path, target_spacing_mm=self.spacing)
            arr = normalize_volume(volume, self.modality, self.ct_low, self.ct_high)
            arr = fit_volume_to_cube(arr, self.image_size)
            scans.append(arr[None, ...])
        return {
            "images": np.stack(scans).astype(np.float32),
            "features": item["features"],
            "feature_mask": item["feature_mask"],
            "time": item["time"],
            "labels": item["labels"],
        }


def collate_sequences(batch):
    max_len = max(len(item["labels"]) for item in batch)
    bsz, image_size = len(batch), batch[0]["images"].shape[-1]
    n_features = batch[0]["features"].shape[-1]
    images = np.zeros((bsz, max_len, 1, image_size, image_size, image_size), dtype=np.float32)
    features = np.zeros((bsz, max_len, n_features), dtype=np.float32)
    feature_mask = np.zeros_like(features)
    times = np.zeros((bsz, max_len), dtype=np.float32)
    labels = np.full((bsz, max_len), -100, dtype=np.int64)
    for i, item in enumerate(batch):
        n = len(item["labels"])
        images[i, :n] = item["images"]
        features[i, :n] = item["features"]
        feature_mask[i, :n] = item["feature_mask"]
        times[i, :n] = item["time"]
        labels[i, :n] = item["labels"]
    return {
        "images": torch.from_numpy(images),
        "features": torch.from_numpy(features),
        "feature_mask": torch.from_numpy(feature_mask),
        "time": torch.from_numpy(times),
        "labels": torch.from_numpy(labels),
        "padding_mask": torch.from_numpy(labels == -100),
    }


class RawImageTemporalFusionModel(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.GroupNorm(4, 8), nn.GELU(), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(2 * n_features, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.image_projection = nn.Linear(64, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.time_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, batch):
        images = batch["images"]
        bsz, length = images.shape[:2]
        valid = ~batch["padding_mask"]
        flat_images = images.reshape(bsz * length, *images.shape[2:])
        valid_flat = valid.reshape(-1)
        image_features = images.new_zeros((bsz * length, 64))
        if valid_flat.any():
            image_features[valid_flat] = self.image_encoder(flat_images[valid_flat])
        image_features = self.image_projection(image_features).reshape(bsz, length, -1)
        features = self.feature_encoder(torch.cat([batch["features"], batch["feature_mask"]], dim=-1))
        fused = self.fusion(torch.cat([image_features, features], dim=-1))
        fused = fused + self.time_encoder((batch["time"] / 12.0).unsqueeze(-1))
        causal_mask = torch.triu(
            torch.ones(length, length, device=fused.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.temporal(
            fused, mask=causal_mask, src_key_padding_mask=batch["padding_mask"]
        )
        return self.classifier(hidden)


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def evaluate(model, loader, device):
    model.eval()
    ys, preds, losses = [], [], []
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch)
            loss = criterion(logits.reshape(-1, 4), batch["labels"].reshape(-1))
            losses.append(float(loss.item()))
            valid = batch["labels"] != -100
            ys.extend(batch["labels"][valid].cpu().tolist())
            preds.extend(logits.argmax(-1)[valid].cpu().tolist())
    if not ys:
        raise ValueError("Evaluation split contains no labelled visits.")
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(ys, preds)),
        "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
        "n_visits": len(ys),
        "classification_report": classification_report(
            ys, preds, labels=[0, 1, 2, 3],
            target_names=["stage_0", "stage_1", "stage_2", "stage_3"],
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(ys, preds, labels=[0, 1, 2, 3]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="multimodal_raw_manifest.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2, help="Number of patients per batch.")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--target-spacing-mm", type=float, default=1.5)
    parser.add_argument("--modality", choices=["generic", "ct", "mri"], default="generic")
    parser.add_argument("--ct-window-low", type=float, default=-1000.0)
    parser.add_argument("--ct-window-high", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="multimodal_raw_results")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 16:
        raise ValueError("epochs/batch-size must be positive and image-size must be >=16.")
    if not np.isfinite(args.target_spacing_mm) or args.target_spacing_mm <= 0:
        raise ValueError("target-spacing-mm must be a finite positive number.")
    if args.modality == "ct" and (
        not np.isfinite(args.ct_window_low) or not np.isfinite(args.ct_window_high)
        or args.ct_window_high <= args.ct_window_low
    ):
        raise ValueError("CT window bounds must be finite with high > low.")
    set_seed(args.seed)
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    frame, raw_feature_cols = validate_manifest(pd.read_csv(manifest_path), manifest_path.parent)
    train_df, valid_df, test_df = split_by_patient(frame, args.seed)
    feature_schema = fit_feature_schema(train_df, raw_feature_cols)
    encoded_frame, feature_cols = encode_feature_frame(frame, feature_schema)
    train_ids, valid_ids, test_ids = (
        set(train_df.patient_id), set(valid_df.patient_id), set(test_df.patient_id)
    )
    encoded_train = encoded_frame[encoded_frame.patient_id.isin(train_ids)].copy()
    encoded_valid = encoded_frame[encoded_frame.patient_id.isin(valid_ids)].copy()
    encoded_test = encoded_frame[encoded_frame.patient_id.isin(test_ids)].copy()
    stats = fit_feature_stats(encoded_train, feature_cols)
    sequences = make_sequences(encoded_frame, feature_cols, stats)
    train_seq = [s for s in sequences if s["patient_id"] in train_ids]
    valid_seq = [s for s in sequences if s["patient_id"] in valid_ids]
    test_seq = [s for s in sequences if s["patient_id"] in test_ids]
    dataset_args = (args.image_size, args.modality, args.ct_window_low,
                    args.ct_window_high, args.target_spacing_mm)
    loaders = [
        DataLoader(MultimodalVisitDataset(seq, *dataset_args), batch_size=args.batch_size,
                   shuffle=(i == 0), num_workers=0, collate_fn=collate_sequences)
        for i, seq in enumerate((train_seq, valid_seq, test_seq))
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawImageTemporalFusionModel(len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_loss, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in loaders[0]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits.reshape(-1, 4), batch["labels"].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        valid_metrics = evaluate(model, loaders[1], device)
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(train_losses)),
            "validation_loss": valid_metrics["loss"],
            "validation_accuracy": valid_metrics["accuracy"],
            "validation_macro_f1": valid_metrics["macro_f1"],
        })
        print(f"Epoch {epoch:02d}: train_loss={np.mean(train_losses):.4f}, "
              f"validation_loss={valid_metrics['loss']:.4f}, "
              f"validation_macro_f1={valid_metrics['macro_f1']:.4f}")
        if valid_metrics["loss"] < best_loss:
            best_loss, stale = valid_metrics["loss"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4:
                print("Early stopping after 4 epochs without validation-loss improvement.")
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders[2], device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "feature_stats": stats,
        "feature_schema": feature_schema,
        "image_size": args.image_size,
        "target_spacing_mm": args.target_spacing_mm,
        "modality": args.modality,
        "ct_window_low": args.ct_window_low if args.modality == "ct" else None,
        "ct_window_high": args.ct_window_high if args.modality == "ct" else None,
        "architecture": "RawImageTemporalFusionModel",
    }, out / "multimodal_raw_model.pt")
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    summary = {
        "task": "causal longitudinal stage classification with raw scan + tabular fusion",
        "n_scans": int(len(frame)), "n_patients": int(frame.patient_id.nunique()),
        "n_feature_columns": len(feature_cols), "feature_columns": feature_cols,
        "split_patients": {
            "train": int(train_df.patient_id.nunique()),
            "validation": int(valid_df.patient_id.nunique()),
            "test": int(test_df.patient_id.nunique()),
        },
        "preprocessing": {
            "modality": args.modality, "image_size_voxels": args.image_size,
            "target_spacing_mm": args.target_spacing_mm, "orientation": "LPS",
            "cube_policy": "center crop/pad after normalization",
            "ct_window_low": args.ct_window_low if args.modality == "ct" else None,
            "ct_window_high": args.ct_window_high if args.modality == "ct" else None,
        },
        "test_metrics": {k: v for k, v in test_metrics.items()
                         if k not in {"classification_report", "confusion_matrix"}},
        "classification_report": test_metrics["classification_report"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "limitations": [
            "Research starter only; not a diagnostic or clinical tool.",
            "Requires verified user-supplied labels and an authorised dataset.",
            "Metrics are only for this split and do not establish clinical performance or external validity.",
            "Target spacing and center crop/padding must be checked for anatomy and field of view.",
            "Uses supplied visit-stage labels to classify stages at visits; it is not a validated future-progression forecast.",
            "Feature normalization statistics are fit on training patients only.",
        ],
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Held-out test metrics:")
    print(json.dumps(summary["test_metrics"], indent=2))
    print(f"Saved model and results under: {out.resolve()}")


if __name__ == "__main__":
    main()
