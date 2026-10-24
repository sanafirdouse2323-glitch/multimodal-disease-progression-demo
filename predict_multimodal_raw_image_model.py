"""Predict per-visit stage probabilities with a trained raw-image fusion checkpoint.

Input CSV requires patient_id, image_path, visit_idx and the tabular feature columns
used during training (missing training feature columns are treated as unavailable).
This script is for research evaluation, not clinical decision-making.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_multimodal_raw_image_model import (
    MultimodalVisitDataset,
    RawImageTemporalFusionModel,
    collate_sequences,
    make_sequences,
)


def prepare_prediction_manifest(frame: pd.DataFrame, base_dir: Path, checkpoint: dict) -> pd.DataFrame:
    required = {"patient_id", "image_path", "visit_idx"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction manifest is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    if frame.empty:
        raise ValueError("Prediction manifest has no rows.")
    for col in required:
        if frame[col].isna().any():
            raise ValueError(f"Prediction manifest contains empty values in {col!r}.")
    frame["patient_id"] = frame["patient_id"].astype(str).str.strip()
    frame["image_path"] = frame["image_path"].astype(str).str.strip()
    if (frame["patient_id"] == "").any() or (frame["image_path"] == "").any():
        raise ValueError("patient_id and image_path must not be blank.")
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
    schema = checkpoint.get("feature_schema")
    if schema:
        raw_feature_cols = list(schema["numeric_columns"]) + list(schema["categorical_categories"].keys())
        for col in raw_feature_cols:
            if col not in frame.columns:
                frame[col] = np.nan
        for col in schema["numeric_columns"]:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
            vals = frame[col].dropna().to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                raise ValueError(f"Feature {col!r} contains infinite values.")
        for col in schema["categorical_categories"]:
            frame[col] = frame[col].astype("string").str.strip().replace("", pd.NA)
    else:
        # Backward compatibility with checkpoints trained before categorical encoding.
        feature_cols = checkpoint["feature_columns"]
        for col in feature_cols:
            if col not in frame.columns:
                frame[col] = np.nan
            frame[col] = pd.to_numeric(frame[col], errors="raise")
            vals = frame[col].dropna().to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                raise ValueError(f"Feature {col!r} contains infinite values.")
    def resolve_path(value):
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (base_dir / path).resolve())
    frame["image_path"] = frame["image_path"].map(resolve_path)
    patient_owners = frame.groupby("image_path")["patient_id"].nunique()
    if (patient_owners > 1).any():
        raise ValueError("An image_path cannot be assigned to multiple patient_id values.")
    for path in frame["image_path"]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Image path in prediction manifest does not exist: {path}")
    # Dummy labels are used only to reuse the sequence batcher; predictions ignore them.
    frame["stage"] = 0
    frame = frame.sort_values(["patient_id", "visit_idx"]).reset_index(drop=True)
    if schema:
        from train_multimodal_raw_image_model import encode_feature_frame
        frame, _ = encode_feature_frame(frame, schema)
    return frame


def validate_checkpoint(checkpoint: object) -> dict:
    """Check architecture, feature schema and image preprocessing before inference."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary produced by the multimodal trainer.")
    if checkpoint.get("architecture") != "RawImageTemporalFusionModel":
        raise ValueError("Checkpoint architecture is not RawImageTemporalFusionModel.")
    artifact_type = checkpoint.get("artifact_type")
    if artifact_type not in (None, "multimodal-disease-progression.multimodal-raw-image"):
        raise ValueError(f"Unsupported multimodal checkpoint artifact type: {artifact_type!r}.")
    artifact_version = checkpoint.get("artifact_version")
    if artifact_version not in (None, 1):
        raise ValueError(f"Unsupported multimodal checkpoint artifact version: {artifact_version!r}.")
    try:
        n_features = int(checkpoint["n_features"])
        image_size = int(checkpoint["image_size"])
        spacing = float(checkpoint["target_spacing_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint is missing valid model or preprocessing dimensions.") from exc
    feature_columns = checkpoint.get("feature_columns")
    feature_stats = checkpoint.get("feature_stats")
    if n_features < 1 or not isinstance(feature_columns, list) or len(feature_columns) != n_features:
        raise ValueError("Checkpoint feature_columns must match a positive n_features value.")
    if not isinstance(feature_stats, dict) or any(col not in feature_stats for col in feature_columns):
        raise ValueError("Checkpoint feature_stats must include every model feature column.")
    if image_size < 16:
        raise ValueError("Checkpoint image_size must be at least 16 voxels.")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Checkpoint target_spacing_mm must be finite and positive.")
    if checkpoint.get("modality") not in {"generic", "ct", "mri"}:
        raise ValueError("Checkpoint modality must be one of: generic, ct, mri.")
    if checkpoint["modality"] == "ct":
        low, high = checkpoint.get("ct_window_low"), checkpoint.get("ct_window_high")
        if low is None or high is None or not np.isfinite(float(low)) or not np.isfinite(float(high)) or float(high) <= float(low):
            raise ValueError("CT checkpoint must contain finite window bounds with high > low.")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Checkpoint model_state_dict is missing.")
    model_config = checkpoint.get("model_config")
    if model_config is not None and (
        not isinstance(model_config, dict)
        or int(model_config.get("n_features", -1)) != n_features
        or int(model_config.get("n_classes", -1)) != 4
    ):
        raise ValueError("Checkpoint model_config does not match the expected four-class architecture.")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="multimodal_raw_model.pt from training")
    parser.add_argument("--manifest", required=True, help="CSV with patient_id,image_path,visit_idx and feature columns")
    parser.add_argument("--output", default="raw_image_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of patients per batch")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prediction manifest not found: {manifest_path}")
    output_path = Path(args.output).expanduser().resolve()
    if output_path in {checkpoint_path, manifest_path}:
        raise ValueError("Output path must not overwrite the checkpoint or input manifest.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load checkpoint {checkpoint_path}. Use a trusted checkpoint produced "
            "by the multimodal training script."
        ) from exc
    checkpoint = validate_checkpoint(checkpoint)
    frame = prepare_prediction_manifest(pd.read_csv(manifest_path), manifest_path.parent, checkpoint)
    feature_cols = checkpoint["feature_columns"]
    sequences = make_sequences(frame, feature_cols, checkpoint["feature_stats"])
    dataset = MultimodalVisitDataset(
        sequences,
        int(checkpoint["image_size"]),
        checkpoint["modality"],
        float(checkpoint["ct_window_low"] if checkpoint["ct_window_low"] is not None else -1000.0),
        float(checkpoint["ct_window_high"] if checkpoint["ct_window_high"] is not None else 1000.0),
        float(checkpoint["target_spacing_mm"]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_sequences)
    model = RawImageTemporalFusionModel(n_features=int(checkpoint["n_features"])).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Checkpoint weights do not match the selected RawImageTemporalFusionModel architecture."
        ) from exc
    model.eval()
    patient_groups = {
        str(pid): group.sort_values("visit_idx")
        for pid, group in frame.groupby("patient_id", sort=True)
    }
    rows = []
    sequence_offset = 0
    with torch.no_grad():
        for batch in loader:
            moved = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits = model(moved)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for batch_index in range(len(batch["labels"])):
                sequence = sequences[sequence_offset]
                patient_id = sequence["patient_id"]
                group = patient_groups[patient_id]
                for visit_position, (_, source_row) in enumerate(group.iterrows()):
                    p = probs[batch_index, visit_position]
                    rows.append({
                        "patient_id": patient_id,
                        "visit_idx": source_row["visit_idx"],
                        "time_months": source_row["time_months"],
                        "predicted_stage": int(np.argmax(p)),
                        "confidence": float(np.max(p)),
                        "prob_stage_0": float(p[0]),
                        "prob_stage_1": float(p[1]),
                        "prob_stage_2": float(p[2]),
                        "prob_stage_3": float(p[3]),
                    })
                sequence_offset += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Predicted {len(rows)} visits across {frame.patient_id.nunique()} patients.")
    print(f"Saved predictions to: {output_path.resolve()}")
    print("Predictions are research outputs, not clinical advice or validated diagnoses.")


if __name__ == "__main__":
    main()
