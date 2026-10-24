"""Run per-scan inference with a versioned image-only 3D CNN checkpoint.

The checkpoint contains the model architecture and the preprocessing parameters used
at training time. This tool reports model scores only; it is not a diagnostic tool.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_raw_image_model import Small3DCNN, read_medical_volume, volume_to_tensor

ARTIFACT_TYPE = "multimodal-disease-progression.image-only-3dcnn"
SUPPORTED_ARTIFACT_VERSION = 1


def validate_checkpoint(checkpoint: object) -> tuple[dict, dict]:
    """Validate the portable artifact contract before allocating the model."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary artifact; retrain with the current trainer.")
    if checkpoint.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(
            "Unsupported image-only checkpoint. Expected an artifact produced by the current "
            "train_raw_image_model.py version; older state_dict-only checkpoints lack preprocessing metadata."
        )
    if checkpoint.get("artifact_version") != SUPPORTED_ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint artifact_version={checkpoint.get('artifact_version')!r}; "
            f"supported version is {SUPPORTED_ARTIFACT_VERSION}."
        )
    if checkpoint.get("architecture") != "Small3DCNN":
        raise ValueError("Checkpoint architecture must be 'Small3DCNN'.")
    model_config = checkpoint.get("model_config")
    preprocessing = checkpoint.get("preprocessing")
    if not isinstance(model_config, dict) or model_config.get("n_classes") != 4:
        raise ValueError("Checkpoint model_config must specify n_classes=4.")
    if not isinstance(preprocessing, dict):
        raise ValueError("Checkpoint is missing preprocessing metadata.")
    required = {"modality", "image_size_voxels", "target_spacing_mm",
                "ct_window_low", "ct_window_high"}
    missing = required - set(preprocessing)
    if missing:
        raise ValueError(f"Checkpoint preprocessing metadata is missing: {sorted(missing)}")
    modality = preprocessing["modality"]
    if modality not in {"generic", "ct", "mri"}:
        raise ValueError(f"Unsupported checkpoint modality: {modality!r}.")
    image_size = preprocessing["image_size_voxels"]
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 16:
        raise ValueError("Checkpoint image_size_voxels must be an integer >= 16.")
    spacing = float(preprocessing["target_spacing_mm"])
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Checkpoint target_spacing_mm must be finite and positive.")
    low, high = preprocessing["ct_window_low"], preprocessing["ct_window_high"]
    if modality == "ct":
        if low is None or high is None:
            raise ValueError("CT checkpoints must include CT window bounds.")
        low, high = float(low), float(high)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("Checkpoint CT window bounds must be finite with high > low.")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint model_state_dict is missing or empty.")
    return model_config, preprocessing


def prepare_prediction_manifest(frame: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """Validate prediction rows and resolve each scan path relative to the CSV."""
    required = {"patient_id", "image_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction manifest is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Prediction manifest has no rows.")
    frame = frame.copy()
    if frame[list(required)].isna().any().any():
        raise ValueError("Prediction manifest contains empty patient_id or image_path values.")
    frame["patient_id"] = frame["patient_id"].astype(str).str.strip()
    frame["image_path"] = frame["image_path"].astype(str).str.strip()
    if (frame["patient_id"] == "").any() or (frame["image_path"] == "").any():
        raise ValueError("patient_id and image_path must not be blank.")

    if "visit_idx" in frame.columns:
        visits = pd.to_numeric(frame["visit_idx"], errors="raise").to_numpy(dtype=float)
        if (not np.isfinite(visits).all()
                or not np.equal(visits, np.floor(visits)).all()
                or (visits < 0).any()):
            raise ValueError("visit_idx must contain finite non-negative integers.")
        frame["visit_idx"] = visits.astype(np.int64)
        if frame.duplicated(["patient_id", "visit_idx"]).any():
            raise ValueError("Each patient must have at most one prediction row per visit_idx.")

    if "time_months" in frame.columns:
        times = pd.to_numeric(frame["time_months"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(times).all() or (times < 0).any():
            raise ValueError("time_months must contain finite, non-negative values.")
        frame["time_months"] = times

    def resolve_path(value: str) -> str:
        path = Path(value).expanduser()
        return str(path.resolve() if path.is_absolute() else (base_dir / path).resolve())

    frame["image_path"] = frame["image_path"].map(resolve_path)
    if frame.duplicated(["patient_id", "image_path"]).any():
        raise ValueError("A patient cannot reference the same image path more than once.")
    owners = frame.groupby("image_path")["patient_id"].nunique()
    if (owners > 1).any():
        raise ValueError("An image_path cannot be assigned to multiple patient_id values.")

    time_order = ["patient_id"] + (["visit_idx"] if "visit_idx" in frame.columns else [])
    if "time_months" in frame.columns:
        ordered = frame.sort_values(time_order, kind="stable")
        for patient_id, visits in ordered.groupby("patient_id", sort=False):
            if (np.diff(visits["time_months"].to_numpy(dtype=float)) < 0).any():
                raise ValueError(
                    f"time_months must not decrease across visits for patient {patient_id!r}."
                )

    for path in frame["image_path"]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Image path in prediction manifest does not exist: {path}")
    return frame.reset_index(drop=True)


def predict_frame(
    frame: pd.DataFrame, model: torch.nn.Module, device: torch.device, preprocessing: dict
) -> pd.DataFrame:
    """Predict each scan with precisely the checkpoint's saved preprocessing settings."""
    modality = preprocessing["modality"]
    image_size = int(preprocessing["image_size_voxels"])
    target_spacing = float(preprocessing["target_spacing_mm"])
    low = preprocessing["ct_window_low"]
    high = preprocessing["ct_window_high"]
    if low is None:
        low = -1000.0
    if high is None:
        high = 1000.0

    output_rows = []
    model.eval()
    with torch.inference_mode():
        for _, row in frame.iterrows():
            volume = read_medical_volume(str(row["image_path"]), target_spacing_mm=target_spacing)
            tensor = volume_to_tensor(
                volume, image_size=image_size, modality=modality,
                ct_window_low=float(low), ct_window_high=float(high),
            ).unsqueeze(0).to(device)
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            result = {
                "patient_id": row["patient_id"],
                "image_path": row["image_path"],
                "predicted_stage": int(np.argmax(probabilities)),
                "confidence": float(np.max(probabilities)),
            }
            if "visit_idx" in frame.columns:
                result["visit_idx"] = int(row["visit_idx"])
            if "time_months" in frame.columns:
                result["time_months"] = float(row["time_months"])
            result.update({f"prob_stage_{index}": float(value)
                           for index, value in enumerate(probabilities)})
            output_rows.append(result)
    return pd.DataFrame(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Versioned raw_image_3dcnn.pt from training")
    parser.add_argument("--manifest", required=True, help="CSV with patient_id,image_path and optional visit metadata")
    parser.add_argument("--output", default="raw_image_predictions.csv", help="Destination CSV path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prediction manifest not found: {manifest_path}")
    if output_path in {checkpoint_path, manifest_path}:
        raise ValueError("Output path must not overwrite the checkpoint or input manifest.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load checkpoint {checkpoint_path}. Use a trusted checkpoint produced "
            "by the current training script."
        ) from exc
    _, preprocessing = validate_checkpoint(checkpoint)
    manifest = prepare_prediction_manifest(pd.read_csv(manifest_path), manifest_path.parent)
    model = Small3DCNN(n_classes=4).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, KeyError) as exc:
        raise ValueError("Checkpoint weights do not match the supported Small3DCNN architecture.") from exc
    predictions = predict_frame(manifest, model, device, preprocessing)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Predicted {len(predictions)} scans across {manifest['patient_id'].nunique()} patients.")
    print(f"Saved predictions to: {output_path}")
    print("Research output only: not a validated diagnosis or clinical decision.")


if __name__ == "__main__":
    main()
