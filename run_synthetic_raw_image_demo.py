"""Run both raw-image training paths end-to-end on generated synthetic NIfTI scans.

This is a pipeline smoke test, not medical research evidence. The generated labels,
measurements, and volumes are artificial and must never be interpreted clinically.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parent
VOLUME_SIZE = 16
VOXEL_SPACING_MM = 2.0
STAGE_REGIONS = (
    (slice(2, 6), slice(2, 6), slice(2, 6)),
    (slice(10, 14), slice(2, 6), slice(2, 6)),
    (slice(2, 6), slice(10, 14), slice(2, 6)),
    (slice(10, 14), slice(10, 14), slice(10, 14)),
)


def make_synthetic_dataset(output_dir: Path, patients: int, seed: int) -> tuple[Path, Path]:
    """Write reproducible toy 3D scans and training/prediction manifests."""
    rng = np.random.default_rng(seed)
    scans_dir = output_dir / "scans"
    scans_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    medicines = ("Synthetic_A", "Synthetic_B", "Synthetic_C")

    for patient_number in range(patients):
        patient_id = f"SYNTH_{patient_number:03d}"
        for visit_idx in range(2):
            stage = (patient_number + visit_idx) % 4
            volume = rng.normal(0.0, 0.7, size=(VOLUME_SIZE,) * 3).astype(np.float32)
            # A simple spatial pattern lets the 3D pipeline process non-identical
            # volumes. It is artificial and has no anatomical or clinical meaning.
            volume[STAGE_REGIONS[stage]] += 1.6
            volume += np.float32(visit_idx * 0.05)

            scan_name = f"{patient_id}_visit{visit_idx}.nii.gz"
            scan_path = scans_dir / scan_name
            image = sitk.GetImageFromArray(volume)
            image.SetSpacing((VOXEL_SPACING_MM,) * 3)
            sitk.WriteImage(image, str(scan_path))

            rows.append({
                "patient_id": patient_id,
                "image_path": f"scans/{scan_name}",
                "stage": stage,
                "visit_idx": visit_idx,
                "time_months": float(visit_idx * 6),
                # These are synthetic measurements, not clinical variables.
                "C_synthetic_measurement": float(stage * 0.7 + rng.normal(0.0, 0.8)),
                "C_synthetic_age": float(35 + patient_number),
                "M_synthetic_medicine": medicines[patient_number % len(medicines)],
            })

    training_manifest = output_dir / "training_manifest.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(training_manifest, index=False)

    # Exercise inference without labels, for two visits from one synthetic subject.
    prediction_manifest = output_dir / "prediction_manifest.csv"
    prediction_rows = frame.loc[frame["patient_id"] == "SYNTH_000"].drop(columns=["stage"]).copy()
    if len(prediction_rows) != 2:
        raise RuntimeError("Synthetic prediction data should contain exactly two visits.")
    # Deliberately test the model's documented handling of an unseen text category.
    prediction_rows.loc[prediction_rows["visit_idx"] == 1, "M_synthetic_medicine"] = (
        "UNSEEN_SYNTHETIC_CATEGORY"
    )
    prediction_rows.to_csv(prediction_manifest, index=False)
    return training_manifest, prediction_manifest


def run(command: list[str], *, label: str, output_dir: Path) -> None:
    print(f"\n=== {label} ===", flush=True)
    print("$ " + shlex.join(command), flush=True)
    environment = os.environ.copy()
    # Keep the small smoke run suitable for CPU-only workspaces.
    environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="raw_image_smoke_demo",
        help="directory for generated scans, manifests, checkpoints, and predictions",
    )
    parser.add_argument("--patients", type=int, default=20,
                        help="number of synthetic subjects; at least 10 are required")
    parser.add_argument("--epochs", type=int, default=1,
                        help="training epochs per model (smoke-test default: 1)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.patients < 10:
        parser.error("--patients must be at least 10 for the patient-level split")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_manifest, prediction_manifest = make_synthetic_dataset(
        output_dir, args.patients, args.seed
    )
    print(
        f"Generated {args.patients * 2} synthetic 3D NIfTI scans for "
        f"{args.patients} synthetic subjects under: {output_dir}",
        flush=True,
    )
    print("No real medical scans or patient records are used.", flush=True)

    image_only_results = output_dir / "image_only_results"
    run([
        sys.executable, str(ROOT / "train_raw_image_model.py"),
        "--manifest", str(training_manifest),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--image-size", str(VOLUME_SIZE),
        "--target-spacing-mm", str(VOXEL_SPACING_MM),
        "--modality", "generic",
        "--output-dir", str(image_only_results),
        "--seed", str(args.seed),
    ], label="Image-only 3D CNN: train and evaluate", output_dir=output_dir)

    multimodal_results = output_dir / "multimodal_results"
    run([
        sys.executable, str(ROOT / "train_multimodal_raw_image_model.py"),
        "--manifest", str(training_manifest),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--image-size", str(VOLUME_SIZE),
        "--target-spacing-mm", str(VOXEL_SPACING_MM),
        "--modality", "generic",
        "--output-dir", str(multimodal_results),
        "--seed", str(args.seed),
    ], label="Raw-image + tabular temporal model: train and evaluate",
       output_dir=output_dir)

    prediction_output = output_dir / "predictions.csv"
    run([
        sys.executable, str(ROOT / "predict_multimodal_raw_image_model.py"),
        "--checkpoint", str(multimodal_results / "multimodal_raw_model.pt"),
        "--manifest", str(prediction_manifest),
        "--output", str(prediction_output),
        "--batch-size", str(args.batch_size),
        "--device", "cpu",
    ], label="Multimodal checkpoint: predict two visits", output_dir=output_dir)

    print("\nSynthetic raw-image smoke run completed.", flush=True)
    print(f"Training manifest: {training_manifest}", flush=True)
    print(f"Prediction manifest: {prediction_manifest}", flush=True)
    print(f"Image-only results: {image_only_results}", flush=True)
    print(f"Multimodal results: {multimodal_results}", flush=True)
    print(f"Two-visit predictions: {prediction_output}", flush=True)
    print(
        "These synthetic metrics only confirm that the code paths execute; they do not "
        "measure clinical performance or establish real-world validity.",
        flush=True,
    )


if __name__ == "__main__":
    main()
