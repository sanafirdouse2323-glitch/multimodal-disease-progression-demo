"""Evaluate whole-modality missingness in the synthetic longitudinal demo.

This experiment is inspired by research on longitudinal multimodal prediction
with missingness. It is not a reproduction of that paper or a clinical study.
The prediction target remains the synthetic stage at an observed visit.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import demo_disease_progression as demo  # noqa: E402

MODALITIES = tuple(demo.GROUPS.keys())
CONDITION_ORDER = ("complete",) + tuple(f"without_{name}" for name in MODALITIES)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "missing_modality_robustness"


def remove_modalities(sequences: list[dict], modalities: tuple[str, ...] | list[str]) -> list[dict]:
    """Return copies with selected modalities entirely unavailable at every visit.

    Both scaled values and per-feature observation indicators are cleared. Labels,
    visit times, other modalities, and the original sequences are left unchanged.
    """
    unknown = set(modalities) - set(MODALITIES)
    if unknown:
        raise ValueError(f"Unknown modality names: {sorted(unknown)}")

    result: list[dict] = []
    for sequence in sequences:
        masked = dict(sequence)
        for name in modalities:
            masked[name] = np.zeros_like(sequence[name])
            masked[f"{name}_mask"] = np.zeros_like(sequence[f"{name}_mask"])
        result.append(masked)
    return result


class ModalityDropoutDataset(Dataset):
    """Apply stochastic whole-modality dropout to training trajectories only."""

    def __init__(self, sequences: list[dict], probability: float = 0.20):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        self.sequences = sequences
        self.probability = probability

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict:
        source = self.sequences[index]
        item = dict(source)
        dropped = [name for name in MODALITIES if random.random() < self.probability]
        # Keep at least one modality available. Complete loss is not the condition
        # this experiment is designed to evaluate.
        if len(dropped) == len(MODALITIES) and dropped:
            dropped.pop(random.randrange(len(dropped)))
        for name in dropped:
            item[name] = np.zeros_like(source[name])
            item[f"{name}_mask"] = np.zeros_like(source[f"{name}_mask"])
        return item


def make_conditions(sequences: list[dict]) -> dict[str, list[dict]]:
    """Create complete-input and leave-one-modality-out evaluation conditions."""
    conditions = {"complete": sequences}
    for name in MODALITIES:
        conditions[f"without_{name}"] = remove_modalities(sequences, [name])
    return conditions


def _loader(sequences: list[dict], batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        demo.PatientDataset(sequences),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=demo.collate_sequences,
        num_workers=0,
    )


def evaluate_condition(model: nn.Module, sequences: list[dict], device: torch.device,
                       batch_size: int) -> dict[str, float | int]:
    """Evaluate visit-level predictions for one held-out condition."""
    result = demo.evaluate_torch_model(model, _loader(sequences, batch_size), device)
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    return {
        "n_visits": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_experiment_model(
    train_sequences: list[dict],
    valid_sequences: list[dict],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    modality_dropout_probability: float,
) -> tuple[nn.Module, list[dict]]:
    """Train a model; choose checkpoints using identical validation conditions.

    Both training strategies are selected using the mean validation loss across
    complete input and all single-modality-missing conditions. The test set is
    not used for checkpoint selection.
    """
    if modality_dropout_probability > 0:
        train_dataset = ModalityDropoutDataset(train_sequences, modality_dropout_probability)
    else:
        train_dataset = demo.PatientDataset(train_sequences)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=demo.collate_sequences, num_workers=0
    )
    validation_conditions = make_conditions(valid_sequences)
    validation_loaders = {
        name: _loader(sequences, batch_size)
        for name, sequences in validation_conditions.items()
    }

    model = demo.MultimodalTemporalModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_validation_loss = float("inf")
    best_state = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = demo.move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        validation_losses = []
        for loader in validation_loaders.values():
            metrics = demo.evaluate_torch_model(model, loader, device)
            validation_losses.append(float(metrics["loss"]))
        mean_validation_loss = float(np.mean(validation_losses))
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "mean_validation_loss_across_conditions": mean_validation_loss,
        })
        print(
            f"    epoch {epoch:02d}/{epochs}: train_loss={history[-1]['train_loss']:.4f}, "
            f"mean_validation_loss={mean_validation_loss:.4f}"
        )
        if mean_validation_loss < best_validation_loss:
            best_validation_loss = mean_validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def summarise_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate independent-run metrics without hiding run-level observations."""
    grouped = results.groupby(["strategy", "condition"], sort=False)
    summary = grouped.agg(
        runs=("seed", "nunique"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    ).reset_index()
    summary["condition"] = pd.Categorical(
        summary["condition"], categories=CONDITION_ORDER, ordered=True
    )
    strategy_order = {"standard": 0, "modality_dropout": 1}
    summary["_strategy_order"] = summary["strategy"].map(strategy_order)
    return summary.sort_values(["condition", "_strategy_order"]).drop(
        columns=["_strategy_order"]
    ).reset_index(drop=True)


def save_plot(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(CONDITION_ORDER))
    width = 0.36
    for offset, strategy, label in [
        (-width / 2, "standard", "Standard training"),
        (width / 2, "modality_dropout", "Modality-dropout training"),
    ]:
        subset = summary[summary["strategy"] == strategy].set_index("condition").reindex(
            CONDITION_ORDER
        )
        means = subset["macro_f1_mean"].to_numpy(dtype=float)
        stds = subset["macro_f1_std"].fillna(0.0).to_numpy(dtype=float)
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=label)
    labels = ["Complete"] + [f"No {name.replace('_', ' ').title()}" for name in MODALITIES]
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Macro-F1 (mean across runs; error bars = SD)")
    ax.set_ylim(0, 1)
    ax.set_title("Synthetic held-out performance when whole modalities are unavailable")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("Provide at least one integer seed, e.g. --seeds 42,43,44")
    if args.n_patients < 30:
        raise ValueError("--n-patients must be at least 30 so all splits contain multiple patients")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if not 0.0 < args.dropout_probability < 1.0:
        raise ValueError("--dropout-probability must be strictly between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    all_rows: list[dict] = []
    split_counts: list[dict] = []
    print(f"Device: {device}")
    print(f"Seeds: {seeds}; patients per run: {args.n_patients}; epochs: {args.epochs}")

    for seed in seeds:
        print(f"\n=== Run seed {seed} ===")
        demo.set_seed(seed)
        df = demo.generate_demo_dataset(n_patients=args.n_patients, seed=seed)
        train_ids, valid_ids, test_ids = demo.split_patient_ids(df, seed=seed)
        train_raw = df[df["patient_id"].isin(train_ids)].copy()
        valid_raw = df[df["patient_id"].isin(valid_ids)].copy()
        test_raw = df[df["patient_id"].isin(test_ids)].copy()

        # Fit preprocessing only on training patients.
        means, stds = demo.fit_feature_stats(train_raw)
        train_df = demo.transform_features(train_raw, means, stds)
        valid_df = demo.transform_features(valid_raw, means, stds)
        test_df = demo.transform_features(test_raw, means, stds)
        train_sequences, valid_sequences, test_sequences = map(
            demo.make_sequences, [train_df, valid_df, test_df]
        )
        split_counts.append({
            "seed": seed,
            "train_patients": len(train_ids),
            "validation_patients": len(valid_ids),
            "test_patients": len(test_ids),
            "test_visits": int(len(test_df)),
        })
        print(
            f"Patients: train={len(train_ids)}, validation={len(valid_ids)}, "
            f"test={len(test_ids)}; test visits={len(test_df)}"
        )

        for strategy, dropout_probability in [
            ("standard", 0.0),
            ("modality_dropout", args.dropout_probability),
        ]:
            # Matched initialisation/training seed makes the strategy comparison
            # easier to interpret; only the data augmentation differs.
            demo.set_seed(seed + 1000)
            print(f"  Training strategy: {strategy}")
            model, history = train_experiment_model(
                train_sequences=train_sequences,
                valid_sequences=valid_sequences,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                modality_dropout_probability=dropout_probability,
            )
            pd.DataFrame(history).to_csv(
                output_dir / f"history_seed_{seed}_{strategy}.csv", index=False
            )

            for condition, conditioned_test in make_conditions(test_sequences).items():
                metrics = evaluate_condition(model, conditioned_test, device, args.batch_size)
                row = {
                    "seed": seed,
                    "strategy": strategy,
                    "condition": condition,
                    "missing_modalities": ",".join(
                        [condition.replace("without_", "")] if condition != "complete" else []
                    ),
                    "n_test_patients": len(test_ids),
                    **metrics,
                }
                all_rows.append(row)
                print(
                    f"    {condition:20s} macro_f1={metrics['macro_f1']:.4f} "
                    f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
                )

    results = pd.DataFrame(all_rows)
    summary = summarise_results(results)
    results.to_csv(output_dir / "run_level_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_by_condition.csv", index=False)
    save_plot(summary, output_dir / "macro_f1_by_missing_modality.png")
    report = {
        "experiment": "whole-modality missingness robustness",
        "data": "synthetic only",
        "prediction_target": "synthetic disease stage at an observed visit",
        "not_claimed": [
            "clinical validity or performance",
            "future disease-progression forecasting",
            "causal treatment-effect estimation",
            "fairness validation on real demographic groups",
        ],
        "n_patients_per_run": args.n_patients,
        "seeds": seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "modality_dropout_probability_per_modality": args.dropout_probability,
        "model_selection": (
            "Checkpoint selected by mean validation loss across complete input and "
            "each leave-one-modality-out condition; test data are not used for selection."
        ),
        "split_counts": split_counts,
        "conditions": list(CONDITION_ORDER),
        "strategies": ["standard", "modality_dropout"],
        "references": [
            {
                "citation": "Chen, Oliva, and Niethammer, MICCAI 2024, "
                            "A Unified Model for Longitudinal Multi-Modal Multi-View Prediction with Missingness",
                "url": "https://papers.miccai.org/miccai-2024/034-Paper2390.html",
                "relationship": "Related methodological context; this script does not reproduce UniLMMV."
            },
            {
                "citation": "Koch, Baumgartner, and Berens, npj Digital Medicine 2024, "
                            "Distribution Shift Detection for the Postmarket Surveillance of Medical AI Algorithms",
                "url": "https://www.nature.com/articles/s41746-024-01085-w",
                "relationship": "Related trustworthy-evaluation context; distribution-shift detection is not implemented here."
            },
        ],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nSaved run-level metrics, summary, plot, and configuration to {output_dir}")
    print("Interpretation: these results describe only the stated synthetic generator.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare standard training with whole-modality dropout on synthetic data."
    )
    parser.add_argument("--n-patients", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout-probability", type=float, default=0.20)
    parser.add_argument("--seeds", default="42,43,44",
                        help="Comma-separated run seeds (default: 42,43,44).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
