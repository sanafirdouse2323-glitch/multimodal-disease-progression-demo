"""Synthetic multimodal longitudinal disease-stage demo.

This is an educational portfolio project, not a clinical model. It predicts the
current synthetic stage from information available at that visit and earlier
visits. All data are generated locally; no patient data are used.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

SEED = 42
OUTPUT_DIR = Path("results")
CLINICAL_COLS = ["lab1", "lab2", "vitals", "meds", "comorbidities", "age", "sex", "bmi"]
TREATMENT_COLS = ["treatment_flag", "time_since_treatment_start", "treatment_duration_days", "treatment_change"]
MEDICINE_COLS = ["medicine_type", "dose_per_day", "frequency_per_day", "cumulative_exposure"]
WEARABLE_COLS = ["steps", "sleep_hours", "heart_rate", "hrv", "activity"]
IMAGING_COLS = ["imaging_score", "lesion_area", "atrophy_index", "tissue_change"]
GROUPS = {
    "clinical": CLINICAL_COLS,
    "treatment": TREATMENT_COLS,
    "medicine": MEDICINE_COLS,
    "wearable": WEARABLE_COLS,
    "imaging": IMAGING_COLS,
}
LABEL_COL = "stage"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_demo_dataset(n_patients: int = 500, seed: int = SEED) -> pd.DataFrame:
    """Generate toy longitudinal data with non-uniform visits and missing values."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    efficacy_by_type = {1: 0.025, 2: 0.035, 3: 0.045, 4: 0.055}

    for pid in range(n_patients):
        age = float(rng.uniform(45, 85))
        bmi = float(np.clip(rng.normal(27, 4.5), 16, 45))
        sex = int(rng.integers(0, 2))
        baseline_risk = float(rng.normal(0, 0.65))
        progression_rate = float(rng.uniform(0.10, 0.32))
        n_visits = int(rng.integers(4, 9))
        gaps = rng.uniform(1.0, 5.0, size=n_visits - 1)
        visit_times = np.concatenate(([0.0], np.cumsum(gaps)))

        latent = baseline_risk
        on_treatment = False
        treatment_start_month = None
        medicine_type = 0
        cumulative_exposure = 0.0

        for visit_idx, months in enumerate(visit_times):
            if visit_idx > 0:
                elapsed = float(months - visit_times[visit_idx - 1])
                # Disease evolves during the interval; an active treatment reduces
                # the simulated drift. This is a toy data-generating rule only.
                effect = efficacy_by_type.get(medicine_type, 0.0) if on_treatment else 0.0
                latent += progression_rate * elapsed * (1.0 - effect) + rng.normal(0, 0.28)

            # Treatment initiation is more likely at higher simulated disease burden.
            p_start = float(np.clip(0.08 + 0.22 / (1.0 + np.exp(-latent)), 0.05, 0.32))
            treatment_change = 0
            if not on_treatment and rng.random() < p_start:
                on_treatment = True
                treatment_start_month = float(months)
                medicine_type = int(rng.integers(1, 5))
                treatment_change = 1
            elif on_treatment and rng.random() < 0.06:
                # A treatment can stop; this avoids exposure continuing after stop.
                on_treatment = False
                treatment_start_month = None
                medicine_type = 0
                treatment_change = 1
            elif on_treatment and rng.random() < 0.10:
                medicine_type = int(rng.integers(1, 5))
                treatment_change = 1

            treatment_flag = int(on_treatment)
            dose_per_day = float(max(0.1, rng.normal(1.0 + 0.35 * medicine_type, 0.15))) if on_treatment else 0.0
            frequency_per_day = float(rng.integers(1, 3)) if on_treatment else 0.0
            time_since_start = max(0.0, float(months - treatment_start_month)) if on_treatment else 0.0
            duration_days = time_since_start * 30.0
            # Exposure is an accumulated proxy, not a clinically meaningful dose.
            if visit_idx > 0 and on_treatment:
                elapsed_days = float(months - visit_times[visit_idx - 1]) * 30.0
                cumulative_exposure += dose_per_day * frequency_per_day * elapsed_days
            elif not on_treatment:
                cumulative_exposure = 0.0

            # Stage is derived from the continuous state; observed features are
            # generated from the state with noise, not directly from the stage.
            stage = int(np.digitize(latent, bins=[-0.65, 0.35, 1.45]))
            noise = lambda sd: float(rng.normal(0.0, sd))
            row = {
                "patient_id": pid, "visit_idx": visit_idx, "time_months": float(months),
                "stage": stage, "age": age, "sex": sex, "bmi": bmi,
                "lab1": 85 + 8.5 * latent + noise(7.0),
                "lab2": 5.2 + 1.5 * latent + noise(1.8),
                "vitals": 71 + 3.5 * latent + noise(5.5),
                "meds": int(max(0, rng.poisson(max(0.2, 1.5 + 0.65 * latent)))),
                "comorbidities": int(max(0, rng.poisson(max(0.2, 1.0 + 0.35 * latent)))),
                "treatment_flag": treatment_flag,
                "time_since_treatment_start": time_since_start,
                "treatment_duration_days": duration_days,
                "treatment_change": treatment_change,
                "medicine_type": medicine_type,
                "dose_per_day": dose_per_day,
                "frequency_per_day": frequency_per_day,
                "cumulative_exposure": cumulative_exposure,
                "steps": float(max(500, 8500 - 850 * latent + noise(1700))),
                "sleep_hours": float(np.clip(7.5 - 0.35 * latent + noise(1.0), 3, 12)),
                "heart_rate": 69 + 4.2 * latent + noise(5.0),
                "hrv": 50 - 4.5 * latent + noise(8.0),
                "activity": float(np.clip(0.68 - 0.10 * latent + noise(0.16), 0, 1)),
                "imaging_score": 18 + 12 * latent + noise(7.0),
                "lesion_area": float(max(0, 4 + 3.2 * latent + noise(2.8))),
                "atrophy_index": float(max(0, 0.18 + 0.12 * latent + noise(0.09))),
                "tissue_change": float(max(0, 0.12 + 0.20 * latent + noise(0.14))),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    # Missingness is generated independently of the target label in this demo.
    # This keeps the toy missingness mechanism easy to explain.
    for cols, rate in [
        (CLINICAL_COLS, 0.08), (TREATMENT_COLS + MEDICINE_COLS, 0.05),
        (WEARABLE_COLS, 0.18), (IMAGING_COLS, 0.15),
    ]:
        for col in cols:
            missing = rng.random(len(df)) < rate
            df.loc[missing, col] = np.nan
    return df


def split_patient_ids(df: pd.DataFrame, seed: int = SEED):
    ids = np.array(sorted(df["patient_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)
    train_ids = set(ids[:n_train])
    valid_ids = set(ids[n_train:n_train + n_valid])
    test_ids = set(ids[n_train + n_valid:])
    return train_ids, valid_ids, test_ids


def fit_feature_stats(train_df: pd.DataFrame):
    """Fit scaling statistics using training patients only."""
    feature_cols = [col for cols in GROUPS.values() for col in cols]
    means = train_df[feature_cols].mean().fillna(0.0)
    stds = train_df[feature_cols].std(ddof=0).replace(0, 1.0).fillna(1.0)
    return means, stds


def transform_features(df: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    """Scale features with training-only statistics and preserve original missingness."""
    out = df.copy()
    for col in means.index:
        # Create the mask before imputation so missing values remain identifiable.
        out[f"{col}__observed"] = out[col].notna().astype(np.float32)
        out[col] = ((out[col] - means[col]) / stds[col]).fillna(0.0)
    return out


def prepare_features(train_df: pd.DataFrame, other_df: pd.DataFrame):
    """Fit scaling on train, then apply the same statistics to another split."""
    means, stds = fit_feature_stats(train_df)
    return transform_features(train_df, means, stds), transform_features(other_df, means, stds)


def make_sequences(df: pd.DataFrame) -> list[dict]:
    sequences = []
    for _, group in df.groupby("patient_id", sort=True):
        group = group.sort_values("visit_idx")
        item = {
            "time": group["time_months"].to_numpy(dtype=np.float32),
            "labels": group[LABEL_COL].to_numpy(dtype=np.int64),
        }
        for name, cols in GROUPS.items():
            values = group[cols].to_numpy(dtype=np.float32)
            masks = group[[f"{c}__observed" for c in cols]].to_numpy(dtype=np.float32)
            item[name] = values
            item[f"{name}_mask"] = masks
        sequences.append(item)
    return sequences


class PatientDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


def collate_sequences(batch):
    max_len = max(len(item["labels"]) for item in batch)
    result = {}
    keys = list(GROUPS.keys()) + [f"{name}_mask" for name in GROUPS] + ["time", "labels"]
    for key in keys:
        arrays = []
        for item in batch:
            arr = item[key]
            pad_shape = (max_len - len(arr),) + arr.shape[1:]
            if key == "labels":
                pad = np.full(pad_shape, -100, dtype=np.int64)
            else:
                pad = np.zeros(pad_shape, dtype=np.float32)
            arrays.append(np.concatenate([arr, pad], axis=0))
        dtype = torch.long if key == "labels" else torch.float32
        result[key] = torch.as_tensor(np.stack(arrays), dtype=dtype)
    result["padding_mask"] = result["labels"].eq(-100)
    return result


class MultimodalTemporalModel(nn.Module):
    def __init__(self, hidden_dim=32, n_heads=4, n_classes=4):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(2 * len(cols), hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
            ) for name, cols in GROUPS.items()
        })
        self.fusion = nn.Linear(hidden_dim * len(GROUPS), hidden_dim)
        self.time_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, batch):
        encoded = []
        for name in GROUPS:
            # Keep values and observation indicators as separate model inputs.
            encoded.append(self.encoders[name](torch.cat([batch[name], batch[f"{name}_mask"]], dim=-1)))
        fused = self.fusion(torch.cat(encoded, dim=-1))
        times = batch["time"].unsqueeze(-1)
        # Scale time to a stable range; time is relative to each patient's first visit.
        time_scale = times / 12.0
        fused = fused + self.time_encoder(time_scale)
        length = fused.size(1)
        # Causal attention prevents visit t from using information from future visits.
        causal_mask = torch.triu(
            torch.ones(length, length, device=fused.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.temporal(
            fused, mask=causal_mask, src_key_padding_mask=batch["padding_mask"]
        )
        return self.classifier(hidden)


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def evaluate_torch_model(model, loader, device):
    model.eval()
    losses, all_y, all_pred = [], [], []
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))
            losses.append(float(loss.item()))
            valid = batch["labels"] != -100
            all_y.extend(batch["labels"][valid].cpu().tolist())
            all_pred.extend(logits.argmax(-1)[valid].cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(all_y, all_pred)),
        "macro_f1": float(f1_score(all_y, all_pred, average="macro", zero_division=0)),
        "y_true": all_y, "y_pred": all_pred,
    }


def train_model(train_sequences, valid_sequences, device, epochs=18, batch_size=16, lr=1e-3):
    train_loader = DataLoader(PatientDataset(train_sequences), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_sequences)
    valid_loader = DataLoader(PatientDataset(valid_sequences), batch_size=batch_size, shuffle=False,
                              collate_fn=collate_sequences)
    model = MultimodalTemporalModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_loss, best_state = float("inf"), None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        metrics = evaluate_torch_model(model, valid_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(batch_losses)), "val_loss": metrics["loss"],
               "val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"]}
        history.append(row)
        print("Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              "val_accuracy={val_accuracy:.4f} | val_macro_f1={val_macro_f1:.4f}".format(**row))
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def flatten_visit_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Flatten visits for baselines, including explicit missingness indicators."""
    cols = [col for group in GROUPS.values() for col in group]
    values = df[cols].to_numpy(dtype=np.float32)
    observed = df[[f"{col}__observed" for col in cols]].to_numpy(dtype=np.float32)
    features = np.concatenate([values, observed], axis=1)
    return features, df[LABEL_COL].to_numpy(dtype=np.int64)


def save_confusion_matrix(y_true, y_pred, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1, 2, 3],
                                            display_labels=["Stage 0", "Stage 1", "Stage 2", "Stage 3"],
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating synthetic multimodal longitudinal data...")
    df = generate_demo_dataset(n_patients=500, seed=SEED)
    train_ids, valid_ids, test_ids = split_patient_ids(df, seed=SEED)
    train_raw = df[df.patient_id.isin(train_ids)].copy()
    valid_raw = df[df.patient_id.isin(valid_ids)].copy()
    test_raw = df[df.patient_id.isin(test_ids)].copy()
    assert not (train_ids & valid_ids or train_ids & test_ids or valid_ids & test_ids)

    # Fit preprocessing once on training patients and reuse it unchanged.
    means, stds = fit_feature_stats(train_raw)
    train_df = transform_features(train_raw, means, stds)
    valid_df = transform_features(valid_raw, means, stds)
    test_df = transform_features(test_raw, means, stds)

    train_sequences, valid_sequences, test_sequences = map(
        make_sequences, [train_df, valid_df, test_df]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; patients: train={len(train_ids)}, validation={len(valid_ids)}, test={len(test_ids)}")

    model, history = train_model(train_sequences, valid_sequences, device=device)
    test_loader = DataLoader(PatientDataset(test_sequences), batch_size=16, shuffle=False,
                             collate_fn=collate_sequences)
    neural = evaluate_torch_model(model, test_loader, device)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    # Simple same-visit baselines. Fit preprocessing/model only on training visits;
    # patient-level split ensures no patient's visits cross partitions.
    X_train, y_train = flatten_visit_features(train_df)
    X_test, y_test = flatten_visit_features(test_df)
    baseline_rows = []
    for name, estimator in [
        ("majority_class", None),
        ("logistic_regression", make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                               LogisticRegression(max_iter=1000, class_weight="balanced"))),
        ("random_forest", make_pipeline(SimpleImputer(strategy="median"),
                                        RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                                                               class_weight="balanced", random_state=SEED,
                                                               n_jobs=-1))),
    ]:
        if estimator is None:
            majority = int(pd.Series(y_train).mode().iloc[0])
            pred = np.full_like(y_test, majority)
        else:
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)
        baseline_rows.append({
            "model": name, "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        })
    baseline_rows.append({
        "model": "multimodal_causal_transformer", "accuracy": neural["accuracy"],
        "macro_f1": neural["macro_f1"],
    })
    pd.DataFrame(baseline_rows).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)

    save_confusion_matrix(neural["y_true"], neural["y_pred"],
                          OUTPUT_DIR / "transformer_confusion_matrix.png",
                          "Transformer: held-out synthetic test set")
    fig, ax = plt.subplots(figsize=(7, 4))
    hist = pd.DataFrame(history)
    ax.plot(hist["epoch"], hist["train_loss"], label="Training loss")
    ax.plot(hist["epoch"], hist["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curve.png", dpi=160)
    plt.close(fig)

    report = {
        "seed": SEED, "n_patients": int(df.patient_id.nunique()),
        "n_visits": int(len(df)), "split_patient_counts": {
            "train": len(train_ids), "validation": len(valid_ids), "test": len(test_ids)
        },
        "test_metrics": baseline_rows,
        "classification_report_transformer": classification_report(
            neural["y_true"], neural["y_pred"], labels=[0, 1, 2, 3],
            target_names=["stage_0", "stage_1", "stage_2", "stage_3"],
            output_dict=True, zero_division=0
        ),
        "important_interpretation": (
            "All records are synthetic. Metrics demonstrate code execution on this toy generator only; "
            "they do not estimate clinical performance."
        ),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nHeld-out test metrics (synthetic data only):")
    print(pd.DataFrame(baseline_rows).to_string(index=False))
    print(f"\nSaved metrics and plots in: {OUTPUT_DIR.resolve()}")
    print("Reminder: these synthetic results are not evidence of clinical validity.")


if __name__ == "__main__":
    main()
