import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ----------------------------
# Feature groups
# ----------------------------
CLINICAL_BASE_COLS = ["lab1", "lab2", "vitals", "meds", "comorbidities"]

TREATMENT_COLS = [
    "treatment_flag",
    "time_since_treatment_start",
    "treatment_duration_days",
    "treatment_change",
]

MEDICINE_COLS = [
    "medicine_type",
    "dose_per_day",
    "frequency_per_day",
    "cumulative_exposure",
]

WEARABLE_COLS = ["steps", "sleep_hours", "heart_rate", "hrv", "activity"]
IMAGING_COLS = ["imaging_score", "lesion_area", "atrophy_index", "tissue_change"]

# ----------------------------
# Synthetic data generation
# ----------------------------
def generate_demo_dataset(n_patients=500, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for pid in range(n_patients):
        age = float(rng.uniform(45, 85))
        sex = int(rng.integers(0, 2))
        bmi = float(rng.normal(27.0, 4.5))
        baseline_risk = float(rng.normal(0.0, 1.0))
        progression_rate = float(rng.uniform(0.12, 0.9))
        visit_count = int(rng.integers(3, 9))

        visit_times = np.array([0.0])
        if visit_count > 1:
            gaps = rng.uniform(1.0, 7.0, size=visit_count - 1)
            visit_times = np.concatenate([visit_times, np.cumsum(gaps)])
        visit_times = np.round(visit_times, 1)

        on_treatment = False
        treatment_start_time = None
        medicine_type = 0
        cumulative_exposure = 0.0

        for visit_idx, months in enumerate(visit_times):
            latent = baseline_risk + progression_rate * months + rng.normal(0.0, 0.7)

            treatment_flag = 0
            medicine_type = 0
            dose_per_day = 0.0
            frequency_per_day = 0.0
            time_since_treatment_start = 0.0
            treatment_duration_days = 0.0
            treatment_change = 0

            if latent > -0.5:
                p_start = 0.10 + 0.30 * min(1.0, max(0.0, (latent + 2.0) / 4.0))
            else:
                p_start = 0.05

            if (not on_treatment) and (rng.random() < p_start):
                on_treatment = True
                treatment_start_time = months
                medicine_type = int(rng.integers(1, 5))
                treatment_flag = 1
                treatment_change = 1
            elif on_treatment:
                treatment_flag = 1
                if rng.random() < 0.15:
                    medicine_type = int(rng.integers(1, 5))
                    treatment_change = 1
                else:
                    medicine_type = medicine_type if medicine_type > 0 else 1

            if on_treatment:
                treatment_flag = 1

                efficacy_by_type = {1: 0.8, 2: 1.0, 3: 1.25, 4: 1.6}
                efficacy = efficacy_by_type.get(medicine_type, 1.0)

                dose_per_day = 0.8 + 0.7 * medicine_type + 0.25 * rng.normal(0.0, 1.0)
                dose_per_day = max(0.0, dose_per_day)

                frequency_per_day = float(int(rng.integers(1, 4)))
                time_since_treatment_start = months - treatment_start_time
                treatment_duration_days = max(0.0, time_since_treatment_start * 30.0)

                daily_exposure = dose_per_day * frequency_per_day
                cumulative_exposure += daily_exposure

                latent = latent - (
                    efficacy * cumulative_exposure * (0.12 + 0.08 * np.log1p(treatment_duration_days + 1.0))
                )

            if on_treatment and rng.random() < 0.08:
                on_treatment = False
                treatment_start_time = None
                medicine_type = 0
                dose_per_day = 0.0
                frequency_per_day = 0.0
                time_since_treatment_start = 0.0
                treatment_duration_days = 0.0
                treatment_flag = 0

            if latent < -1.5:
                stage = 0
            elif latent < -0.2:
                stage = 1
            elif latent < 1.2:
                stage = 2
            else:
                stage = 3

            lab1 = 82.0 + 12.0 * stage + rng.normal(0.0, 6.5)
            lab2 = 5.0 + 2.2 * stage + rng.normal(0.0, 1.5)
            vitals = 70.0 + 5.5 * stage + rng.normal(0.0, 5.0)
            meds = int(max(0, rng.poisson(2 + stage)))
            comorbidities = int(max(0, rng.poisson(1 + stage / 2)))

            steps = max(1000, 9000 - 1400 * stage + rng.normal(0.0, 2000))
            sleep_hours = max(4.0, 8.0 - 0.8 * stage + rng.normal(0.0, 1.2))
            heart_rate = 68.0 + 7.0 * stage + rng.normal(0.0, 5.0)
            hrv = 52.0 - 7.5 * stage + rng.normal(0.0, 8.0)
            activity = max(0.0, min(1.0, 0.7 - 0.18 * stage + rng.normal(0.0, 0.18)))

            imaging_score = 15.0 + 18.0 * stage + rng.normal(0.0, 5.0)
            lesion_area = max(0.0, 3.0 + 6.0 * stage + rng.normal(0.0, 2.5))
            atrophy_index = max(0.0, 0.15 + 0.25 * stage + rng.normal(0.0, 0.08))
            tissue_change = max(0.0, 0.08 + 0.45 * stage + rng.normal(0.0, 0.12))

            row = {
                "patient_id": pid,
                "visit_idx": visit_idx,
                "time_months": months,
                "stage": stage,
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "baseline_risk": baseline_risk,
                "progression_rate": progression_rate,
                "latent_score": latent,
                "lab1": lab1,
                "lab2": lab2,
                "vitals": vitals,
                "meds": meds,
                "comorbidities": comorbidities,
                "treatment_flag": treatment_flag,
                "time_since_treatment_start": time_since_treatment_start,
                "treatment_duration_days": treatment_duration_days,
                "treatment_change": treatment_change,
                "medicine_type": medicine_type,
                "dose_per_day": dose_per_day,
                "frequency_per_day": frequency_per_day,
                "cumulative_exposure": cumulative_exposure,
                "steps": steps,
                "sleep_hours": sleep_hours,
                "heart_rate": heart_rate,
                "hrv": hrv,
                "activity": activity,
                "imaging_score": imaging_score,
                "lesion_area": lesion_area,
                "atrophy_index": atrophy_index,
                "tissue_change": tissue_change,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    clinical_missing = 0.05 + 0.12 * df["stage"] / 3.0
    wearable_missing = 0.08 + 0.18 * df["stage"] / 3.0
    imaging_missing = 0.10 + 0.20 * df["stage"] / 3.0

    for col in CLINICAL_BASE_COLS + TREATMENT_COLS + MEDICINE_COLS:
        mask = rng.random(len(df)) < clinical_missing.to_numpy()
        df.loc[mask, col] = np.nan

    for col in WEARABLE_COLS:
        mask = rng.random(len(df)) < wearable_missing.to_numpy()
        df.loc[mask, col] = np.nan

    for col in IMAGING_COLS:
        mask = rng.random(len(df)) < imaging_missing.to_numpy()
        df.loc[mask, col] = np.nan

    return df

# ----------------------------
# Split by patient
# ----------------------------
def split_by_patient(df, valid_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    patient_ids = sorted(df["patient_id"].unique())
    rng.shuffle(patient_ids)

    split_idx = int(len(patient_ids) * (1.0 - valid_fraction))
    train_ids = set(patient_ids[:split_idx])
    valid_ids = set(patient_ids[split_idx:])

    train_df = df[df["patient_id"].isin(train_ids)].copy()
    valid_df = df[df["patient_id"].isin(valid_ids)].copy()
    return train_df, valid_df

# ----------------------------
# Standardization
# ----------------------------
def compute_group_stats(train_df, cols):
    stats = {}
    for col in cols:
        mean = train_df[col].mean()
        std = train_df[col].std(ddof=0)
        if pd.isna(std) or std < 1e-8:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def apply_standardization(df, stats):
    for col, info in stats.items():
        df[col] = (df[col] - info["mean"]) / info["std"]
        df[col] = df[col].fillna(0.0)
    return df

# ----------------------------
# Sequence construction
# ----------------------------
def make_patient_sequences(df):
    sequences = []
    for _, g in df.groupby("patient_id"):
        g = g.sort_values("visit_idx").reset_index(drop=True)

        clinical = g[CLINICAL_BASE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
        treatment = g[TREATMENT_COLS].fillna(0.0).to_numpy(dtype=np.float32)
        medicine = g[MEDICINE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
        wearable = g[WEARABLE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
        imaging = g[IMAGING_COLS].fillna(0.0).to_numpy(dtype=np.float32)

        clinical_mask = g[CLINICAL_BASE_COLS].notna().astype(float).to_numpy(dtype=np.float32)
        treatment_mask = g[TREATMENT_COLS].notna().astype(float).to_numpy(dtype=np.float32)
        medicine_mask = g[MEDICINE_COLS].notna().astype(float).to_numpy(dtype=np.float32)
        wearable_mask = g[WEARABLE_COLS].notna().astype(float).to_numpy(dtype=np.float32)
        imaging_mask = g[IMAGING_COLS].notna().astype(float).to_numpy(dtype=np.float32)

        labels = g["stage"].to_numpy(dtype=np.int64)

        sequences.append({
            "clinical": clinical,
            "clinical_mask": clinical_mask,
            "treatment": treatment,
            "treatment_mask": treatment_mask,
            "medicine": medicine,
            "medicine_mask": medicine_mask,
            "wearable": wearable,
            "wearable_mask": wearable_mask,
            "imaging": imaging,
            "imaging_mask": imaging_mask,
            "label": labels,
        })
    return sequences

# ----------------------------
# Dataset + collate
# ----------------------------
class PatientSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_patient_sequences(batch):
    max_len = max(item["clinical"].shape[0] for item in batch)

    clinical_list = []
    clinical_mask_list = []
    treatment_list = []
    treatment_mask_list = []
    medicine_list = []
    medicine_mask_list = []
    wearable_list = []
    wearable_mask_list = []
    imaging_list = []
    imaging_mask_list = []
    labels_list = []

    def pad_2d(arr, pad_len):
        if arr.shape[0] == pad_len:
            return arr
        pad = np.zeros((pad_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    for item in batch:
        seq_len = item["clinical"].shape[0]

        clinical_list.append(pad_2d(item["clinical"].astype(np.float32), max_len))
        clinical_mask_list.append(pad_2d(item["clinical_mask"].astype(np.float32), max_len))
        treatment_list.append(pad_2d(item["treatment"].astype(np.float32), max_len))
        treatment_mask_list.append(pad_2d(item["treatment_mask"].astype(np.float32), max_len))
        medicine_list.append(pad_2d(item["medicine"].astype(np.float32), max_len))
        medicine_mask_list.append(pad_2d(item["medicine_mask"].astype(np.float32), max_len))
        wearable_list.append(pad_2d(item["wearable"].astype(np.float32), max_len))
        wearable_mask_list.append(pad_2d(item["wearable_mask"].astype(np.float32), max_len))
        imaging_list.append(pad_2d(item["imaging"].astype(np.float32), max_len))
        imaging_mask_list.append(pad_2d(item["imaging_mask"].astype(np.float32), max_len))
        labels_list.append(np.pad(item["label"], (0, max_len - seq_len), constant_values=-100))

    return {
        "clinical": torch.tensor(np.stack(clinical_list), dtype=torch.float32),
        "clinical_mask": torch.tensor(np.stack(clinical_mask_list), dtype=torch.float32),
        "treatment": torch.tensor(np.stack(treatment_list), dtype=torch.float32),
        "treatment_mask": torch.tensor(np.stack(treatment_mask_list), dtype=torch.float32),
        "medicine": torch.tensor(np.stack(medicine_list), dtype=torch.float32),
        "medicine_mask": torch.tensor(np.stack(medicine_mask_list), dtype=torch.float32),
        "wearable": torch.tensor(np.stack(wearable_list), dtype=torch.float32),
        "wearable_mask": torch.tensor(np.stack(wearable_mask_list), dtype=torch.float32),
        "imaging": torch.tensor(np.stack(imaging_list), dtype=torch.float32),
        "imaging_mask": torch.tensor(np.stack(imaging_mask_list), dtype=torch.float32),
        "labels": torch.tensor(np.stack(labels_list), dtype=torch.long),
    }

# ----------------------------
# Model blocks
# ----------------------------
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class SeparateStreamTransformer(nn.Module):
    def __init__(
        self,
        clinical_dim,
        treatment_dim,
        medicine_dim,
        wearable_dim,
        imaging_dim,
        hidden_dim=32,
        n_heads=4,
        n_classes=4,
    ):
        super().__init__()

        self.clinical_encoder = ModalityEncoder(clinical_dim, hidden_dim)
        self.treatment_encoder = ModalityEncoder(treatment_dim, hidden_dim)
        self.medicine_encoder = ModalityEncoder(medicine_dim, hidden_dim)
        self.wearable_encoder = ModalityEncoder(wearable_dim, hidden_dim)
        self.imaging_encoder = ModalityEncoder(imaging_dim, hidden_dim)

        self.fusion = nn.Linear(hidden_dim * 5, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, clinical, clinical_mask, treatment, treatment_mask, medicine, medicine_mask, wearable, wearable_mask, imaging, imaging_mask):
        c = self.clinical_encoder(clinical * clinical_mask)
        t = self.treatment_encoder(treatment * treatment_mask)
        m = self.medicine_encoder(medicine * medicine_mask)
        w = self.wearable_encoder(wearable * wearable_mask)
        i = self.imaging_encoder(imaging * imaging_mask)

        fused = torch.cat([c, t, m, w, i], dim=-1)
        fused = self.fusion(fused)

        out = self.transformer(fused)
        logits = self.classifier(out)
        return logits

# ----------------------------
# Training
# ----------------------------
def train_model(train_sequences, valid_sequences, epochs=25, batch_size=16, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        PatientSequenceDataset(train_sequences),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_patient_sequences,
    )
    valid_loader = DataLoader(
        PatientSequenceDataset(valid_sequences),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_patient_sequences,
    )

    model = SeparateStreamTransformer(
        clinical_dim=len(CLINICAL_BASE_COLS),
        treatment_dim=len(TREATMENT_COLS),
        medicine_dim=len(MEDICINE_COLS),
        wearable_dim=len(WEARABLE_COLS),
        imaging_dim=len(IMAGING_COLS),
        hidden_dim=32,
        n_heads=4,
        n_classes=4,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            clinical = batch["clinical"].to(device)
            clinical_mask = batch["clinical_mask"].to(device)
            treatment = batch["treatment"].to(device)
            treatment_mask = batch["treatment_mask"].to(device)
            medicine = batch["medicine"].to(device)
            medicine_mask = batch["medicine_mask"].to(device)
            wearable = batch["wearable"].to(device)
            wearable_mask = batch["wearable_mask"].to(device)
            imaging = batch["imaging"].to(device)
            imaging_mask = batch["imaging_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits = model(
                clinical, clinical_mask,
                treatment, treatment_mask,
                medicine, medicine_mask,
                wearable, wearable_mask,
                imaging, imaging_mask,
            )

            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)

            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        all_pred = []
        all_label = []

        with torch.no_grad():
            for batch in valid_loader:
                clinical = batch["clinical"].to(device)
                clinical_mask = batch["clinical_mask"].to(device)
                treatment = batch["treatment"].to(device)
                treatment_mask = batch["treatment_mask"].to(device)
                medicine = batch["medicine"].to(device)
                medicine_mask = batch["medicine_mask"].to(device)
                wearable = batch["wearable"].to(device)
                wearable_mask = batch["wearable_mask"].to(device)
                imaging = batch["imaging"].to(device)
                imaging_mask = batch["imaging_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(
                    clinical, clinical_mask,
                    treatment, treatment_mask,
                    medicine, medicine_mask,
                    wearable, wearable_mask,
                    imaging, imaging_mask,
                )

                logits_flat = logits.reshape(-1, logits.size(-1))
                labels_flat = labels.reshape(-1)

                loss = criterion(logits_flat, labels_flat)
                val_loss += loss.item()

                pred = logits.argmax(dim=-1)
                valid_mask = labels != -100
                all_pred.extend(pred[valid_mask].cpu().tolist())
                all_label.extend(labels[valid_mask].cpu().tolist())

        avg_val_loss = val_loss / max(1, len(valid_loader))
        acc = accuracy_score(all_label, all_pred)
        macro_f1 = f1_score(all_label, all_pred, average="macro", zero_division=0)

        print(f"Epoch {epoch + 1:02d} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | acc={acc:.4f} | macro_f1={macro_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ----------------------------
# Main
# ----------------------------
def main():
    print("Generating synthetic multimodal disease progression dataset...")
    df = generate_demo_dataset(n_patients=500, seed=42)

    train_df, valid_df = split_by_patient(df, valid_fraction=0.2, seed=42)

    clinical_cols = CLINICAL_BASE_COLS + TREATMENT_COLS + MEDICINE_COLS
    clinical_stats = compute_group_stats(train_df, clinical_cols)
    wearable_stats = compute_group_stats(train_df, WEARABLE_COLS)
    imaging_stats = compute_group_stats(train_df, IMAGING_COLS)

    train_df = apply_standardization(train_df.copy(), clinical_stats)
    valid_df = apply_standardization(valid_df.copy(), clinical_stats)

    train_df = apply_standardization(train_df, wearable_stats)
    valid_df = apply_standardization(valid_df, wearable_stats)

    train_df = apply_standardization(train_df, imaging_stats)
    valid_df = apply_standardization(valid_df, imaging_stats)

    train_sequences = make_patient_sequences(train_df)
    valid_sequences = make_patient_sequences(valid_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nTraining separate treatment and medicine Transformer model...")
    train_model(train_sequences, valid_sequences, epochs=25, batch_size=16, lr=1e-3, device=device)

    print("\nModel training complete.")
    print("This is a synthetic treatment-aware progression baseline for methodological development.")

if __name__ == "__main__":
    main()
