import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

np.random.seed(42)

CLINICAL_COLS = ["lab1", "lab2", "vitals", "meds", "comorbidities"]
WEARABLE_COLS = ["steps", "sleep_hours", "heart_rate", "hrv", "activity"]
IMAGING_COLS = ["imaging_score", "lesion_area", "atrophy_index", "tissue_change"]


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

        for visit_idx, months in enumerate(visit_times):
            latent = baseline_risk + progression_rate * months + rng.normal(0.0, 0.7)

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
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "baseline_risk": baseline_risk,
                "progression_rate": progression_rate,
                "latent_score": latent,
                "stage": stage,
                "lab1": lab1,
                "lab2": lab2,
                "vitals": vitals,
                "meds": meds,
                "comorbidities": comorbidities,
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

    for col in CLINICAL_COLS:
        mask = rng.random(len(df)) < clinical_missing.to_numpy()
        df.loc[mask, col] = np.nan

    for col in WEARABLE_COLS:
        mask = rng.random(len(df)) < wearable_missing.to_numpy()
        df.loc[mask, col] = np.nan

    for col in IMAGING_COLS:
        mask = rng.random(len(df)) < imaging_missing.to_numpy()
        df.loc[mask, col] = np.nan

    return df


def make_patient_sequences(df):
    patient_sequences = []
    for _, g in df.groupby("patient_id"):
        g = g.sort_values("visit_idx").reset_index(drop=True)

        clinical = g[CLINICAL_COLS].fillna(0.0).to_numpy(dtype=np.float32)
        wearable = g[WEARABLE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
        imaging = g[IMAGING_COLS].fillna(0.0).to_numpy(dtype=np.float32)

        clinical_mask = g[CLINICAL_COLS].notna().astype(float).to_numpy(dtype=np.float32)
        wearable_mask = g[WEARABLE_COLS].notna().astype(float).to_numpy(dtype=np.float32)
        imaging_mask = g[IMAGING_COLS].notna().astype(float).to_numpy(dtype=np.float32)
        labels = g["stage"].to_numpy(dtype=np.int64)

        patient_sequences.append(
            {
                "clinical": clinical,
                "clinical_mask": clinical_mask,
                "wearable": wearable,
                "wearable_mask": wearable_mask,
                "imaging": imaging,
                "imaging_mask": imaging_mask,
                "label": labels,
            }
        )

    return patient_sequences


class PatientSequenceDataset(Dataset):
    def __init__(self, patient_sequences):
        self.data = patient_sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_patient_sequences(batch):
    max_len = max(item["clinical"].shape[0] for item in batch)

    clinical_list = []
    clinical_mask_list = []
    wearable_list = []
    wearable_mask_list = []
    imaging_list = []
    imaging_mask_list = []
    labels_list = []

    for item in batch:
        seq_len = item["clinical"].shape[0]

        def pad_2d(arr, pad_len):
            if arr.shape[0] == pad_len:
                return arr
            pad = np.zeros((pad_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
            return np.concatenate([arr, pad], axis=0)

        clinical_list.append(pad_2d(item["clinical"].astype(np.float32), max_len))
        clinical_mask_list.append(pad_2d(item["clinical_mask"].astype(np.float32), max_len))
        wearable_list.append(pad_2d(item["wearable"].astype(np.float32), max_len))
        wearable_mask_list.append(pad_2d(item["wearable_mask"].astype(np.float32), max_len))
        imaging_list.append(pad_2d(item["imaging"].astype(np.float32), max_len))
        imaging_mask_list.append(pad_2d(item["imaging_mask"].astype(np.float32), max_len))
        labels_list.append(np.pad(item["label"], (0, max_len - seq_len), constant_values=-100))

    return {
        "clinical": torch.tensor(np.stack(clinical_list), dtype=torch.float32),
        "clinical_mask": torch.tensor(np.stack(clinical_mask_list), dtype=torch.float32),
        "wearable": torch.tensor(np.stack(wearable_list), dtype=torch.float32),
        "wearable_mask": torch.tensor(np.stack(wearable_mask_list), dtype=torch.float32),
        "imaging": torch.tensor(np.stack(imaging_list), dtype=torch.float32),
        "imaging_mask": torch.tensor(np.stack(imaging_mask_list), dtype=torch.float32),
        "labels": torch.tensor(np.stack(labels_list), dtype=torch.long),
    }


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


class MultimodalProgressionModel(nn.Module):
    def __init__(self, clinical_dim, wearable_dim, imaging_dim, hidden_dim=32, n_classes=4):
        super().__init__()
        self.clinical_encoder = ModalityEncoder(clinical_dim, hidden_dim)
        self.wearable_encoder = ModalityEncoder(wearable_dim, hidden_dim)
        self.imaging_encoder = ModalityEncoder(imaging_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, clinical, clinical_mask, wearable, wearable_mask, imaging, imaging_mask):
        clinical_emb = self.clinical_encoder(clinical * clinical_mask)
        wearable_emb = self.wearable_encoder(wearable * wearable_mask)
        imaging_emb = self.imaging_encoder(imaging * imaging_mask)

        fused = torch.cat([clinical_emb, wearable_emb, imaging_emb], dim=-1)
        fused = self.fusion(fused)

        out, _ = self.lstm(fused)
        logits = self.classifier(out)
        return logits


def train_model(train_sequences, valid_sequences, epochs=25, batch_size=16, lr=1e-3):
    train_dataset = PatientSequenceDataset(train_sequences)
    valid_dataset = PatientSequenceDataset(valid_sequences)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_patient_sequences)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_patient_sequences)

    model = MultimodalProgressionModel(
        clinical_dim=len(CLINICAL_COLS),
        wearable_dim=len(WEARABLE_COLS),
        imaging_dim=len(IMAGING_COLS),
        hidden_dim=32,
        n_classes=4,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            clinical = batch["clinical"]
            clinical_mask = batch["clinical_mask"]
            wearable = batch["wearable"]
            wearable_mask = batch["wearable_mask"]
            imaging = batch["imaging"]
            imaging_mask = batch["imaging_mask"]
            labels = batch["labels"]

            logits = model(clinical, clinical_mask, wearable, wearable_mask, imaging, imaging_mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)

            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        all_pred = []
        all_label = []

        with torch.no_grad():
            for batch in valid_loader:
                clinical = batch["clinical"]
                clinical_mask = batch["clinical_mask"]
                wearable = batch["wearable"]
                wearable_mask = batch["wearable_mask"]
                imaging = batch["imaging"]
                imaging_mask = batch["imaging_mask"]
                labels = batch["labels"]

                logits = model(clinical, clinical_mask, wearable, wearable_mask, imaging, imaging_mask)
                logits_flat = logits.reshape(-1, logits.size(-1))
                labels_flat = labels.reshape(-1)

                val_loss += criterion(logits_flat, labels_flat).item()

                pred = logits.argmax(dim=-1)
                valid_mask = labels != -100
                all_pred.extend(pred[valid_mask].cpu().tolist())
                all_label.extend(labels[valid_mask].cpu().tolist())

        avg_val_loss = val_loss / max(1, len(valid_loader))
        acc = accuracy_score(all_label, all_pred)
        macro_f1 = f1_score(all_label, all_pred, average="macro")

        print(f"Epoch {epoch + 1:02d} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | acc={acc:.4f} | macro_f1={macro_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def main():
    print("Generating synthetic multimodal disease progression dataset...")
    df = generate_demo_dataset(n_patients=500, seed=42)

    print("\nDataset preview:")
    print(df.head())
    print("\nDisease stage distribution:")
    print(df["stage"].value_counts().sort_index())
    print("\nMissing values:")
    print(df.isna().sum())

    patient_ids = sorted(df["patient_id"].unique())
    rng = np.random.default_rng(42)
    rng.shuffle(patient_ids)

    split_idx = int(0.8 * len(patient_ids))
    train_ids = set(patient_ids[:split_idx])
    valid_ids = set(patient_ids[split_idx:])

    train_df = df[df["patient_id"].isin(train_ids)].copy()
    valid_df = df[df["patient_id"].isin(valid_ids)].copy()

    print(f"\nTrain patients: {len(train_ids)}")
    print(f"Validation patients: {len(valid_ids)}")

    train_sequences = make_patient_sequences(train_df)
    valid_sequences = make_patient_sequences(valid_df)

    print("\nTraining multimodal progression model...")
    model = train_model(train_sequences, valid_sequences, epochs=25, batch_size=16, lr=1e-3)

    print("\nModel training complete.")
    print("Prototype ready for extension to transformer-based designs or real clinical data.")


if __name__ == "__main__":
    main()
