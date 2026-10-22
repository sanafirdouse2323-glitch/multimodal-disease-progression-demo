# Multimodal Disease Progression Demo

This repository contains a synthetic proof-of-concept prototype for modelling disease progression using multimodal health data.

## Overview

The prototype simulates a realistic health-data setting in which patient trajectories are represented by:

- clinical records
- wearable time-series signals
- imaging-derived features
- irregular visit schedules
- missing observations

At each visit, the model predicts the patient’s disease stage. This creates a realistic and reproducible benchmark for disease trajectory modelling.

## Dataset

The synthetic dataset includes patient trajectories spanning multiple irregular visits. The disease stage increases with patient-specific latent risk and time, while each modality is generated to reflect disease progression.

## Model

The model uses:

1. modality-specific encoders for clinical, wearable, and imaging data
2. multimodal fusion into a shared latent representation
3. an LSTM to model longitudinal disease dynamics
4. a stage-classification head

The design is intentionally compact but realistic enough to support a PhD demo or research prototype.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo_disease_progression.py
```

## Expected outcome

The script generates synthetic data, trains a multimodal disease progression model, and prints training/validation metrics such as loss, accuracy, and macro F1 score.

## Notes

This is a synthetic demonstration project and is intended for research prototyping and methodological exploration. It is not a clinical decision-support system.
