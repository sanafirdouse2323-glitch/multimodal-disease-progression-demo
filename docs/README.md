# Multimodal Disease-Progression Demo

This repository contains a synthetic Python/PyTorch demonstration of multimodal longitudinal stage classification.

## Overview

The synthetic data generator creates patient visits with non-uniform time intervals and five feature groups: clinical measurements, treatment variables, medicine variables, wearable measurements, and imaging-derived features.

Missing observations are represented by explicit observation indicators created before imputation. Patients are split by patient ID into training, validation, and test partitions. Scaling statistics are fitted on training patients only and reused unchanged for the other partitions.

## Model and task

Separate small encoders process each feature group. Their outputs are fused, combined with a representation of visit time, and passed through a Transformer. A causal attention mask prevents a visit from attending to later visits, and a padding mask marks padded sequence positions.

The task is **current synthetic stage classification**: the model predicts the stage at each visit from features available at that visit and earlier visits. This is not future progression forecasting or time-to-event modelling.

## Run

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python demo_disease_progression.py
```

The script writes held-out test metrics, training history, a confusion matrix, a training curve, and a JSON run summary to `results/`.

## Limitations

All data are synthetic. This project demonstrates implementation and evaluation workflow only; it does not establish clinical validity or support claims about real treatment effects. See the main README for further details.
