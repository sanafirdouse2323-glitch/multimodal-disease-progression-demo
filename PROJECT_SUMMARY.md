# Project Summary

This repository is a small Python/PyTorch portfolio demonstration for a PhD application in machine learning in medicine. It uses **synthetic data only** to demonstrate data preparation, multimodal feature integration, longitudinal sequence modelling, training, and evaluation.

## What the demo does

The generator creates synthetic patient records across non-uniform visit intervals. Each visit contains feature groups for clinical measurements, treatment history, medicine type/dose/frequency/duration and an exposure proxy, wearable measurements, and imaging-derived features.

Some measurements are missing by design. The preprocessing pipeline records which values were observed before imputing missing values. It fits scaling statistics on training patients and applies those same statistics to validation and test patients.

The model encodes the five feature groups, combines their representations, adds a visit-time representation, and uses a Transformer with a causal attention mask and padding mask. It predicts the **current synthetic stage at each visit**, using the current visit's features and earlier visits. It does not predict a future event or time to progression.

## Evaluation

Patients are split into separate training, validation, and test groups. The Transformer is evaluated on held-out test patients and compared with a majority-class baseline, logistic regression, and random forest. The script saves test metrics, training history, a confusion matrix, a training curve, and a run summary in `results/`.

## Limitations

The data-generating process is deliberately simplified. It is not a clinical dataset, and its results cannot establish real-world predictive performance, fairness, clinical utility, or causal treatment effects. The treatment mechanism is illustrative only. A single seeded synthetic run is a learning demonstration, not clinical validation.

See the main README for setup and execution instructions.
