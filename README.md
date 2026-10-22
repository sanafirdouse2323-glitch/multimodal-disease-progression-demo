# Multimodal Disease Progression Demo

A synthetic, reproducible demo project for modelling disease progression from multimodal health data, including clinical records, wearable time-series measurements, and imaging-derived features.

This repository is designed as a research prototype and thesis/demo project. It demonstrates a realistic workflow for longitudinal disease modelling under irregular visit schedules and missing data, which are common challenges in real-world clinical datasets.

## Motivation

Disease progression is typically influenced by multiple sources of information that evolve over time. In many chronic conditions, a patient’s health state is reflected in:

- clinical measurements and laboratory tests
- wearable-derived physiological and behavioural signals
- imaging-derived biomarkers and structural changes

A model that integrates these modalities can provide a richer and more clinically informative representation of progression than any single modality alone.

## Research question

How can we model disease progression over time using multimodal health data when patient visits are irregular and some observations are missing?

## Project goal

This project implements a synthetic benchmark for multimodal disease progression modelling. The objective is to learn a temporal representation of each patient’s disease state and predict disease stage from a combination of clinical, wearable, and imaging-derived features.

## Dataset characteristics

The synthetic data generator creates a realistic longitudinal patient dataset with:

- patient-level trajectories across multiple visits
- irregular time intervals between visits
- disease stage labels from 0 to 3
- clinical variables
- wearable measurements
- imaging-derived features
- missing values in each modality

This makes the dataset suitable as a proof-of-concept benchmark for disease progression research.

## Model overview

The demo implements a multimodal temporal model with the following structure:

1. Modality-specific encoders for clinical, wearable, and imaging inputs
2. Fusion of modality embeddings into a shared latent representation
3. Temporal modelling through an LSTM sequence model
4. Prediction of disease stage at each visit
5. Robustness to missing values by using modality masks

Although simple, this design reflects a realistic methodological foundation for longitudinal multimodal health modelling.

## Repository contents

- `demo_disease_progression.py` — synthetic data generation, preprocessing, model definition, and training pipeline
- `requirements.txt` — Python dependencies
- `README.md` — project overview and usage instructions

## Requirements

The project uses the following Python libraries:

- Python 3.9+
- PyTorch
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Setup

Clone the repository:

```bash
git clone https://github.com/sanafirdouse2323-glitch/multimodal-disease-progression-demo.git
cd multimodal-disease-progression-demo
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the demo

```bash
python demo_disease_progression.py
```

This script will:

- generate a synthetic multimodal dataset
- simulate irregular visits and missingness
- build patient-level sequences
- train the disease progression model
- print training loss and validation accuracy

## Example output

The script prints training diagnostics such as:

```text
Epoch 01 | train_loss=... | val_loss=... | acc=...
Epoch 02 | train_loss=... | val_loss=... | acc=...
...
```

## Use cases

This prototype is useful for:

- methodological demonstrations in a thesis or dissertation
- introductory teaching on multimodal health modelling
- benchmarking a longitudinal disease progression pipeline
- extending to real clinical datasets
- developing stronger transformer-based or survival-based models

## Extension ideas

This demo can be extended in several ways:

- replace LSTM with a Temporal Transformer
- use cross-attention multimodal fusion
- add time-to-event prediction for disease progression
- model missingness more explicitly with learned masks
- validate on real chronic disease datasets
- incorporate uncertainty estimation and calibration analysis

## Limitations

This is a synthetic demonstration project and is not intended to represent a real medical dataset or clinical decision system. It is designed for reproducible method development, research exploration, and thesis/demo use.

## Summary

This repository demonstrates a realistic foundation for disease progression modelling using multimodal longitudinal data. It captures the central technical challenges of the field: irregular follow-up, missing observations, multimodal integration, and temporal disease dynamics.

It is appropriate as a proof-of-concept baseline for a PhD project, research proposal, or demo presentation.

## License

This project is provided for demonstration and research use.
