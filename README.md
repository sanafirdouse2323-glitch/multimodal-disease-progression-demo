# Multimodal Disease Progression Demo

A synthetic, reproducible prototype for modelling disease progression from multimodal health data, including clinical records, wearable time-series measurements, imaging-derived features, and time-varying treatment exposure.

This repository is designed as a research prototype and PhD-style demo project. It demonstrates a synthetic setting that reflects common challenges in longitudinal clinical data, including irregular visit schedules, missing data, and evolving treatment exposure.

## Motivation

Chronic diseases often develop gradually over years, and their trajectories vary substantially between individuals. Understanding this progression is essential for early detection, personalised intervention, and improved long-term patient outcomes. Large multimodal health datasets that combine medical imaging, wearable sensor data, and clinical measurements now make it possible to study disease progression from multiple complementary perspectives.

A key challenge is that disease progression is usually influenced not only by observed biomarkers, but also by treatment decisions and medication exposure over time. Treatment can modify the apparent progression trajectory, and different medicine types may have different effects on disease dynamics. In a realistic longitudinal setting, treatment depends on more than a binary flag: the dose amount, frequency, duration, and cumulative exposure are all relevant to progression.

## Research question

How can we model disease progression over time using multimodal health data when patient visits are irregular, observations are incomplete, and treatment exposure changes across follow-up?

## Project goal

This project implements a synthetic benchmark for multimodal disease progression modelling. The objective is to learn a temporal representation of each patient’s disease state and predict disease stage from a combination of clinical, wearable, imaging-derived, and treatment-related features.

## Dataset characteristics

The synthetic data generator creates a realistic longitudinal patient dataset with:

- patient-level trajectories across multiple visits
- irregular time intervals between visits
- disease stage labels from 0 to 3
- clinical variables
- wearable measurements
- imaging-derived features
- treatment indicators and medication information
- dose, frequency, duration, and cumulative exposure variables
- missing values in each modality

This makes the dataset suitable as a proof-of-concept benchmark for disease progression research.

## Treatment-aware design

The current prototype explicitly includes treatment variables, including:

- treatment flag
- time since treatment started
- treatment duration in days
- treatment change indicator
- medicine type
- dose per day
- frequency per day
- cumulative exposure

These are treated as time-varying clinical covariates that can influence disease trajectory. This makes the synthetic benchmark more realistic than a model that only includes imaging, wearables, and generic clinical variables.

## Model overview

The demo implements a multimodal temporal model with the following structure:

1. modality-specific encoders for clinical, treatment, medicine, wearable, and imaging inputs
2. separate treatment and medicine streams to preserve treatment-specific structure
3. multimodal fusion into a shared latent representation
4. a Transformer encoder for temporal modelling
5. a stage-classification head
6. robustness to missing values through modality masks

Although simple, this design reflects a realistic methodological foundation for longitudinal multimodal health modelling.

## Repository contents

- `demo_disease_progression.py` — synthetic data generation, preprocessing, treatment-aware modelling, and training pipeline
- `requirements.txt` — Python dependencies
- `docs/README.md` — short project documentation
- `README.md` — project overview and usage instructions
- `PROJECT_SUMMARY.md` — concise project summary

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

- generate a synthetic multimodal dataset with treatment and medicine exposure history
- simulate irregular visits and missingness
- build patient-level sequences
- train the treatment-aware Transformer model
- print training and validation metrics

## Example output

The script prints metrics such as:

```text
Epoch 01 | train_loss=... | val_loss=... | acc=... | macro_f1=...
Epoch 02 | train_loss=... | val_loss=... | acc=... | macro_f1=...
...
```

## Use cases

This prototype is useful for:

- methodological demonstrations in a thesis or dissertation
- introductory teaching on multimodal health modelling
- benchmarking longitudinal disease progression pipelines
- extending to real clinical datasets
- developing stronger transformer-based or survival-based models
- exploring treatment-aware progression modelling

## Extension ideas

This demo can be extended in several ways:

- replace the current sequence model with a more expressive temporal Transformer architecture
- use cross-attention between treatment, medicine, and other modalities
- add time-to-event prediction for progression risk
- model treatment effects explicitly through causal or counterfactual approaches
- incorporate uncertainty estimation and calibration analysis
- validate on real chronic disease datasets

## Limitations

This is a synthetic demonstration project and is not intended to represent a real medical dataset or clinical decision system. It is designed for reproducible method development, research exploration, and thesis/demo use only.

## Summary

This repository demonstrates a realistic synthetic foundation for disease progression modelling using multimodal longitudinal data, including treatment and medicine exposure variables. It captures several central technical challenges of the field: irregular follow-up, missing observations, multimodal integration, evolving treatment exposure, and temporal disease dynamics.

It is appropriate as a proof-of-concept baseline for a PhD project, research proposal, or demo presentation.

## License

This project is provided for demonstration and research use.
