# Multimodal Disease-Progression Demo

A small **Python/PyTorch portfolio demo** for learning how to work with synthetic longitudinal health data. It combines clinical measurements, wearable signals, imaging-derived features, and treatment/medicine history, then predicts a patient's **current synthetic disease stage** from the current and earlier visits.

This project is a Python/PyTorch portfolio prototype intended to demonstrate data preparation, temporal model implementation, baseline comparison, testing, and evaluation practices relevant to research in machine learning for medicine. It is **not a clinical tool**, does not use real patient records, and does not establish clinical performance.

## What it demonstrates

- Synthetic patient trajectories with non-uniform visit intervals.
- Five feature groups: clinical, treatment, medicine, wearable, and imaging-derived features.
- Explicit observation indicators so the model can distinguish imputed values from observed values.
- Patient-level train/validation/test splits.
- Training-only feature standardisation; validation and test data do not fit preprocessing statistics.
- A causal attention mask so a prediction at a visit cannot attend to later visits.
- A padding mask so padded visits do not contribute as genuine observations.
- Comparison with a majority-class baseline, logistic regression, and random forest.
- Held-out test metrics, a confusion matrix, a training curve, and a JSON run summary.
- A fixed random seed for a reproducible starting point.

## Relevance to trustworthy machine learning for medicine

The project provides a small, synthetic setting in which to practise longitudinal modelling, multimodal feature fusion, missing-data handling, and evaluation of both classification and probability calibration. These are relevant methodological themes for research in machine learning for medicine, but this repository is not a complete implementation of any specific PhD project's research agenda. In particular, it does not establish causal effects, real-world robustness, subgroup fairness, uncertainty reliability, or clinical utility.

## Important interpretation

All data and labels are simulated. The stage is generated from a synthetic latent state, and measurements are generated from that state with noise. This makes the task useful for a code demonstration, but it is much simpler than real clinical data. The reported scores measure performance only on this particular synthetic generator; they should not be interpreted as evidence of clinical usefulness, fairness, causal treatment effects, or generalisation to real patients.

The model predicts the **current stage at each visit** using that visit's available features and the patient's earlier visits. It is not a future-event or time-to-progression model.

## Requirements

- Python 3.10 or later recommended
- PyTorch
- NumPy
- pandas
- scikit-learn
- matplotlib

## Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/sanafirdouse2323-glitch/multimodal-disease-progression-demo.git
cd multimodal-disease-progression-demo
python -m venv .venv
```

Activate the environment:

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python demo_disease_progression.py
```

The script prints held-out test metrics and saves the following files under `results/`:

- `test_metrics.csv` — test accuracy and macro-F1 for all baselines and the Transformer
- `training_history.csv` — training and validation loss and validation metrics by epoch
- `training_curve.png` — training/validation loss plot
- `transformer_confusion_matrix.png` — test-set confusion matrix
- `run_summary.json` — run configuration, split sizes, metrics, and interpretation note

Results are generated when you run the script; no scores are claimed in advance.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Main implementation

- `demo_disease_progression.py` — synthetic data generation, preprocessing, sequence batching, model, baselines, evaluation, and plots
- `tests/test_demo.py` — small checks for the data pipeline, patient splits, padding, and model output
- `requirements.txt` — dependencies

## Limitations

- Synthetic data only; no real-world or clinical validation.
- The data-generating process is intentionally simplified and may not capture real disease biology or care pathways.
- The treatment process is simulated and cannot support claims about real treatment effects.
- The current task is stage classification, not survival analysis or future progression forecasting.
- A single seeded run is for demonstration; stronger evaluation would require repeated seeds and real datasets with appropriate access, governance, and outcome definitions.

## Direct raw CT/MRI image input (research starter)

An optional 3D CNN starter can read raw 3D NIfTI scans (`.nii`, `.nii.gz`) or a directory containing one DICOM series, instead of requiring hand-crafted imaging feature columns. It reorients scans to LPS and resamples them to configurable isotropic spacing before center crop/padding to a fixed input shape. It requires a manifest containing verified `patient_id`, `image_path`, and integer `stage` labels (0–3). It predicts the supplied label from each scan; it does not generate labels and it is not clinically validated.

See [RAW_IMAGE_GUIDE.md](RAW_IMAGE_GUIDE.md) for the image-only baseline, preprocessing, and limitations. For longitudinal multimodal fusion, `train_multimodal_raw_image_model.py` combines raw scan embeddings from a 3D CNN with clinical, treatment, medicine, and wearable features. Numeric measurements are standardized, and categorical text such as sex or medicine names is automatically one-hot encoded using categories learned from training patients only. Missingness masks are retained before a causal Transformer processes visits. Start from [multimodal_raw_manifest_template.csv](multimodal_raw_manifest_template.csv); see the guide for exact columns and a smoke-run command. A companion `predict_multimodal_raw_image_model.py` CLI loads the saved checkpoint and writes per-visit stage probabilities. This is a research baseline and has not been clinically validated.

## Visual interactive interface

A Streamlit web interface is available in `app.py`. Run `python -m pip install -r requirements.txt` followed by `streamlit run app.py` to open it in your browser. It includes an explicitly illustrative demo view, a trained-checkpoint inference view, and model/limitations information. See [VISUAL_DEMO_GUIDE.md](VISUAL_DEMO_GUIDE.md) for setup and input-file requirements. The illustrative demo does not run the model; real inference requires a trained checkpoint, a prediction manifest, and the corresponding scan files.

## Research-inspired whole-feature-group missingness experiment

For a focused experiment related to missing-data robustness in longitudinal learning, see [the whole-feature-group robustness experiment](experiments/README.md). It compares standard training with stochastic feature-group dropout on synthetic longitudinal data, evaluates complete inputs and each single-feature-group-missing condition, and reports classification and calibration metrics. The groups are not five independent sensing modalities, and the experiment is not a reproduction of the cited papers or a clinical validation study.

## License

No open-source license has been selected yet. Until one is added, do not assume the repository is licensed for reuse beyond rights granted by applicable law.
