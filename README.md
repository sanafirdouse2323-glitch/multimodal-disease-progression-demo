# Multimodal Disease-Progression Demo

A small **Python/PyTorch portfolio demo** for learning how to work with synthetic longitudinal health data. It combines clinical measurements, wearable signals, imaging-derived features, and treatment/medicine history, then predicts a patient's **current synthetic disease stage** from the current and earlier visits.

This project is intended to demonstrate programming, data preparation, model training, and evaluation skills for a PhD application. It is **not a clinical tool**, does not use real patient records, and does not establish clinical performance.

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
- SimpleITK for the optional raw-image pipeline
- Streamlit for the optional browser interface

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

## Run the project in Replit

This branch configures Replit's **Run** button to execute the core Python project with `python demo_disease_progression.py`. Streamlit is optional and is not required to train or evaluate this core synthetic demo.

1. In [Replit](https://replit.com/), choose **Import from GitHub** and import [this repository](https://github.com/sanafirdouse2323-glitch/multimodal-disease-progression-demo).
2. In the Replit Shell, switch to the development branch so you use this configuration and the separate CI workflows:

   ```bash
   git fetch origin
   git switch test/streamlit-browser-workflow
   ```

3. Install dependencies in the Shell. The CPU-only PyTorch command is suitable for a CPU workspace:

   ```bash
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements.txt
   ```

4. Click **Run**. It should run the core synthetic Python demo and write its results to `results/`.

### Optional: open the Streamlit interface

To use the browser interface, run this in the Replit Shell instead of the core demo:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Open the Replit Webview when Streamlit starts. This interface is optional; it is a separate way to interact with the project, not the training entry point. The guided file-upload/browser-inference flow is still under separate validation, so do not assume that interface is fully verified just because the core Python workflow passes. Use synthetic files while testing.

## Run the raw-image pipeline end to end (synthetic smoke test)

The repository has two separate raw-image training paths:

- `train_raw_image_model.py` trains an image-only 3D CNN.
- `train_multimodal_raw_image_model.py` trains a raw-image + tabular longitudinal model. `predict_multimodal_raw_image_model.py` runs inference from that model's saved checkpoint.

Both need 3D image files and manifest CSVs. To verify the pipeline without having a medical dataset ready, run:

```bash
python run_synthetic_raw_image_demo.py
```

This script generates artificial NIfTI volumes and valid manifests for 20 synthetic subjects with two visits each. It then trains and evaluates both raw-image models for one epoch and runs two-visit inference using the multimodal checkpoint. It checks that the code paths execute; the synthetic metrics are **not** evidence of clinical accuracy or real-world performance.

By default, generated data and results are saved in `raw_image_smoke_demo/`:

- `scans/` — synthetic NIfTI files
- `training_manifest.csv` and `prediction_manifest.csv` — manifests for training and inference
- `image_only_results/` — image-only checkpoint, training history, and run summary
- `multimodal_results/` — multimodal checkpoint, training history, and run summary
- `predictions.csv` — two-visit prediction output

The script regenerates its own `SYNTH_...` scan files and manifests each time it is run. This 3D-CNN smoke test takes longer and uses more CPU than the core demo, so the default Replit Run button intentionally does not run it.

### Train on an authorised dataset

For real research data, do not use the synthetic demo labels or measurements. Prepare an accurate manifest using verified labels and the actual image paths for your authorised dataset, then follow [RAW_IMAGE_GUIDE.md](RAW_IMAGE_GUIDE.md). For the longitudinal multimodal model, the basic command is:

```bash
python train_multimodal_raw_image_model.py --manifest multimodal_raw_manifest.csv --modality generic
```

The manifest needs one row per patient visit, including `patient_id`, `image_path`, integer `stage` labels from 0 to 3, and `visit_idx`, plus at least one supported tabular feature. Every referenced scan must exist. The script uses patient-level train/validation/test splits. Its output describes performance only on the supplied labelled dataset and does not establish clinical validity.

## Run core tests without the browser interface

On Replit/Linux, run the unit and CLI integration tests without launching Streamlit or requiring Playwright/Chromium:

```bash
SKIP_STREAMLIT_BROWSER_TEST=1 python -m unittest discover -s tests -v
```

In Windows PowerShell, set the environment variable first:

```powershell
$env:SKIP_STREAMLIT_BROWSER_TEST = "1"
python -m unittest discover -s tests -v
```

The raw-image training/checkpoint/prediction tests still run. Only the extra browser automation portion is skipped. The Streamlit browser test is a separate workflow and requires Playwright plus Chromium.

## Main implementation

- `demo_disease_progression.py` — synthetic tabular/longitudinal data generation, preprocessing, model, baselines, evaluation, and plots
- `train_raw_image_model.py` — image-only raw-volume 3D CNN
- `train_multimodal_raw_image_model.py` — raw-image + tabular longitudinal model
- `predict_multimodal_raw_image_model.py` — checkpoint-based multimodal per-visit inference
- `run_synthetic_raw_image_demo.py` — generates toy NIfTI scans and exercises both raw-image training paths
- `tests/` — unit, smoke, and CLI integration tests
- `requirements.txt` — dependencies

## Limitations

- Synthetic data only; no real-world or clinical validation.
- The data-generating process is intentionally simplified and may not capture real disease biology or care pathways.
- The treatment process is simulated and cannot support claims about real treatment effects.
- The current core task is stage classification, not survival analysis or future progression forecasting.
- The raw-image scripts classify supplied visit-stage labels; they do not infer or verify labels and do not establish future progression.
- A single seeded run is for demonstration; stronger evaluation would require repeated seeds and real datasets with appropriate access, governance, and outcome definitions.
- Raw-image normalization, orientation, target spacing, and center crop/padding must be reviewed for the specific scan modality, anatomy, and task.

## Direct raw CT/MRI image input (research starter)

An optional 3D CNN starter can read raw 3D NIfTI scans (`.nii`, `.nii.gz`) or a directory containing one DICOM series, instead of requiring hand-crafted imaging feature columns. It reorients scans to LPS and resamples them to configurable isotropic spacing before center crop/padding to a fixed input shape. It requires a manifest containing verified `patient_id`, `image_path`, and integer `stage` labels (0–3). It predicts the supplied label from each scan; it does not generate labels and it is not clinically validated.

See [RAW_IMAGE_GUIDE.md](RAW_IMAGE_GUIDE.md) for the image-only baseline, preprocessing, and limitations. For longitudinal multimodal fusion, `train_multimodal_raw_image_model.py` combines raw scan embeddings from a 3D CNN with clinical, treatment, medicine, and wearable features. Numeric measurements are standardized, and categorical text such as sex or medicine names is automatically one-hot encoded using categories learned from training patients only. Missingness masks are retained before a causal Transformer processes visits. Start from [multimodal_raw_manifest_template.csv](multimodal_raw_manifest_template.csv); see the guide for exact columns and a smoke-run command. A companion `predict_multimodal_raw_image_model.py` CLI loads the saved checkpoint and writes per-visit stage probabilities. This is a research baseline and has not been clinically validated.

## Visual interactive interface

A Streamlit web interface is available in `app.py`. Run `python -m pip install -r requirements.txt` followed by `python -m streamlit run app.py` to open it in your browser. It includes an explicitly illustrative demo view, a trained-checkpoint inference view, and model/limitations information. See [VISUAL_DEMO_GUIDE.md](VISUAL_DEMO_GUIDE.md) for setup and input-file requirements. The illustrative demo does not run the model; real inference requires a trained checkpoint, a prediction manifest, and the corresponding scan files.

## License

No open-source license has been selected yet. Until one is added, do not assume the repository is licensed for reuse beyond rights granted by applicable law.
