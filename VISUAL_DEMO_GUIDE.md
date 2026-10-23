# Visual demo guide

The Streamlit interface gives the repository a browser-based visual front end.

## Run locally

From the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your browser.

## Views

- **Interactive demo:** shows the intended workflow with clearly labelled illustrative values. This view does not run the model and its chart is not a prediction.
- **Run trained model:** runs inference using a real checkpoint created by `train_multimodal_raw_image_model.py`, a prediction CSV, and a ZIP archive containing the corresponding scan files.
- **About the model:** explains the architecture, task, and limitations.

## Preparing the prediction manifest (CSV)

The app provides a downloadable example CSV under **Run trained model → New to CSV files? Start here**. Open it in Excel or LibreOffice, replace the placeholder values, and save/export it as CSV.

Use **one row per patient visit and scan**:

- `patient_id`: pseudonymous identifier. Reuse it across that patient's visits; do not put names or direct identifiers here.
- `image_path`: relative path to the scan inside the ZIP archive, such as `scans/P001/visit0.nii.gz`.
- `visit_idx`: visit order for each patient (baseline `0`, next visit `1`, then `2`, etc.).
- `time_months`: optional months since baseline (baseline `0`, then for example `6` and `12`). If omitted, the prediction script falls back to `visit_idx`; this fallback is an ordering convenience, not a real elapsed-time measurement.

The template shows examples of feature columns: `C_` clinical, `T_` treatment, `M_` medicine, and `W_` wearable. These are examples, not a required universal list. Feature names and their meaning must match the schema used to train the checkpoint. Include the columns the model was trained with; unavailable values can be left blank, but do not invent measurements or category values. The prediction script fills missing expected feature columns as unavailable.

Example for a patient with two visits:

| patient_id | image_path | visit_idx | time_months |
|---|---|---:|---:|
| P001 | scans/P001/visit0.nii.gz | 0 | 0 |
| P001 | scans/P001/visit1.nii.gz | 1 | 6 |

The example identifiers and paths are placeholders. Each referenced scan must exist at the exact relative path inside the ZIP. The same patient should not have duplicate rows for the same `visit_idx`.

## Guided data entry in the app

The **Run trained model** view now includes a guided entry form. Upload the checkpoint first so the form can read its saved feature schema. Then add one visit at a time:

1. Enter a pseudonymous patient ID (reuse it for the same patient across visits).
2. Set visit order (`visit_idx`: baseline 0, next visit 1, and so on) and months since baseline (`time_months`: baseline 0).
3. Attach the scan file for that visit. The app generates a relative ZIP path automatically.
4. Fill in the feature fields shown from the checkpoint's saved schema. Prefixes group fields into clinical (`C_`), treatment (`T_`), medicine (`M_`), and wearable (`W_`) sections. Fields without one of these prefixes are shown in the clinical/general section.
5. Leave values blank only when they are genuinely unavailable. For numeric fields, enter numbers using the same units and definitions as the training data; for categorical fields, use category values consistent with training data. Do not guess.
6. Add subsequent visits, review the table, and select **Validate and prepare manifest + scan ZIP**. The app checks duplicate patient/visit indices and non-decreasing elapsed time, then lets you download the generated files or use them directly for inference.

The guided form depends on the checkpoint having usable feature metadata. Older checkpoints without a raw categorical schema may only expose their stored feature columns; use the CSV upload route if those columns are already one-hot encoded or if you need a specific raw feature schema. The form cannot determine whether entered medical values are clinically correct, whether units match training, or whether a scan is appropriate.

## Running real inference in the app

1. Train the multimodal model on an authorised, correctly labelled dataset. The checkpoint is usually `multimodal_raw_results/multimodal_raw_model.pt` (depending on the training output directory).
2. Prepare a prediction CSV with `patient_id`, `image_path`, and `visit_idx`, plus feature columns used during training where available. `stage` is not required for prediction.
3. Make a ZIP archive containing the scan files. Preserve the relative paths used in `image_path`. For example, if the CSV says `scans/P001/visit0.nii.gz`, that exact relative path should exist inside the ZIP.
4. In the app, select **Run trained model** and upload the checkpoint, CSV, and ZIP.
5. Review the output table and per-visit stage-probability chart, then download the prediction CSV if needed.

The app extracts scan files into a temporary workspace, rejects archive paths that escape that workspace, and runs inference on CPU. Large scans may take time; the app imposes a 15-minute inference limit and a 2 GiB uncompressed ZIP limit. For larger studies, use the command-line pipeline on a suitable research workstation.

## Safety and interpretation

- Do not use identifiable patient information unless your approvals and storage arrangements explicitly permit it.
- Do not commit checkpoints, scans, prediction files, or private datasets to a public repository.
- The interactive demo values are placeholders. They are not generated by the trained model.
- The model outputs supplied visit-stage classes; this is not a validated forecast of future progression, a diagnosis, or clinical advice.
- Repository CI currently demonstrates operation on synthetic data. Real-world or clinical performance has not been established.
