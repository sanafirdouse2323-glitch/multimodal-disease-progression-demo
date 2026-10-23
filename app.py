"""Interactive Streamlit front end for the multimodal research prototype.

Demo mode uses illustrative values only. Real inference requires a trained checkpoint,
a prediction manifest, and an archive of the corresponding authorised scan files.
"""
from __future__ import annotations

import io
import os
import subprocess
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Multimodal Disease-Progression Demo", page_icon="🧠", layout="wide")

st.title("Multimodal Disease-Progression Demo")
st.caption("Research prototype • raw medical images + clinical, treatment, medicine and wearable data")
st.warning(
    "Research demonstration only. This is not a diagnostic tool. Demo-mode values are illustrative, "
    "not model predictions. Real-data performance and clinical usefulness have not been established."
)

with st.sidebar:
    st.header("Navigate")
    mode = st.radio("Choose a view", ["Interactive demo", "Run trained model", "About the model"])
    st.divider()
    st.caption("Keep identifiable patient data off public services. Use only data you are authorised to process.")

if mode == "Interactive demo":
    st.subheader("Explore the workflow")
    st.info(
        "This view demonstrates the interface and data flow. It does not load a trained checkpoint "
        "and does not calculate predictions from the values entered here."
    )
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown("### 1. Visit information")
        patient_id = st.text_input("Pseudonymous patient ID", value="DEMO-001")
        visit_idx = st.number_input("Visit index", min_value=0, value=0, step=1)
        time_months = st.number_input("Months from baseline", min_value=0.0, value=0.0, step=1.0)
        st.markdown("### 2. Example feature values")
        a, b = st.columns(2)
        with a:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=45)
            sex = st.selectbox("Sex (example category)", ["Not specified", "Female", "Male", "Other"])
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, value=24.1, step=0.1)
            steps = st.number_input("Steps per day", min_value=0, value=6500, step=500)
        with b:
            hemoglobin = st.number_input("Hemoglobin (example units)", min_value=0.0, value=12.5, step=0.1)
            treatment_days = st.number_input("Treatment duration (days)", min_value=0, value=180, step=30)
            medicine = st.selectbox("Medicine (example category)", ["Not specified", "Metformin", "Insulin", "Other"])
            sleep = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        st.caption("These inputs are for interface demonstration only and are not sent to a model.")
    with right:
        st.markdown("### 3. Example output layout")
        st.metric("Prediction", "Not run", help="No trained model is executed in interactive demo mode.")
        st.write("A real run will display probabilities returned by your trained checkpoint.")
        example = pd.DataFrame({
            "Stage": ["Stage 0", "Stage 1", "Stage 2", "Stage 3"],
            "Illustrative probability": [0.12, 0.23, 0.48, 0.17],
        })
        st.bar_chart(example.set_index("Stage"))
        st.caption("Illustrative chart only — values are placeholders, not a prediction.")
        st.markdown("**Feature groups**")
        st.write("Clinical · Treatment · Medicine · Wearable · 3D scan")
        st.markdown("**Current model output**")
        st.write("Stage probabilities, confidence, patient/visit identifiers.")
    with st.expander("Entered example values (not submitted to a model)"):
        st.json({
            "patient_id": patient_id, "visit_idx": visit_idx, "time_months": time_months,
            "C_age": age, "C_sex": sex, "C_bmi": bmi, "C_hemoglobin": hemoglobin,
            "T_treatment_duration_days": treatment_days, "M_medicine_name": medicine,
            "W_steps": steps, "W_sleep_hours": sleep,
        })

elif mode == "Run trained model":
    st.subheader("Run inference with your trained checkpoint")
    st.markdown(
        "Upload a checkpoint produced by `train_multimodal_raw_image_model.py`, a CSV manifest, "
        "and a ZIP archive containing the scans at the relative paths referenced by that CSV."
    )

    # Beginner-friendly manifest builder: the example row is a template, not real patient data.
    st.markdown("### New to CSV files? Start here")
    st.write(
        "A manifest is a spreadsheet saved as CSV: **one row per patient visit and scan**. "
        "You do not need to enter every possible feature. Start with the required columns, "
        "then fill in the feature columns your trained model expects. Leave genuinely unavailable "
        "feature values blank rather than guessing."
    )
    template_columns = [
        "patient_id", "image_path", "visit_idx", "time_months",
        "C_age", "C_sex", "C_bmi", "C_hemoglobin", "C_comorbidities_count",
        "T_treatment_flag", "T_treatment_duration_days", "T_treatment_change",
        "M_medicine_name", "M_dose_per_day", "M_frequency_per_day", "M_cumulative_exposure",
        "W_steps", "W_sleep_hours", "W_heart_rate", "W_hrv", "W_activity",
    ]
    template_row = {
        "patient_id": "REPLACE_WITH_PSEUDONYMOUS_ID",
        "image_path": "scans/REPLACE_WITH_ID/visit0.nii.gz",
        "visit_idx": 0,
        "time_months": 0,
    }
    template_df = pd.DataFrame([{col: template_row.get(col, "") for col in template_columns}])
    st.download_button(
        "Download example CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_manifest_template.csv",
        mime="text/csv",
        help="Open this file in Excel or LibreOffice, replace the example values, then save as CSV.",
    )
    with st.expander("What goes in each column?"):
        st.markdown("""
        - **patient_id** — a pseudonymous ID, reused for all visits belonging to the same person. Do not use names.
        - **image_path** — path to that visit's scan *inside the ZIP*, for example `scans/P001/visit0.nii.gz`.
        - **visit_idx** — visit order for each patient: baseline = 0, next visit = 1, next = 2. Do not use a calendar date here.
        - **time_months** — months elapsed since that patient's baseline visit (baseline = 0, then e.g. 6 or 12). If omitted, the app uses `visit_idx` as a fallback.
        - **C_ columns** — clinical data, such as age, sex, BMI, or lab measurements.
        - **T_ columns** — treatment data, such as treatment duration or treatment changes.
        - **M_ columns** — medicine data, such as medicine name, dose, or frequency.
        - **W_ columns** — wearable data, such as steps, sleep, or heart rate.
        
        Use one row per visit, not one row per feature group. If a patient has three visits, create three rows with the same `patient_id`, visit indices 0, 1, and 2, and the corresponding scan path in each row. Feature column names must match the model's training schema; this template is an example, not a universal clinical data standard.
        """)
    with st.expander("Example: two visits for one patient"):
        st.dataframe(pd.DataFrame([
            {"patient_id": "P001", "image_path": "scans/P001/visit0.nii.gz", "visit_idx": 0, "time_months": 0, "C_age": 45, "C_sex": "Female", "M_medicine_name": "ExampleMedicine"},
            {"patient_id": "P001", "image_path": "scans/P001/visit1.nii.gz", "visit_idx": 1, "time_months": 6, "C_age": 45, "C_sex": "Female", "M_medicine_name": "ExampleMedicine"},
        ]), use_container_width=True)
        st.caption("Illustrative rows only. Replace all example IDs, paths, and values with authorised data. The scan paths must exist inside your ZIP.")

    checkpoint_file = st.file_uploader("Model checkpoint (.pt)", type=["pt"])
    manifest_file = st.file_uploader("Prediction manifest (.csv)", type=["csv"])
    scans_zip = st.file_uploader("Scan files archive (.zip)", type=["zip"])
    st.caption(
        "Manifest requires patient_id, image_path, and visit_idx. Include the same feature columns "
        "used during training where available. Archive paths must match image_path values relative to the manifest."
    )

    if st.button("Run prediction", type="primary", disabled=not (checkpoint_file and manifest_file and scans_zip)):
        try:
            manifest_preview = pd.read_csv(manifest_file)
            required = {"patient_id", "image_path", "visit_idx"}
            missing = required - set(manifest_preview.columns)
            if missing:
                st.error(f"Manifest is missing required columns: {', '.join(sorted(missing))}")
                st.stop()
            if manifest_preview.empty:
                st.error("The manifest contains no rows.")
                st.stop()

            with tempfile.TemporaryDirectory(prefix="multimodal_demo_") as temp_name:
                work = Path(temp_name).resolve()
                checkpoint_path = work / "checkpoint.pt"
                manifest_path = work / "prediction_manifest.csv"
                output_path = work / "predictions.csv"
                checkpoint_path.write_bytes(checkpoint_file.getvalue())
                manifest_path.write_bytes(manifest_file.getvalue())

                # Extract only regular files and reject paths that escape the workspace.
                with zipfile.ZipFile(io.BytesIO(scans_zip.getvalue())) as archive:
                    members = archive.infolist()
                    if len(members) > 5000:
                        st.error("Archive contains too many entries (limit: 5,000).")
                        st.stop()
                    total_uncompressed = sum(m.file_size for m in members)
                    if total_uncompressed > 2 * 1024 * 1024 * 1024:
                        st.error("Archive is larger than the 2 GiB safety limit.")
                        st.stop()
                    for member in members:
                        target = (work / member.filename).resolve()
                        if target != work and work not in target.parents:
                            st.error("Archive contains an unsafe path; extraction was stopped.")
                            st.stop()
                        if member.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with archive.open(member) as source, target.open("wb") as dest:
                                shutil.copyfileobj(source, dest, length=1024 * 1024)

                # The manifest lives at the workspace root, so relative image paths resolve from there.
                missing_paths = []
                for value in manifest_preview["image_path"].dropna().astype(str):
                    candidate = Path(value).expanduser()
                    resolved = candidate.resolve() if candidate.is_absolute() else (work / candidate).resolve()
                    if work not in resolved.parents and resolved != work:
                        missing_paths.append(value)
                    elif not resolved.exists():
                        missing_paths.append(value)
                if missing_paths:
                    st.error(
                        "Some image_path entries do not match files in the ZIP (showing up to 10): "
                        + ", ".join(missing_paths[:10])
                    )
                    st.stop()

                command = [
                    sys.executable, str(Path(__file__).with_name("predict_multimodal_raw_image_model.py")),
                    "--checkpoint", str(checkpoint_path),
                    "--manifest", str(manifest_path),
                    "--output", str(output_path),
                    "--device", "cpu",
                ]
                with st.spinner("Loading scans and running inference on CPU. This may take a while..."):
                    completed = subprocess.run(
                        command, capture_output=True, text=True, timeout=900, check=False,
                        cwd=str(Path(__file__).resolve().parent),
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                    )
                if completed.returncode != 0:
                    st.error("Prediction failed. Review the diagnostic output below.")
                    st.code((completed.stderr or completed.stdout)[-12000:])
                elif not output_path.exists():
                    st.error("The prediction process ended without creating an output CSV.")
                    st.code(completed.stdout[-4000:])
                else:
                    results = pd.read_csv(output_path)
                    st.success(f"Prediction completed for {len(results)} visit(s).")
                    st.dataframe(results, use_container_width=True)
                    prob_cols = [c for c in ["prob_stage_0", "prob_stage_1", "prob_stage_2", "prob_stage_3"] if c in results]
                    if prob_cols:
                        selected_row = st.selectbox(
                            "Choose visit to visualise",
                            results.index.tolist(),
                            format_func=lambda i: f"Patient {results.loc[i, 'patient_id']} — visit {results.loc[i, 'visit_idx']}",
                        )
                        chart = results.loc[[selected_row], prob_cols].T
                        chart.columns = ["Probability"]
                        chart.index = [name.replace("prob_stage_", "Stage ") for name in chart.index]
                        st.bar_chart(chart)
                    st.download_button(
                        "Download predictions CSV",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="multimodal_predictions.csv",
                        mime="text/csv",
                    )
                    st.caption(
                        "These are model outputs, not validated diagnoses. Interpret only within an approved research protocol."
                    )
        except subprocess.TimeoutExpired:
            st.error("Inference exceeded the 15-minute limit. Try fewer scans or run locally from the command line.")
        except zipfile.BadZipFile:
            st.error("The scan archive is not a valid ZIP file.")
        except Exception as exc:
            st.error(f"Could not run inference: {type(exc).__name__}: {exc}")

else:
    st.subheader("Model and scientific scope")
    st.markdown("""
    **Architecture:** 3D CNN scan embeddings fused with tabular features and missingness masks,
    followed by a causal Transformer across a patient's visits.

    **Supported tabular prefixes:** `C_` clinical, `T_` treatment, `M_` medicine, and `W_` wearable.
    Numeric and categorical text features are supported by the training pipeline.

    **Prediction task:** the model predicts the supplied stage label for each visit. It is not a validated
    forecast of future progression, and it does not establish causal treatment effects.

    **Validation status:** repository CI tests use synthetic data. No real-patient or clinical validation
    is claimed. Model quality depends on label quality, data size, preprocessing, site/scanner effects,
    and an appropriate independent evaluation.
    """)
    st.markdown("### Local setup")
    st.code("python -m pip install -r requirements.txt\nstreamlit run app.py", language="powershell")
