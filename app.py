"""Interactive Streamlit front end for the multimodal research prototype.

Demo mode uses illustrative values only. Real inference requires a trained checkpoint,
a prediction manifest, and an archive of the corresponding authorised scan files.
"""
from __future__ import annotations

import io
import os
import subprocess
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
                                dest.write(source.read())

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
