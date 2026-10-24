# Whole-feature-group missingness experiment

This experiment extends the repository's synthetic multimodal longitudinal demo with a focused, reproducible comparison relevant to trustworthy machine learning for medicine.

## Why this experiment

The Machine Learning in Medicine Lab lists disease-progression modelling, performance generalisation, subgroup discovery, and bias detection among its interests:
https://www.mlm-lab.ch/research/

The first methodological reference below studies longitudinal multimodal prediction when data modalities/views are missing. The second is Professor Koch's research on distribution-shift detection for medical-AI monitoring. This experiment borrows the *evaluation question*—what happens when input information is unavailable?—but it does **not** reproduce either paper's method, dataset, or findings.

## What is compared

Two versions of the same temporal Transformer are trained on the same synthetic-data split for each run seed:

1. **Standard training**: no additional whole-group dropout.
2. **Feature-group dropout training**: independently makes each feature group unavailable with probability 0.20 when a training patient trajectory is loaded. At least one group is retained.

Evaluation uses a patient-disjoint test split and six conditions: complete inputs, or one of the five input groups fully unavailable. A group is hidden by zeroing both its scaled values and its observation indicators. Model checkpoints are selected using validation loss averaged over the same six conditions; the held-out test set is not used to choose checkpoints.

The five groups in this prototype are clinical, treatment, medicine, wearable, and imaging-derived features. They are code-level feature groups, not five independent sensing modalities: treatment and medicine are separate groups of clinical context, and the image data in the main experiment are synthetic image-derived feature columns, not raw scans.

## Run it

Install the project requirements first, then run from the repository root:

```bash
python experiments/missing_modality_robustness.py
```

The default run uses 500 synthetic patients per run, ten epochs, and seeds 42, 43, and 44. A quicker smoke run is:

```bash
python experiments/missing_modality_robustness.py \
  --n-patients 40 --epochs 1 --batch-size 16 --seeds 42 --cpu \
  --output-dir results/missing_modality_robustness_smoke
```

Options are available for patient count, epoch count, batch size, learning rate, dropout probability, seeds, output directory, and CPU execution. For less noisy estimates, increase the number of seeds and report their variation; do not rely on a single run.

## Outputs and metrics

- `run_level_metrics.csv`: every run, training strategy, and test condition.
- `summary_by_condition.csv`: mean and standard deviation across seeds for accuracy, balanced accuracy, macro-F1, expected calibration error, and multiclass Brier score.
- `macro_f1_by_missing_modality.png`: plotted macro-F1 by condition and training strategy, with across-run standard-deviation error bars.
- `run_summary.json`: configuration, split sizes, conditions, and interpretation limits.
- `history_seed_*_*.csv`: training and validation-loss history for each strategy/run.

The experiment also records **expected calibration error (ECE)** and the **multiclass Brier score** to give a first look at confidence quality, not just classification performance. Lower ECE/Brier is generally preferable under their definitions, but neither metric alone establishes that probabilities are clinically reliable. ECE depends on binning; report it as a descriptive estimate, especially for small test sets.

## Interpretation limits

- Every sample and label is synthetic. Results describe only this toy data-generating process.
- The prediction target remains the **stage at an observed visit**. It does not predict a future stage, time-to-progression, or a treatment effect.
- Randomly masking a feature group is a controlled stress test; it does not model every real clinical missingness mechanism.
- The experiment does not implement hidden-subgroup discovery, fairness mitigation, distribution-shift detection, or label-free performance estimation.
- It provides no evidence of real-patient performance or clinical utility.

## Research references

1. Boqi Chen, Junier Oliva, and Marc Niethammer. “A Unified Model for Longitudinal Multi-Modal Multi-View Prediction with Missingness.” MICCAI 2024, pp. 410–420. This paper uses the Osteoarthritis Initiative for future-timepoint prediction; its methods/results are not reproduced here.  
   Paper: https://papers.miccai.org/miccai-2024/034-Paper2390.html  
   Code: https://github.com/uncbiag/UniLMMV

2. Lisa M. Koch, Christian F. Baumgartner, and Philipp Berens. “Distribution Shift Detection for the Postmarket Surveillance of Medical AI Algorithms: A Retrospective Simulation Study.” *npj Digital Medicine* 7, 120 (2024). This work studies shift detection on retinal-image data; shift detectors are not implemented in this experiment.  
   Paper: https://www.nature.com/articles/s41746-024-01085-w  
   Code: https://github.com/lmkoch/postmarket-shift-detection

3. Alceu Bissoto, Thao-Dan Hoang, Tim Flühmann, S. Sun, Christian F. Baumgartner, and Lisa M. Koch. “Subgroup Performance Analysis in Hidden Stratifications.” MICCAI 2025. This is a useful future reading direction for evaluating subgroup performance; subgroup discovery is not implemented here.  
   Lab reference: https://www.mlm-lab.ch/research/  
   Paper: https://papers.miccai.org/miccai-2025/paper/0459_paper.pdf
