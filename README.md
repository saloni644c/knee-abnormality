# Knee MRI Abnormality Detection

A machine learning research project focused on detecting 12 abnormalities in knee MRI examinations. The project investigates automated analysis of 3D knee MRI volumes, using radiology reports and doctor-verified labels to develop and evaluate abnormality detection methods.

> **Status:** Work in progress. Initial experiments on radiology report-based label extraction have been completed. MRI preprocessing and image-based model development are planned. Not for clinical use.

## Problem

| | |
|---|---|
| Task | Multi-label classification of 12 findings per knee MRI exam |
| Findings | ACL, MCL, medial meniscus, lateral meniscus, medial OA, lateral OA, patellofemoral OA, effusion, synovitis, Baker's cyst, bone contusion, fracture |
| Exams | about 4,400 |
| Doctor-verified labels | 58 |
| Report languages | 9 |

Manual labels exist for only a small fraction of exams, but every exam has a report. The approach is to turn reports into weak labels, measure how trustworthy they are, and train an image-only model on them. Reports are used only to create labels. Inference uses the MRI alone.

## Method

The project is organized into two main stages, progressing from radiology report-based label extraction to image-based abnormality detection.

**Stage 1 — Radiology Report-Based Label Extraction**

Use large language models (currently MedGemma-4B) to extract the 12 target abnormalities from free-text radiology reports. The extracted labels are evaluated against the available doctor-verified cases using per-finding metrics, including AUC, precision, recall, F1-score, and Cohen's kappa. Error analysis is used to assess label quality and identify limitations.

**Stage 2 — MRI-Based Abnormality Detection**

Use the extracted labels to train a multi-label convolutional neural network on 3D knee MRI volumes. This stage includes MRI preprocessing, dataset preparation, model development, and evaluation. An MRNet-style architecture serves as a reference for the image-based model.

The objective is to develop a model that predicts the 12 target abnormalities directly from MRI volumes, without requiring radiology reports at inference time.

Experiment versions, prompts, evaluation results, and methodological decisions are documented in `docs/experiments.md` and the `experiments/` directory.

## Results so far

### Stage 1: Radiology Report-Based Label Extraction

An initial MedGemma-4B experiment evaluated the extraction of 12 knee abnormalities against 58 doctor-verified MRI studies.

The experiment achieved a macro AUC of **0.784 across measurable findings**. Performance varied by finding, with stronger results for ACL extraction and Baker's cyst detection, while osteoarthritis findings showed limitations in threshold-based classification.

Detailed per-finding metrics are available in [`experiments/medgemma_v1/metrics_per_finding.csv`](experiments/medgemma_v1/metrics_per_finding.csv).

These results provide an initial benchmark for improving report-based label extraction. The evaluation set is small, and the results should be interpreted as preliminary.

### Stage 2: MRI-Based Abnormality Detection

CNN training and image-based evaluation results will be added as this stage progresses.

## Repository Structure

The repository separates reusable implementation code, exploratory notebooks, experiment-specific results, and project documentation.

- `src/` — Project scripts, reusable functions, and implementation code.
- `notebooks/` — Notebooks for experimentation, analysis, and evaluation.
- `experiments/` — Experiment-specific results and documentation, organized by approach and version.
- `docs/` — Project methodology, experiment history, and development notes.
- `tests/` — Tests for implemented functionality.

## Progress

The project is organized into two main stages:

| Stage | Description | Status |
|---|---|---|
| 1 | Radiology report-based label extraction and validation | Initial MedGemma baseline evaluated; further improvements ongoing |
| 2 | MRI preprocessing, CNN training, and image-based evaluation | Planned / in development |

Stage 1 establishes a baseline for extracting abnormalities from radiology reports. Stage 2 will use the resulting labels to train and evaluate an image-based model.

Progress and results will be documented as the project develops.

## Reproduce

Running the pipeline requires:
- A Kaggle notebook with a GPU, with the RSNA competition data attached.
- A Hugging Face account that has accepted the MedGemma terms, since the model is gated.

The scripts are not yet in a state where a single command reproduces the results. Setup and run instructions will be added when `src/` is complete.

## Limitations

- Weak labels are noisy and are not ground truth.
- Only 58 gold cases are available for validation, so per-language estimates have wide uncertainty.
- The prompt for ACL was refined on the same 58 cases used to report its score, so that score is optimistic.
- This is a research prototype and has not been validated for clinical use.

## Data

The data belongs to the RSNA Knee Abnormality Detection competition and is not included here. Use must follow the competition rules.

## References

- Das et al., *Weakly supervised language models for automated extraction of critical findings from radiology reports*, npj Digital Medicine, 2025.
- Felfeliyan et al., 2022. Work on soft labels and noise-robust loss for knee MRI with partial and noisy annotations.
- Bien et al., *Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet*, PLOS Medicine, 2018.
- Irvin et al., *CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison*, AAAI, 2019.
- Smit et al., *CheXbert: Combining automatic labelers and expert annotations for accurate radiology report labeling using BERT*, EMNLP, 2020.

## Author

Saloni Chourasiya 
