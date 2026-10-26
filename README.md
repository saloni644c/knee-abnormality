# Knee MRI Abnormality Detection from Radiology Reports

Weakly supervised, multi-label detection of 12 knee findings on 3D knee MRI. Training labels are extracted from free-text radiology reports with an LLM, then validated against a small set of doctor-verified cases.

> **Status:** work in progress. Label extraction (Phase 1) is under active development. Not for clinical use.

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

This project follows the weak supervision framing of Das et al. (2025), who use an instruction-tuned LLM to label unlabelled reports, then validate those labels against a manually labelled set.

**Label extraction (Phase 1).** An instruction-tuned LLM (currently MedGemma-4B) scores each report for each finding. For every finding it returns probabilities for `present`, `absent`, and `not_mentioned`, plus a verbatim evidence quote in the original language, so every label can be audited. Raw probabilities are saved. Hard 0/1 labels are derived only for evaluation.

**Validation (Phase 2).** Extracted labels are compared with the gold cases per finding and per language, using AUC, precision, recall, F1, and Cohen's kappa, with an error analysis. Gold coverage is uneven across languages, so pooled scores are not reported alone.

**Imaging (Phases 3 to 5).** 3D MRI preprocessing and an MRNet-style multi-label CNN (Bien et al., 2018), evaluated per finding. This is an MRNet-style model, not a replication, so it should not be compared directly with the paper's numbers.

Prompt versions and experiment outcomes are logged in `docs/experiments.md`.

## Results so far

| Finding | Gold cases | AUC (p_present) | Accuracy (p > 0.5) | Note |
|---|---|---|---|---|
| ACL | 58 | 0.944 | 0.862 | Prompt refined on these cases, so optimistic |

Meniscus, OA, and other findings are not yet validated.

## Repository structure

All logic and prompts live in `src/`. Notebooks only call the scripts. Some scripts are still being written, so check the status above before running them.

## Progress

| Phase | Description | Status |
|---|---|---|
| 1 | LLM weak-label extraction | In progress |
| 2 | Validation against gold labels | Planned |
| 3 | 3D MRI preprocessing | Planned |
| 4 | Multi-label classifier | Planned |
| 5 | Per-finding evaluation | Planned |

Results will be added here as each phase is completed.

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

Saloni Chourasiya, MSc Quantum Engineering, University of Innsbruck.
