
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
