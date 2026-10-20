# RSNA Knee MRI Abnormality Detection

## Overview

This research project investigates automated detection of 12 knee abnormalities using radiology reports and three-dimensional (3D) knee MRI scans.

The project explores automated label extraction from radiology reports, weakly supervised learning, and multi-label deep learning to identify abnormalities from MRI data.

## Research Objectives

- Extract structured abnormality labels from radiology reports.
- Assess the quality of automatically generated labels against available manually labeled cases.
- Develop a pipeline for processing 3D knee MRI studies.
- Investigate multi-label deep learning methods for MRI-based abnormality detection.
- Evaluate model performance separately for each abnormality.

## Target Abnormalities

The project considers 12 findings:

1. Anterior cruciate ligament (ACL)
2. Medial collateral ligament (MCL)
3. Medial meniscus abnormality
4. Lateral meniscus abnormality
5. Medial compartment osteoarthritis
6. Lateral compartment osteoarthritis
7. Patellofemoral osteoarthritis
8. Joint effusion
9. Synovitis
10. Baker's cyst
11. Bone contusion
12. Fracture

## Methodology

The planned research workflow consists of:

1. Dataset exploration and quality assessment.
2. Automated extraction of labels from radiology reports.
3. Evaluation of extracted labels against available manually labeled cases.
4. MRI preprocessing and preparation of 3D image volumes.
5. Development of multi-label image classification models.
6. Injury-specific evaluation and error analysis.

Automatically generated labels will be treated as pseudo-labels rather than verified ground truth.

## Dataset

The project uses data associated with the RSNA Knee Abnormality Detection competition.

The dataset will remain in its authorized storage location. Large MRI files and restricted competition data will not be included in this repository.

## Evaluation

Model performance will be assessed separately for each target abnormality using appropriate metrics, including:

- Area under the receiver operating characteristic curve (ROC-AUC)
- Precision
- Recall
- F1-score
- Confusion matrix

Metrics will be reported according to the availability and suitability of reference labels.

## Repository Structure

```text
rsna-knee-mri-abnormality-detection/
├── notebooks/
├── src/
├── docs/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Project Status

Research and development in progress.

## Reproducibility and Data Use

Code, methodology, and evaluation procedures will be documented as the project develops. Dataset access and use must comply with the applicable competition rules and data-use requirements.