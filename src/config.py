```python
# Configuration for MedGemma report-label extraction

# Model configuration
MODEL_PATH = "google/medgemma-4b-it"
USE_4BIT = True

# Dataset configuration
REPORTS_CSV = "/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv"
GOLD_CSV = "/kaggle/working/gold.csv"

ID_COL = "StudyInstanceUID"
TEXT_COL = "Report"

# Generation configuration
MAX_NEW_TOKENS = 900
OUT_PATH = "/kaggle/working/extract_v1.jsonl"

# Findings to extract
FINDINGS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# Mapping between finding names and gold-label column names
GOLD_COLS = {finding: finding for finding in FINDINGS}

# Valid extraction states
STATES = [
    "present",
    "absent",
    "uncertain",
    "not_mentioned",
]

# Valid severity levels
SEVERITIES = [
    "none",
    "mild",
    "moderate",
    "severe",
    "unspecified",
]

# Batch inference configuration
BATCH_TEST_N = 30
BATCH_SIZE = 5
BATCH_MAX_NEW_TOKENS = MAX_NEW_TOKENS

# Binary decision configuration
UNCERTAIN_IS_POSITIVE = False
```