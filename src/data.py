from pathlib import Path

import pandas as pd


TARGET_COLUMNS = [
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


def load_training_data(data_root):
    """Load training studies and training MRI series metadata."""
    data_root = Path(data_root)

    train_df = pd.read_csv(data_root / "train.csv")
    train_series_df = pd.read_csv(data_root / "train_series.csv")

    return train_df, train_series_df


def load_test_data(data_root):
    """Load test studies and test MRI series metadata."""
    data_root = Path(data_root)

    test_df = pd.read_csv(data_root / "test.csv")
    test_series_df = pd.read_csv(data_root / "test_series.csv")

    return test_df, test_series_df


def get_gold_cases(train_df):
    """
    Return studies with all 12 target labels available.

    Missing labels are not treated as negative labels.
    """
    missing_columns = [
        col for col in TARGET_COLUMNS
        if col not in train_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing target columns: {missing_columns}"
        )

    gold_mask = train_df[TARGET_COLUMNS].notna().all(axis=1)

    return train_df.loc[gold_mask].copy().reset_index(drop=True)


def get_study_index(train_df):
    """Create a lookup from StudyInstanceUID to the study record."""
    if "StudyInstanceUID" not in train_df.columns:
        raise ValueError("StudyInstanceUID column is missing.")

    if train_df["StudyInstanceUID"].duplicated().any():
        raise ValueError("StudyInstanceUID contains duplicate values.")

    return train_df.set_index("StudyInstanceUID", drop=False)


def get_study_report(train_df, study_id):
    """Retrieve the report associated with a study ID."""
    rows = train_df.loc[
        train_df["StudyInstanceUID"] == study_id
    ]

    if rows.empty:
        raise KeyError(f"Study not found: {study_id}")

    report = rows.iloc[0]["Report"]

    if pd.isna(report):
        return None

    return str(report)


def get_study_labels(train_df, study_id):
    """Retrieve the 12 target labels for a study ID."""
    rows = train_df.loc[
        train_df["StudyInstanceUID"] == study_id
    ]

    if rows.empty:
        raise KeyError(f"Study not found: {study_id}")

    return rows.iloc[0][TARGET_COLUMNS].to_dict()


def get_study_series(series_df, study_id):
    """Return all MRI series metadata for a study."""
    return series_df.loc[
        series_df["StudyInstanceUID"] == study_id
    ].copy()


def get_mri_files(data_root, study_id, series_id=None):
    """
    Find DICOM files for a study, optionally restricted to one series.

    Expected structure:
    data_root/
        train_series/
            StudyInstanceUID/
                SeriesInstanceUID/
                    image.dcm
    """
    data_root = Path(data_root)
    study_dir = data_root / "train_series" / str(study_id)

    if not study_dir.is_dir():
        study_dir = data_root / "test_series" / str(study_id)

    if not study_dir.is_dir():
        raise FileNotFoundError(
            f"No MRI directory found for study {study_id}"
        )

    search_dir = (
        study_dir / str(series_id)
        if series_id is not None
        else study_dir
    )

    if not search_dir.is_dir():
        raise FileNotFoundError(
            f"No series directory found: {search_dir}"
        )

    return sorted(search_dir.rglob("*.dcm"))