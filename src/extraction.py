```python
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import FINDINGS, ID_COL, TEXT_COL, REPORTS_CSV
from .generate import generate_batch
from .parse import parse_output
from .decision import to_binary


def extract_gold_labels(
    gold_df,
    model,
    processor,
    batch_size=5,
    max_new_tokens=None,
    output_path=None,
):
    """
    Extract labels for gold-labeled studies.

    Returns a list of records containing raw output, parsed findings,
    binary predictions, and parsing diagnostics.
    """
    from .config import MAX_NEW_TOKENS

    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS

    pool = (
        gold_df
        .dropna(subset=[TEXT_COL])
        .reset_index(drop=True)
    )

    records = []
    run_start = time.time()
    total_reports = len(pool)

    for start in range(0, total_reports, batch_size):
        batch_df = pool.iloc[start:start + batch_size]
        texts = batch_df[TEXT_COL].fillna("").astype(str).tolist()

        raw_outputs = generate_batch(
            texts,
            model,
            processor,
            max_new_tokens=max_new_tokens,
        )

        if len(raw_outputs) != len(batch_df):
            raise RuntimeError(
                f"Expected {len(batch_df)} outputs, "
                f"received {len(raw_outputs)}."
            )

        for (_, row), raw in zip(batch_df.iterrows(), raw_outputs):
            parsed, n_bad, parse_ok = parse_output(raw)

            predictions = {
                finding: to_binary(finding, parsed[finding])
                for finding in FINDINGS
            }

            records.append({
                ID_COL: row[ID_COL],
                "raw_output": raw,
                "parsed": parsed,
                "predictions": predictions,
                "n_bad": n_bad,
                "parse_ok": parse_ok,
            })

        print(
            f"Completed {len(records)}/{total_reports} gold reports | "
            f"Elapsed: {(time.time() - run_start) / 60:.1f} min",
            flush=True,
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records


def extract_non_gold_labels(
    reports_df,
    model,
    processor,
    batch_size=5,
    start_index=0,
    end_index=None,
    pseudo_labels_csv="/kaggle/working/pseudo_labels.csv",
    error_log_jsonl="/kaggle/working/pseudo_label_errors.jsonl",
    max_new_tokens=None,
):
    """
    Extract binary pseudo-labels from studies without all 12 gold labels.

    Saves valid labels after every batch. Existing study IDs are skipped
    so a run can resume from a previously saved CSV.
    """
    from .config import MAX_NEW_TOKENS

    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    required_cols = [ID_COL, TEXT_COL, *FINDINGS]
    missing_cols = [
        col for col in required_cols if col not in reports_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in reports data: {missing_cols}"
        )

    # Build the non-gold pool.
    gold_mask = reports_df[FINDINGS].notna().all(axis=1)

    non_gold_pool = (
        reports_df.loc[~gold_mask]
        .dropna(subset=[ID_COL, TEXT_COL])
        .copy()
        .drop_duplicates(subset=[ID_COL])
        .reset_index(drop=True)
    )

    end = (
        len(non_gold_pool)
        if end_index is None
        else min(end_index, len(non_gold_pool))
    )

    if start_index < 0 or start_index >= len(non_gold_pool) or end <= start_index:
        raise ValueError(
            f"Invalid range. Non-gold pool has "
            f"{len(non_gold_pool)} eligible studies."
        )

    run_pool = (
        non_gold_pool.iloc[start_index:end]
        .copy()
        .reset_index(drop=True)
    )

    labels_path = Path(pseudo_labels_csv)
    errors_path = Path(error_log_jsonl)

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    label_columns = [ID_COL, *FINDINGS]

    # Load existing labels to support resuming.
    if labels_path.exists():
        saved_df = pd.read_csv(labels_path)

        if ID_COL not in saved_df.columns:
            raise ValueError(
                f"Existing output is missing {ID_COL}: {labels_path}"
            )

        if not all(col in saved_df.columns for col in label_columns):
            raise ValueError(
                "Existing pseudo-label CSV does not have all expected "
                "label columns. Rename or move it before starting."
            )

        saved_df = (
            saved_df[label_columns]
            .drop_duplicates(subset=[ID_COL], keep="last")
        )

        already_done = set(saved_df[ID_COL].astype(str))
    else:
        saved_df = pd.DataFrame(columns=label_columns)
        already_done = set()

    pending_df = (
        run_pool[
            ~run_pool[ID_COL].astype(str).isin(already_done)
        ]
        .copy()
        .reset_index(drop=True)
    )

    print("Total studies in input:", len(reports_df))
    print("Gold studies excluded:", int(gold_mask.sum()))
    print("Eligible non-gold studies:", len(non_gold_pool))
    print(f"Selected range [{start_index}, {end}): {len(run_pool)} studies")
    print("Already saved IDs:", len(already_done))
    print("Studies remaining in selected range:", len(pending_df))

    started_at = datetime.now()
    run_t0 = time.time()
    saved_this_run = 0
    failed_this_run = 0

    for batch_start in range(0, len(pending_df), batch_size):
        batch_df = pending_df.iloc[
            batch_start:batch_start + batch_size
        ]
        texts = batch_df[TEXT_COL].fillna("").astype(str).tolist()

        batch_no = batch_start // batch_size + 1
        batch_t0 = time.time()

        print(
            f"\n[{datetime.now():%H:%M:%S}] "
            f"Batch {batch_no}: {len(texts)} studies",
            flush=True,
        )

        try:
            raw_outputs = generate_batch(
                texts,
                model,
                processor,
                max_new_tokens=max_new_tokens,
            )

            if len(raw_outputs) != len(batch_df):
                raise RuntimeError(
                    f"Expected {len(batch_df)} outputs but received "
                    f"{len(raw_outputs)}"
                )

        except Exception as exc:
            failed_this_run += len(batch_df)

            with errors_path.open("a", encoding="utf-8") as ef:
                for _, row in batch_df.iterrows():
                    ef.write(
                        json.dumps(
                            {
                                ID_COL: row[ID_COL],
                                "error": repr(exc),
                                "stage": "generation",
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    )

            print("Batch generation failed:", repr(exc), flush=True)
            continue

        new_rows = []

        with errors_path.open("a", encoding="utf-8") as ef:
            for (_, row), raw in zip(batch_df.iterrows(), raw_outputs):
                parsed, n_bad, parse_ok = parse_output(raw)

                if not parse_ok or n_bad > 0:
                    failed_this_run += 1

                    ef.write(
                        json.dumps(
                            {
                                ID_COL: row[ID_COL],
                                "parse_ok": parse_ok,
                                "n_bad": n_bad,
                                "raw_output": raw,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    )
                    continue

                binary = {
                    finding: int(to_binary(finding, parsed[finding]))
                    for finding in FINDINGS
                }

                new_rows.append({
                    ID_COL: row[ID_COL],
                    **binary,
                })

        if new_rows:
            saved_df = pd.concat(
                [
                    saved_df,
                    pd.DataFrame(new_rows, columns=label_columns),
                ],
                ignore_index=True,
            )

            saved_df = saved_df.drop_duplicates(
                subset=[ID_COL],
                keep="last",
            )

            # Checkpoint after every batch.
            saved_df.to_csv(labels_path, index=False)

            already_done.update(
                str(row[ID_COL]) for row in new_rows
            )
            saved_this_run += len(new_rows)

        elapsed = time.time() - run_t0
        completed = min(
            batch_start + len(batch_df),
            len(pending_df),
        )
        avg_per_study = elapsed / max(completed, 1)
        remaining_seconds = avg_per_study * max(
            len(pending_df) - completed,
            0,
        )

        print(
            f"Batch time: {time.time() - batch_t0:.1f}s | "
            f"Valid labels saved this run: {saved_this_run} | "
            f"Failed/skipped: {failed_this_run} | "
            f"ETA: {remaining_seconds / 60:.1f} min",
            flush=True,
        )

    print("\nRUN FINISHED")
    print("Started:", started_at.strftime("%Y-%m-%d %H:%M:%S"))
    print("Finished:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("New studies saved:", saved_this_run)
    print("Failed/skipped studies:", failed_this_run)
    print("Total unique studies in CSV:", len(saved_df))
    print("Labels CSV:", labels_path)
    print("Error log:", errors_path)

    return saved_df
```