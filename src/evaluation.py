
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from .config import FINDINGS, ID_COL


def evaluate_predictions(batch_records, gold_df):
    # Match each prediction to its gold-labelled report
    gold_lookup = gold_df.set_index(ID_COL)
    score_rows = []
    all_correct = 0
    all_labels = 0
    baseline_correct = 0

    for record in batch_records:
        report_id = record[ID_COL]

        if report_id not in gold_lookup.index:
            print("Gold label not found for:", report_id)
            continue

        gold_row = gold_lookup.loc[report_id]

        for finding in FINDINGS:
            actual = int(gold_row[finding])
            predicted = record["predictions"].get(finding)

            # Skip invalid or missing predictions
            if predicted not in (0, 1):
                continue

            predicted = int(predicted)
            score_rows.append({
                "Report_ID": report_id,
                "Finding": finding,
                "Gold": actual,
                "Predicted": predicted,
                "Correct": int(actual == predicted),
            })

            all_labels += 1
            all_correct += int(actual == predicted)
            baseline_correct += int(actual == 0)

    scores_df = pd.DataFrame(score_rows)

    if scores_df.empty:
        print("No valid predictions could be scored.")
        return scores_df, pd.DataFrame(), pd.DataFrame()

    summary = []

    for finding in FINDINGS:
        part = scores_df[scores_df["Finding"] == finding]

        if part.empty:
            continue

        y_true = part["Gold"]
        y_pred = part["Predicted"]

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        summary.append({
            "Finding": finding,
            "N": len(part),
            "Gold_Pos": int(y_true.sum()),
            "Pred_Pos": int(y_pred.sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Accuracy": (tp + tn) / len(part),
            "Precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "F1": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "All_Zero_Baseline": float((y_true == 0).mean()),
        })

    summary_df = pd.DataFrame(summary)

    print("REPORT-LEVEL RESULTS")
    print("Reports scored:", scores_df["Report_ID"].nunique())
    print("Valid finding predictions:", all_labels)
    print("Correct predictions:", all_correct)
    print(
        "Overall accuracy:",
        round(all_correct / all_labels, 4) if all_labels else "N/A"
    )
    print(
        "All-zero baseline accuracy:",
        round(baseline_correct / all_labels, 4)
        if all_labels else "N/A"
    )

    print("\nPER-FINDING RESULTS")
    print(summary_df.round(3).to_string(index=False))

    print("\nINDIVIDUAL REPORT PREDICTIONS")
    print(scores_df.to_string(index=False))

    # Correct vs incorrect predictions per injury
    injury_summary = (
        scores_df
        .groupby("Finding")
        .agg(
            Reports_Scored=("Correct", "count"),
            Correct=("Correct", "sum"),
        )
        .reset_index()
    )

    injury_summary["Wrong"] = (
        injury_summary["Reports_Scored"] - injury_summary["Correct"]
    )

    injury_summary["Accuracy_%"] = (
        100 * injury_summary["Correct"]
        / injury_summary["Reports_Scored"]
    ).round(2)

    injury_summary = injury_summary.rename(
        columns={"Finding": "Injury"}
    )

    injury_summary = injury_summary[
        ["Injury", "Reports_Scored", "Correct", "Wrong", "Accuracy_%"]
    ]

    print("\nCORRECT VS INCORRECT PREDICTIONS PER INJURY")
    print(injury_summary.to_string(index=False))

    print(
        "\nTotal finding-level predictions:",
        injury_summary["Reports_Scored"].sum()
    )
    print("Total correct:", injury_summary["Correct"].sum())
    print("Total wrong:", injury_summary["Wrong"].sum())

    return scores_df, summary_df, injury_summary
