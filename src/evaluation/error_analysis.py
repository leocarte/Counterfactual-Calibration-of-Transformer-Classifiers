"""Error analysis tools for qualitative review."""
import pandas as pd
import numpy as np
from typing import Optional


def extract_error_cases(
    texts: list[str],
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float],
    keywords: list[str],
    top_n: int = 50,
) -> dict[str, pd.DataFrame]:
    """
    Extract the most interesting error cases for expert review.

    Returns DataFrames of:
    - false_positives: Sorted by confidence (most confident errors first)
    - false_negatives: Sorted by confidence
    - boundary_cases: True positives and negatives with probability near 0.5
    """
    df = pd.DataFrame({
        "text": texts,
        "true_label": y_true,
        "predicted_label": y_pred,
        "confidence": y_prob,
        "keyword": keywords,
    })

    # False positives: predicted antisemitic but actually not
    fp = df[(df["predicted_label"] == 1) & (df["true_label"] == 0)]
    fp = fp.sort_values("confidence", ascending=False).head(top_n)

    # False negatives: predicted not antisemitic but actually is
    fn = df[(df["predicted_label"] == 0) & (df["true_label"] == 1)]
    fn = fn.sort_values("confidence", ascending=True).head(top_n)  # lowest confidence = most confident miss

    # Boundary cases: predictions near decision threshold
    df["boundary_score"] = abs(df["confidence"] - 0.5)
    boundary = df.sort_values("boundary_score").head(top_n)

    return {
        "false_positives": fp,
        "false_negatives": fn,
        "boundary_cases": boundary,
    }


def categorize_errors(errors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add placeholder columns for manual error categorization.

    Categories (to be filled by human reviewer):
    - political_criticism: Legitimate Israel criticism flagged as antisemitic
    - coded_antisemitism: Implicit/coded hate missed by model
    - counterspeech: Denouncing hate confused with propagating it
    - sarcasm_irony: Ironic/sarcastic intent misread
    - annotator_disagreement: Text is genuinely ambiguous
    - other: Doesn't fit above categories
    """
    errors_df = errors_df.copy()
    errors_df["error_category"] = ""
    errors_df["reviewer_notes"] = ""
    return errors_df


def export_for_expert_review(
    error_cases: dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    """Export error cases as Excel workbook with separate sheets."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in error_cases.items():
            categorized = categorize_errors(df)
            categorized.to_excel(writer, sheet_name=name[:31], index=False)
