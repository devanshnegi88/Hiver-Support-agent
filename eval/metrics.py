"""
Automated (non-LLM-judged) metrics: things we can score with plain code
against the golden set's gold labels.
"""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def intent_metrics(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(gold, pred, labels=labels, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(gold, pred, labels=labels).tolist(),
        "labels_order": labels,
    }


def escalation_metrics(gold: list[bool], pred: list[bool]) -> dict:
    """Precision/recall framed around 'escalate' as the positive class.
    Recall matters more here than precision: a missed escalation (false
    negative) means a human never sees a case that needed one — a false
    positive just costs a human a few extra seconds of review.
    """
    return {
        "precision_escalate": precision_score(gold, pred, zero_division=0),
        "recall_escalate": recall_score(gold, pred, zero_division=0),
        "f1_escalate": f1_score(gold, pred, zero_division=0),
        "false_negative_rate": _false_negative_rate(gold, pred),
    }


def _false_negative_rate(gold: list[bool], pred: list[bool]) -> float:
    fn = sum(1 for g, p in zip(gold, pred) if g and not p)
    positives = sum(1 for g in gold if g)
    return fn / positives if positives else 0.0


def summarize_to_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results)
