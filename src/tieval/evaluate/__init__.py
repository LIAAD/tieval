from typing import Dict, List, Tuple

from tieval.entities import Entity, Timex
from tieval.evaluate.metrics import temporal_awareness
from tieval.evaluate.metrics import temporal_precision
from tieval.evaluate.metrics import temporal_recall
from tieval.links import TLink


def confusion_matrix(annotations, predictions):
    tp = sum(1 for pred in predictions if pred in annotations)
    fp = sum(1 for pred in predictions if pred not in annotations)
    fn = sum(1 for annot in annotations if annot not in predictions)
    return tp, fp, fn


def f_score(recall, precision):
    return 2 * recall * precision / (recall + precision) if recall + precision else 0.0


def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) else 1.0


def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) else 1.0


def timex_identification(
        annotations: Dict[str, List[Timex]],
        predictions: Dict[str, List[Timex]],
        verbose: bool = False
) -> Dict:
    n_docs = len(annotations)

    M_precision, M_recall = 0, 0
    tps, fps, fns = 0, 0, 0
    for doc in annotations:
        true = set(t.offsets for t in annotations[doc] if not t.is_dct)
        pred = set(p.offsets for p in predictions[doc])

        tp, fp, fn = confusion_matrix(true, pred)

        # update macro metrics counts
        M_precision += precision(tp, fp)
        M_recall += recall(tp, fn)

        # update micro metrics counts
        tps += tp
        fps += fp
        fns += fn

    # compute macro metrics
    M_precision /= n_docs
    M_recall /= n_docs
    M_f1 = f_score(M_recall, M_precision)

    # compute micro metrics
    m_precision = precision(tps, fps)
    m_recall = recall(tps, fns)
    m_f1 = f_score(m_recall, m_precision)

    result = {
        "micro": {
            "recall": m_recall,
            "precision": m_precision,
            "f1": m_f1
        },
        "macro": {
            "recall": M_recall,
            "precision": M_precision,
            "f1": M_f1
        }
    }

    if verbose:
        print(result)

    return result


def event_identification(
        annotations: Dict[str, List[Entity]],
        predictions: Dict[str, List[Entity]],
        verbose=False
) -> Dict:
    n_docs = len(annotations)

    M_precision, M_recall = 0, 0
    tps, fps, fns = 0, 0, 0
    for doc in annotations:
        true = set(t.offsets for t in annotations[doc])
        pred = set(p.offsets for p in predictions[doc])

        tp, fp, fn = confusion_matrix(true, pred)

        # update macro metrics counts
        M_precision += precision(tp, fp)
        M_recall += recall(tp, fn)

        # update micro metrics counts
        tps += tp
        fps += fp
        fns += fn

    # compute macro metrics
    M_precision /= n_docs
    M_recall /= n_docs
    M_f1 = f_score(M_recall, M_precision)

    # compute micro metrics
    m_precision = precision(tps, fps)
    m_recall = recall(tps, fns)
    m_f1 = f_score(m_recall, m_precision)

    result = {
        "micro": {
            "recall": m_recall,
            "precision": m_precision,
            "f1": m_f1
        },
        "macro": {
            "recall": M_recall,
            "precision": M_precision,
            "f1": M_f1
        }
    }

    if verbose:
        print(result)

    return result


def tlink_identification(
        annotation: Dict[str, TLink],
        predictions: Dict[str, TLink],
        verbose=False
) -> Dict:
    return {}


def tlink_classification(
        annotations: Dict[str, List[TLink]],
        predictions: Dict[str, List[TLink]],
        verbose=False
) -> Dict:
    n_docs = len(annotations)

    M_accuracy, M_precision, M_recall = 0, 0, 0
    M_precision_t, M_recall_t = 0, 0
    tps, fps, fns = 0, 0, 0
    r_nums, r_dens = 0, 0  # temporal recall numerator and denominator
    p_nums, p_dens = 0, 0  # temporal recall numerator and denominator
    n_relations = 0
    for doc in annotations:
        true = set(annotations[doc])
        pred = set(predictions[doc])

        # standard metrics
        tp, fp, fn = confusion_matrix(true, pred)
        # update micro metrics counts
        tps += tp
        fps += fp
        fns += fn
        n_relations += len(true)

        # update macro metrics counts
        M_accuracy += tp / len(true)
        M_precision += precision(tp, fp)
        M_recall += recall(tp, fp)

        # temporal metrics
        p_numerator, p_denominator = temporal_precision(true, pred)
        r_numerator, r_denominator = temporal_recall(true, pred)

        p_nums += p_numerator
        p_dens += p_denominator
        r_nums += r_numerator
        r_dens += r_denominator

        M_precision_t += p_numerator / p_denominator
        M_recall_t += r_numerator / r_denominator

    # compute macro metrics
    M_accuracy /= n_docs
    M_precision /= n_docs
    M_recall /= n_docs

    M_precision_t /= n_docs
    M_recall_t /= n_docs

    # compute micro metrics
    m_accuracy = tps / n_relations
    m_precision = precision(tps, fps)
    m_recall = recall(tps, fns)

    m_precision_t = p_nums / p_dens
    m_recall_t = r_nums / r_dens

    result = {
        "micro": {
            "accuracy": m_accuracy,
            "recall": m_recall,
            "precision": m_precision,
            "f1": f_score(m_recall, m_precision),
            "temporal_recall": m_recall_t,
            "temporal_precision": m_precision_t,
            "temporal_awareness": f_score(m_recall_t, m_precision_t)
        },

        "macro": {
            "accuracy": M_accuracy,
            "recall": M_recall,
            "precision": M_precision,
            "f1": f_score(M_recall, M_precision),
            "temporal_recall": M_recall_t,
            "temporal_precision": M_precision_t,
            "temporal_awareness": f_score(M_recall_t, M_precision_t)
        }
    }

    if verbose:
        print(result)

    return result


def span_identification(
        annotations: List[List[Tuple[int, int]]],
        predictions: List[List[Tuple[int, int]]],
        strict: bool = True
) -> Dict:
    n_docs = len(annotations)
    M_precision, M_recall = 0, 0
    tps, fps, fns = 0, 0, 0
    for annot, pred in zip(annotations, predictions):
        annot = set(annot)
        pred = set(pred)
        if strict:
            tp, fp, fn = confusion_matrix(annot, pred)
        else:  # relaxed
            span_annot = [Offsets(s_annot, e_annot) for s_annot, e_annot in annot]
            span_pred = [Offsets(s_pred, e_pred) for s_pred, e_pred in pred]
            tp, fp, fn = confusion_matrix(span_annot, span_pred)

        # update macro metrics counts
        M_precision += precision(tp, fp)
        M_recall += recall(tp, fn)

        # update micro metrics counts
        tps += tp
        fps += fp
        fns += fn

    # compute macro metrics
    M_precision /= n_docs
    M_recall /= n_docs
    M_f1 = f_score(M_recall, M_precision)

    # compute micro metrics
    m_precision = precision(tps, fps)
    m_recall = recall(tps, fns)
    m_f1 = f_score(m_recall, m_precision)

    result = {
        "micro": {
            "recall": m_recall,
            "precision": m_precision,
            "f1": m_f1
        },
        "macro": {
            "recall": M_recall,
            "precision": M_precision,
            "f1": M_f1
        }
    }

    return result


class Offsets:
    """An annotation offsets build to check if two entities overlap."""

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Offsets({self.start}, {self.end})"

    def __eq__(self, other) -> bool:
        """If the offsets overlap, the offsets are considered equal."""
        if (self.end < other.start) or (other.end < self.start):
            return False
        return True
