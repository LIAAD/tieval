from typing import Dict, Iterable, Set

from tieval.entities import Entity
from tieval.links import TLink

from typing import Set

from tieval.closure import temporal_closure
from tieval.links import TLink


from tabulate import tabulate


def _print_table(result):
    rows = list(set(result.keys()))
    rows.sort()

    cols = list(set(key for row in rows for key in result[row]))
    cols.sort()

    content = [[row] + [round(result[row][col], 2) for col in cols]
               for row in rows]

    tabel = tabulate(
        tabular_data=content,
        headers=cols,
        tablefmt='orgtbl'
    )

    print(tabel)


def confusion_matrix(annotations, predictions):

    tp = len(annotations & predictions)
    fp = len(predictions - annotations)
    fn = len(annotations - predictions)

    return tp, fp, fn


def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) else 0


def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) else 0


def temporal_recall(
        prediction: Set[TLink],
        annotation: Set[TLink]
):

    prediction_closure = temporal_closure(prediction)

    numerator = len(annotation & prediction_closure)
    denominator = len(annotation)

    return numerator / denominator


def temporal_precision(
        annotation: Set[TLink],
        prediction: Set[TLink]
):

    annotation_closure = temporal_closure(annotation)

    numerator = len(prediction & annotation_closure)
    denominator = len(prediction)

    return numerator / denominator


def temporal_awareness(
        annotation: Set[TLink],
        prediction: Set[TLink],
):
    """Compute the temporal awareness of a system.

    Temporal awareness is a f1 measure that takes into account the temporal
    closure of a system. For more information refer to the original paper
    (UzZaman et al.)[https://aclanthology.org/P11-2061.pdf]"""

    recall = temporal_recall(annotation, prediction)
    precision = temporal_precision(annotation, prediction)

    return 2 * recall * precision / (recall + precision)


def identification(
        annotations: Dict[str, Entity],
        predictions: Dict[str, Entity],
        verbose=False
) -> Dict:

    n_docs = len(annotations)

    M_precision, M_recall = 0, 0
    tps, fps, fns = 0, 0, 0
    for doc in annotations:
        true = set(t.endpoints for t in annotations[doc])
        pred = set(p.endpoints for p in predictions[doc])

        tp = len(true & pred)
        fp = len(pred - true)
        fn = len(true - pred)

        # update macro metrics counts
        M_precision += tp / (tp + fp) if (tp + fp) else 0
        M_recall += tp / (tp + fn) if (tp + fn) else 0

        # update micro metrics counts
        tps += tp
        fps += fp
        fns += fn

    # compute macro metrics
    M_precision /= n_docs
    M_recall /= n_docs

    # compute micro metrics
    m_precision = tps / (tps + fps)
    m_recall = tps / (tps + fns)

    result = {
        "micro": {
            "recall": m_recall,
            "precision": m_precision,
            "f1": 2 * m_recall * m_precision / (m_recall + m_precision)
        },
        "macro": {
            "recall": M_recall,
            "precision": M_precision,
            "f1": 2 * M_recall * M_precision / (M_recall + M_precision)
        }
    }

    if verbose:
        _print_table(result)

    return result


def timex_identification(
        annotations: Dict[str, Entity],
        predictions: Dict[str, Entity],
        verbose=False
) -> Dict:
    return identification(annotations, predictions, verbose)


def event_identification(
        annotations: Dict[str, Entity],
        predictions: Dict[str, Entity],
        verbose=False
) -> Dict:
    return identification(annotations, predictions, verbose)


def tlink_identification(
        annotation: Dict[str, TLink],
        predictions: Dict[str, TLink],
        verbose=False
) -> Dict:
    return None


def tlink_classification(
        annotations: Dict[str, TLink],
        predictions: Dict[str, TLink],
        verbose=False
) -> Dict:

    n_docs = len(annotations)

    M_precision, M_recall = 0, 0
    M_precision_t, M_recall_t = 0, 0
    tps, fps, fns = 0, 0, 0
    for doc in annotations:
        true = set(annotations[doc])
        pred = set(predictions[doc])

        tp, fp, fn = confusion_matrix(true, pred)

        # update macro metrics counts
        M_precision += precision(tp, fp)
        M_recall += recall(tp, fp)

        M_precision_t += temporal_precision(true, pred)
        M_recall_t += temporal_recall(true, pred)

        # update micro metrics counts
        tps += tp
        fps += fp
        fns += fn

    # compute macro metrics
    M_precision /= n_docs
    M_recall /= n_docs

    M_precision_t /= n_docs
    M_recall_t /= n_docs

    # compute micro metrics
    m_precision = tps / (tps + fps)
    m_recall = tps / (tps + fns)

    result = {
        "micro": {
            "recall": m_recall,
            "precision": m_precision,
            "f1": 2 * m_recall * m_precision / (m_recall + m_precision)
        },
        "macro": {
            "recall": M_recall,
            "precision": M_precision,
            "f1": 2 * M_recall * M_precision / (M_recall + M_precision),
            "temporal_recall": M_recall_t,
            "temporal_precision": M_precision_t,
            "temporal_awareness": 2 * M_recall_t * M_precision_t / (M_recall_t + M_precision_t)
        }
    }

    if verbose:
        _print_table(result)

    return result
