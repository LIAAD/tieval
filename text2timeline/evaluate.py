from typing import Dict

from text2timeline.entities import Entity
from text2timeline.links import TLink

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

        tp = len(true and pred)
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
    return None
