""" Evaluation metrics.

Functions:
----------
    - temporal_recall
    - temporal_precision
    - temporal_awareness

"""

from typing import Set

from tieval.closure import temporal_closure
from tieval.links import TLink


def temporal_recall(
        prediction: Set[TLink],
        annotation: Set[TLink]
):

    prediction_closure = temporal_closure(prediction)

    numerator = len(annotation & prediction_closure)
    denominator = len(annotation)

    return numerator / denominator


def temporal_precision(
        prediction: Set[TLink],
        annotation: Set[TLink],
):

    annotation_closure = temporal_closure(annotation)

    numerator = len(prediction & annotation_closure)
    denominator = len(prediction)

    return numerator / denominator


def temporal_awareness(
        prediction: Set[TLink],
        annotation: Set[TLink],
):
    """Compute the temporal awareness of a system.

    Temporal awareness is a f1 measure that takes into account the temporal
    closure of a system. For more information refer to the original paper
    (UzZaman et al.)[https://aclanthology.org/P11-2061.pdf]
    """

    recall = temporal_recall(annotation, prediction)
    precision = temporal_precision(annotation, prediction)

    return 2 * recall * precision / (recall + precision)




