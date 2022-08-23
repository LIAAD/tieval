from typing import Set, Tuple

from tieval.closure import temporal_closure
from tieval.links import TLink


def temporal_recall(
        prediction: Set[TLink],
        annotation: Set[TLink]
) -> Tuple[int, int]:
    """Compute recall by taking into account the temporal closure of the predictions.

    :param Set[TLink] prediction: The TLinks predicted by the system.
    :param Set[TLink] annotation: The reference TLinks.
    """

    prediction_closure = temporal_closure(prediction)

    numerator = len(annotation & prediction_closure)
    denominator = len(annotation)

    return numerator, denominator


def temporal_precision(
        prediction: Set[TLink],
        annotation: Set[TLink],
) -> Tuple[int, int]:
    """Compute precision by taking into account the temporal closure of the annotations.

    :param Set[TLink] prediction: The TLinks predicted by the system.
    :param Set[TLink] annotation: The reference TLinks.
    """

    annotation_closure = temporal_closure(annotation)

    numerator = len(prediction & annotation_closure)
    denominator = len(prediction)

    return numerator, denominator


def temporal_awareness(
        prediction: Set[TLink],
        annotation: Set[TLink],
) -> float:
    """Compute the temporal awareness of a system.

    Temporal awareness is a f1 measure that takes into account the temporal
    closure of a system. For more information refer to the original `paper`_.

    :param Set[TLink] prediction: The TLinks predicted by the system.
    :param Set[TLink] annotation: The reference TLinks.

    .. _paper: https://aclanthology.org/P11-2061.pdf
    """

    r_n, r_d = temporal_recall(annotation, prediction)
    recall = r_n / r_d

    p_n, p_d = temporal_precision(annotation, prediction)
    precision = p_n / p_d

    return 2 * recall * precision / (recall + precision)
