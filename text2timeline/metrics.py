
from typing import Set

from text2timeline.closure import temporal_closure
from text2timeline.links import TLink


def temporal_recall(system: Set[TLink], annotation: Set[TLink]):

    system_closure = temporal_closure(system)

    numerator = len(annotation.intersection(system_closure))
    denominator = len(annotation)

    return numerator / denominator


def temporal_precision(system: Set[TLink], annotation: Set[TLink]):

    annotation_closure = temporal_closure(annotation)

    numerator = len(system.intersection(annotation_closure))
    denominator = len(system)

    return numerator / denominator


def temporal_awareness(system: Set[TLink], annotation: Set[TLink]):
    """Compute the temporal awareness of a system.

    Temporal awareness is a f1 measure that takes into account the temporal closure of a system. For more information
    refer to the original paper (UzZaman et al.)[https://aclanthology.org/P11-2061.pdf]"""

    recall = temporal_recall(system, annotation)
    precision = temporal_precision(system, annotation)

    return 2 * recall * precision / (recall + precision)
