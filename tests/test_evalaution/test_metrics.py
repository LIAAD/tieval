
import pytest

from tieval.links import TLink
from tieval.evaluate.metrics import temporal_precision
from tieval.evaluate.metrics import temporal_recall
from tieval.evaluate.metrics import temporal_awareness


# examples extracted from https://github.com/bethard/anaforatools/blob/master/anafora/test/test_anafora_evaluate.py


@pytest.fixture
def annotation():

    return {
        TLink("A", "B", "BEFORE"),
        TLink("B", "C", "IS_INCLUDED"),
        TLink("D", "C", "INCLUDES"),
        TLink("E", "D", "CONTAINS"),
        TLink("F", "E", "AFTER"),
        TLink("G", "H", "BEGINS-ON"),
        TLink("I", "G", "BEFORE"),
        TLink("J", "K", "IBEFORE"),
        TLink("K", "L", "BEGUN_BY"),
        TLink("L", "K", "BEGINS")

        # inferred:
        # A before B
        # A before F
        # B before F
        # C includes B
        # C before F
        # D includes B
        # D includes C
        # D before F
        # E includes B
        # E includes C
        # E includes D
        # E before F
        # G simultaneous-start H
        # I before G
        # I before H
    }


@pytest.fixture
def system():
    return {
        TLink("A", "B", "BEFORE"),    # (+)
        TLink("A", "B", "BEFORE"),    # annotation, system duplicate
        TLink("B", "A", "AFTER"),     # duplicate
        TLink("B", "E", "CONTAINS"),  # (-)
        TLink("B", "E", "INCLUDES"),  # duplicate
        TLink("B", "F", "BEFORE"),    # (+)
        TLink("F", "D", "AFTER"),     # (+)
        TLink("H", "I", "AFTER"),     # (+)
        TLink("J", "L", "IBEFORE"),   # (+)
        TLink("K", "L", "BEGUN_BY"),  # (+)

        # inferred:
        # (+) A before B
        # ( ) A before E
        # ( ) A before F
        # ( ) B includes E
        # ( ) B before F
        # ( ) D before F
        # (+) E before F
        # ( ) G after I
        # ( ) J i-before L
        # (+) K begun-by L
        # (+) J i-before K
    }


def test_temporal_recall(system, annotation):

    numerator, denominator = temporal_recall(prediction=system, annotation=annotation)
    assert (numerator / denominator) == (4 / 9)


def test_temporal_precision(system, annotation):

    numerator, denominator = temporal_precision(prediction=system, annotation=annotation)
    assert (numerator / denominator) == (6 / 7)


def test_temporal_awareness(system, annotation):

    ta = temporal_awareness(prediction=system, annotation=annotation)
    assert round(ta, 3) == round(24 / 41, 3)
