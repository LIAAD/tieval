
import pytest

from text2timeline.links import TLink

from text2timeline.evaluate.metrics import temporal_precision
from text2timeline.evaluate.metrics import temporal_recall
from text2timeline.evaluate.metrics import temporal_awareness


# examples extracted from https://github.com/bethard/anaforatools/blob/master/anafora/test/test_anafora_evaluate.py


@pytest.fixture
def annotation():

    return {
        TLink("l1", "A", "B", "BEFORE"),
        TLink("l2", "B", "C", "IS_INCLUDED"),
        TLink("l3", "D", "C", "INCLUDES"),
        TLink("l4", "E", "D", "CONTAINS"),
        TLink("l5", "F", "E", "AFTER"),
        TLink("l6", "G", "H", "BEGINS-ON"),
        TLink("l7", "I", "G", "BEFORE"),
        TLink("l8", "J", "K", "IBEFORE"),
        TLink("l9", "K", "L", "BEGUN_BY"),
        TLink("l10", "L", "K", "BEGINS")

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
        TLink("l1", "A", "B", "BEFORE"),   # (+)
        TLink("l2", "A", "B", "BEFORE"),   #annotation, system duplicate
        TLink("l3", "B", "A", "AFTER"),    # duplicate
        TLink("l4", "B", "E", "CONTAINS"), # (-)
        TLink("l5", "B", "E", "INCLUDES"), # duplicate
        TLink("l6", "B", "F", "BEFORE"),   # (+)
        TLink("l7", "F", "D", "AFTER"),    # (+)
        TLink("l8", "H", "I", "AFTER"),    # (+)
        TLink("l9", "J", "L", "IBEFORE"),  # (+)
        TLink("l10", "K", "L", "BEGUN_BY"), # (+)

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

    recall = temporal_recall(system, annotation)
    assert recall == (4 / 9)


def test_temporal_precision(system, annotation):

    precision = temporal_precision(system, annotation)
    assert precision == (6 / 7)


def test_temporal_awareness(system, annotation):

    awareness = round(temporal_awareness(system, annotation), 3)
    assert awareness == round(24 / 41, 3)
