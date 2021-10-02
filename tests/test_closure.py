
from text2timeline.links import TLink
from text2timeline.closure import temporal_closure

import pytest


@pytest.fixture
def tlinks():

    return {
        TLink("l1", "B", "A", "before"),
        TLink("l2", "D", "H", "ends"),
        TLink("l3", "D", "G", "before"),
        TLink("l4", "A", "F", "before"),
        TLink("l5", "C", "A", "simultaneous"),
        TLink("l6", "B", "E", "after"),
    }


@pytest.fixture
def tlinks_closure():
    return {
        TLink("l1", "B", "A", "before"),
        TLink("l2", "D", "H", "ends"),
        TLink("l3", "D", "G", "before"),
        TLink("l4", "A", "F", "before"),
        TLink("l5", "C", "A", "simultaneous"),
        TLink("l6", "B", "E", "after"),

        TLink("l7", "E", "A", "before"),
        TLink("l8", "E", "F", "before"),
        TLink("l9", "E", "C", "before"),
        TLink("l10", "B", "F", "before"),
        TLink("l11", "B", "C", "before"),
        TLink("l12", "H", "G", "before"),
        TLink("l13", "F", "C", "after"),
    }


def test_temporal_closure(tlinks, tlinks_closure):

    assert temporal_closure(tlinks) == tlinks_closure
