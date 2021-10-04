
from text2timeline.links import TLink
from text2timeline.closure import temporal_closure

import pytest


@pytest.fixture
def annotation():
    sample1 = {
        TLink("l1", "B", "A", "before"),
        TLink("l2", "D", "H", "ends"),
        TLink("l3", "D", "G", "before"),
        TLink("l4", "A", "F", "before"),
        TLink("l5", "C", "A", "simultaneous"),
        TLink("l6", "B", "E", "after"),
    }

    sample2 = {
        TLink("l1", "A", "B", "before"),
        TLink("l1", "B", "C", "is_included"),
        TLink("l1", "D", "C", "includes"),
        TLink("l1", "E", "D", "contains"),
        TLink("l1", "F", "E", "after"),
        TLink("l1", "G", "H", "begins-on"),
        TLink("l1", "I", "G", "before"),
        TLink("l1", "J", "K", "ibefore"),
        TLink("l1", "K", "L", "begun_by"),
        TLink("l1", "L", "K", "begins")
    }

    return sample1, sample2


@pytest.fixture
def closure():
    sample1 = {
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

    sample2 = {
        TLink("li", "A", "B", "before"),
        TLink("li", "A", "F", "before"),
        TLink("li", "B", "F", "before"),
        TLink("li", "C", "B", "includes"),
        TLink("li", "C", "F", "before"),
        TLink("li", "D", "B", "includes"),
        TLink("li", "D", "C", "includes"),
        TLink("li", "D", "F", "before"),
        TLink("li", "E", "B", "includes"),
        TLink("li", "E", "C", "includes"),
        TLink("li", "E", "D", "includes"),
        TLink("li", "E", "F", "before"),
        TLink("l1", "G", "H", "begins-on"),
        TLink("li", "I", "G", "before"),
        TLink("li", "I", "H", "before"),
        TLink("li", "J", "L", "ibefore"),
        TLink("l1", "K", "L", "begun_by"),
        TLink("li", "J", "K", "ibefore"),
    }

    return sample1, sample2


def test_temporal_closure(annotation, closure):

    for ann, clo in zip(annotation, closure):

        inferred = temporal_closure(ann)

        assert inferred == clo
