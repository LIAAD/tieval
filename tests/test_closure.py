import pytest

from tieval.links import TLink
from tieval.closure import temporal_closure


@pytest.fixture
def annotation():
    sample1 = {
        TLink("B", "A", "before"),
        TLink("D", "H", "ends"),
        TLink("D", "G", "before"),
        TLink("A", "F", "before"),
        TLink("C", "A", "simultaneous"),
        TLink("B", "E", "after"),
    }

    sample2 = {
        TLink("A", "B", "before"),
        TLink("B", "C", "is_included"),
        TLink("D", "C", "includes"),
        TLink("E", "D", "contains"),
        TLink("F", "E", "after"),
        TLink("G", "H", "begins-on"),
        TLink("I", "G", "before"),
        TLink("J", "K", "ibefore"),
        TLink("K", "L", "begun_by"),
        TLink("L", "K", "begins")
    }

    return sample1, sample2


@pytest.fixture
def closure():
    sample1 = {
        TLink("B", "A", "before"),
        TLink("D", "H", "ends"),
        TLink("D", "G", "before"),
        TLink("A", "F", "before"),
        TLink("C", "A", "simultaneous"),
        TLink("B", "E", "after"),
        TLink("E", "A", "before"),
        TLink("E", "F", "before"),
        TLink("E", "C", "before"),
        TLink("B", "F", "before"),
        TLink("B", "C", "before"),
        TLink("H", "G", "before"),
        TLink("F", "C", "after"),
    }

    sample2 = {
        TLink("A", "B", "before"),
        TLink("A", "F", "before"),
        TLink("B", "F", "before"),
        TLink("C", "B", "includes"),
        TLink("C", "F", "before"),
        TLink("D", "B", "includes"),
        TLink("D", "C", "includes"),
        TLink("D", "F", "before"),
        TLink("E", "B", "includes"),
        TLink("E", "C", "includes"),
        TLink("E", "D", "includes"),
        TLink("E", "F", "before"),
        TLink("G", "H", "begins-on"),
        TLink("I", "G", "before"),
        TLink("I", "H", "before"),
        TLink("J", "L", "ibefore"),
        TLink("K", "L", "begun_by"),
        TLink("J", "K", "ibefore"),
    }

    return sample1, sample2


def test_temporal_closure(annotation, closure):

    for ann, clo in zip(annotation, closure):

        inferred = temporal_closure(ann)

        assert inferred == clo
