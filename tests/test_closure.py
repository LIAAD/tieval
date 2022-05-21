import pytest


from tieval.links import TLink
from tieval.closure import temporal_closure


def test_temporal_closure_1():

    annotation = {
        TLink("B", "A", "before"),
        TLink("D", "H", "ends"),
        TLink("D", "G", "before"),
        TLink("A", "F", "before"),
        TLink("C", "A", "simultaneous"),
        TLink("B", "E", "after"),
    }

    closure = {
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

    inferred = temporal_closure(annotation)
    assert inferred == closure


def test_temporal_closure_2():

    annotation = {
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

    closure = {
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

    inferred = temporal_closure(annotation)
    assert inferred == closure
