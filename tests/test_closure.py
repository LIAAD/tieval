from tieval.links import TLink
from tieval.closure import temporal_closure
import tieval.datasets as datasets


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
    assert set(inferred) == closure


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


def test_temporal_closure_3():

    annotation = {
        TLink("A", "B", "before"),
        TLink("A", "B", "before"),
        TLink("B", "A", "after"),
        TLink("B", "E", "contains"),
        TLink("B", "E", "includes"),
        TLink("B", "F", "before"),
        TLink("F", "D", "after"),
        TLink("H", "I", "after"),
        TLink("J", "L", "ibefore"),
        TLink("K", "L", "begun_by"),
    }

    closure = {
        TLink("A", "B", "before"),
        TLink("A", "E", "before"),
        TLink("A", "F", "before"),
        TLink("B", "E", "includes"),
        TLink("B", "F", "before"),
        TLink("D", "F", "before"),
        TLink("E", "F", "before"),
        TLink("H", "I", "after"),
        TLink("J", "L", "ibefore"),
        TLink("K", "L", "begun_by"),
        TLink("J", "K", "ibefore"),
    }

    inferred = temporal_closure(annotation)

    assert inferred == closure


def test_temporal_closure_tempeavl_3_test():
    dataset = datasets.read("tempeval_3")
    n_tlinks = sum([len(doc.tlinks) for doc in dataset.test])
    n_tlinks_closure = sum([len(doc.temporal_closure) for doc in dataset.test])
    
    assert n_tlinks_closure >= n_tlinks
    assert n_tlinks == 929
    assert n_tlinks_closure == 3458


def test_temporal_closure_tempeavl_3_train():
    dataset = datasets.read("tempeval_3")
    n_tlinks = sum([len(doc.tlinks) for doc in dataset.train])
    n_tlinks_closure = sum([len(doc.temporal_closure) for doc in dataset.train])
    
    assert n_tlinks_closure >= n_tlinks
    assert n_tlinks == 10952
    assert n_tlinks_closure == 77747
