import pytest

from text2timeline.temporal_relation import PointRelation
from text2timeline.temporal_relation import TemporalRelation


class TestPointRelation:

    def test_relation(self, point_relations):
        """Check if point relation inference is working."""

        assert PointRelation(end_start="<").relation == ["<", "<", "<", "<"]
        assert PointRelation(start_end=">").relation == [">", ">", ">", ">"]
        assert PointRelation(end_start="=").relation == ["<", "<", "=", "<"]
        assert PointRelation(start_end="=").relation == [">", "=", ">", ">"]
        assert PointRelation(start_start="<", end_end=">").relation == ["<", "<", ">", ">"]
        assert PointRelation(start_start=">", end_end="<").relation == [">", "<", ">", "<"]
        assert PointRelation(start_start="=").relation == ["=", "<", ">", None]
        assert PointRelation(end_end="=").relation == [None, "<", ">", "="]
        assert PointRelation(start_start="=", end_end="<").relation == ["=", "<", ">", "<"]
        assert PointRelation(start_start="=", end_end=">").relation == ["=", "<", ">", ">"]
        assert PointRelation(start_start=">", end_end="=").relation == [">", "<", ">", "="]
        assert PointRelation(start_start="<", end_end="=").relation == ["<", "<", ">", "="]
        assert PointRelation(start_start="=", end_end="=").relation == ["=", "<", ">", "="]
        assert PointRelation(start_start="<", end_start=">", end_end="<").relation == ["<", "<", ">", "<"]
        assert PointRelation().relation == [None, None, None, None]
        assert PointRelation(start_start="<", end_end="<").relation == ["<", "<", None, "<"]
        assert PointRelation(start_start=">", end_end=">").relation == [">", None, ">", ">"]

    def test_order(self):

        assert PointRelation(end_start="<").order == [[1, 2], [3, 4]]
        assert PointRelation(start_end=">").order == [[3, 4], [1, 2]]
        assert PointRelation(end_start="=").order == [[1, 2], [2, 4]]
        assert PointRelation(start_end="=").order == [[2, 4], [1, 2]]
        assert PointRelation(start_start="<", end_end=">").order == [[1, 4], [2, 3]]
        assert PointRelation(start_start=">", end_end="<").order == [[2, 3], [1, 4]]
        assert PointRelation(start_start="=", end_end="<").order == [[1, 3], [1, 4]]
        assert PointRelation(start_start="=", end_end=">").order == [[1, 4], [1, 3]]
        assert PointRelation(start_start=">", end_end="=").order == [[2, 3], [1, 3]]
        assert PointRelation(start_start="<", end_end="=").order == [[1, 3], [2, 3]]
        assert PointRelation(start_start="=", end_end="=").order == [[1, 3], [1, 3]]
        assert PointRelation(start_start="<", end_start=">", end_end="<").order == [[1, 3], [2, 4]]

        # TODO: what should be the order when one end point is not defined?
        assert PointRelation(start_start="=").order == [[1, 3], [1, 3]]
        assert PointRelation(end_end="=").order == [[1, 3], [1, 3]]
        assert PointRelation().order == [[1, 2], [1, 2]]
        assert PointRelation(start_start="<", end_end="<").order == [[], []]
        assert PointRelation(start_start=">", end_end=">").order == [[], []]




def test_temporal_relation_inversion():

    assert (~TemporalRelation("AFTER")).interval == "BEFORE"
    assert (~TemporalRelation("BEFORE")).interval == "AFTER"
    assert (~TemporalRelation("BEGINS")).interval == "BEGUN_BY"
    assert (~TemporalRelation("BEGINS-ON")).interval == "BEGINS-ON"
    assert (~TemporalRelation("BEGUN_BY")).interval == "BEGINS"
    assert (~TemporalRelation("ENDED_BY")).interval == "ENDS"
    assert (~TemporalRelation("ENDS")).interval == "ENDED_BY"
    assert (~TemporalRelation("ENDS-ON")).interval == "ENDS-ON"
    assert (~TemporalRelation("IAFTER")).interval == "IBEFORE"
    assert (~TemporalRelation("IBEFORE")).interval == "IAFTER"
    assert (~TemporalRelation("INCLUDES")).interval == "IS_INCLUDED"
    assert (~TemporalRelation("IS_INCLUDED")).interval == "INCLUDES"
    assert (~TemporalRelation("SIMULTANEOUS")).interval == "SIMULTANEOUS"
    assert (~TemporalRelation("OVERLAP")).interval == "OVERLAPPED"
    assert (~TemporalRelation("OVERLAPPED")).interval == "OVERLAP"
    assert (~TemporalRelation("VAGUE")).interval == "VAGUE"
