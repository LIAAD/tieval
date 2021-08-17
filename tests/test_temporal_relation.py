import pytest

from text2timeline.temporal_relation import PointRelation
from text2timeline.temporal_relation import TemporalRelation


def test_point_relation_inference():
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
