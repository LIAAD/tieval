import pytest

from text2timeline.temporal_relation import PointRelation
from text2timeline.temporal_relation import _INTERVAL_TO_POINT_RELATION

_INTERVAL_TO_POINT_COMPLETE = {
    "BEFORE": ["<", "<", "<", "<"],
    "AFTER": [">", ">", ">", ">"],
    "IBEFORE": ["<", "<", "=", "<"],
    "IAFTER": [">", "=", ">", ">"],
    "INCLUDES": ["<", "<", ">", ">"],
    "IS_INCLUDED": [">", "<", ">", "<"],
    "BEGINS-ON": ["=", "<", ">", None],
    "ENDS-ON": [None, "<", ">", "="],
    "BEGINS": ["=", "<", ">", "<"],
    "BEGUN_BY": ["=", "<", ">", ">"],
    "ENDS": [">", "<", ">", "="],
    "ENDED_BY": ["<", "<", ">", "="],
    "SIMULTANEOUS": ["=", "<", ">", "="],
    "OVERLAP": ["<", "<", ">", "<"],
    "VAGUE": [None, None, None, None],
    "BEFORE-OR-OVERLAP": ["<", "<", None, "<"],
    "OVERLAP-OR-AFTER": [">", None, ">", ">"]
}


def test_point_relation():

    for int_rel, pnt_rel in _INTERVAL_TO_POINT_COMPLETE.items():
        pnt_rel_resolved = _INTERVAL_TO_POINT_RELATION[int_rel].relation[:4]
        pnt_rel_resolved = [rel for _, rel, _ in pnt_rel_resolved]
        assert pnt_rel == pnt_rel_resolved, f"There is a problem with the way the {int_rel} is resolved."


_INVERSE_INTERVAL_RELATION = {
    "AFTER": "BEFORE",
    "BEFORE": "AFTER",
    "BEGINS": "BEGUN_BY",
    "BEGINS-ON": "BEGINS-ON",
    "BEGUN_BY": "BEGINS",
    "ENDED_BY": "ENDS",
    "ENDS": "ENDED_BY",
    "ENDS-ON": "ENDS-ON",
    "IAFTER": "IBEFORE",
    "IBEFORE": "IAFTER",
    "INCLUDES": "IS_INCLUDED",
    "IS_INCLUDED": "INCLUDES",
    "SIMULTANEOUS": "SIMULTANEOUS",
    "OVERLAP": "OVERLAP",
    "VAGUE": "VAGUE"
}

