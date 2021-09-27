import pytest

from text2timeline.base import Event
from text2timeline.base import Timex
from text2timeline.base import TLink


@pytest.fixture
def tlink():

    source = Event({
        "eid": "e1"
    })

    target = Timex({
        "tid": "t0"
    })

    return TLink(
        id="l70",
        source=source,
        target=target,
        relation="BEFORE"
    )


@pytest.fixture
def tlink0():
    source = Event({
        "eid": "e2",
    })

    target = Event({
        "eid": "e3",
    })

    return TLink(
        id="l6",
        source=source,
        target=target,
        relation="INCLUDES"
    )


class TestTLink:

    def test_id(self, tlink):
        assert tlink.id == "l70"

    def test_relation(self, tlink):

        relation = tlink.relation

        assert relation.interval == "BEFORE"
        assert relation.point == ["<", "<", "<", "<"]

    def test_invert(self, tlink):

        inverted_tlink = ~tlink
        relation = inverted_tlink.relation

        assert relation.interval == "AFTER"
        assert relation.point == [">", ">", ">", ">"]

    # TODO: check equality assertion of temporal relations.
    def test_equal(self, tlink, tlink0):
        assert tlink == tlink
        # assert tlink == ~tlink
        assert tlink != tlink0
