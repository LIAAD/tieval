import pytest

from text2timeline.base import Event
from text2timeline.base import Timex
from text2timeline.base import TLink


@pytest.fixture
def tlink():

    return TLink(
        id="l70",
        source="e1",
        target="t0",
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

    def test_equal(self, tlink, tlink0):
        assert tlink == tlink
        assert tlink == ~tlink
        assert tlink != tlink0

    def test_inference(self):

        ab = TLink(id='l0', source="A", target="B", relation="after")
        ac = TLink(id='l1', source="A", target="C", relation="simultaneous")
        ba = TLink(id='l2', source="B", target="A", relation="before")
        ca = TLink(id='l3', source="C", target="A", relation="simultaneous")

        bc = TLink(id='li1', source="B", target="C", relation="before")

        assert ab & ac == bc
        assert ab & ca == bc
        assert ba & ac == bc
        assert ab & ca == bc
