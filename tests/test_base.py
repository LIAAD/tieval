import pytest

from text2timeline.base import Event
from text2timeline.base import Timex
from text2timeline.base import TLink


@pytest.fixture
def event():
    return Event({
        "eid": "e1",
        "class": "OCCURRENCE",
        "text": "expansion",
        "endpoints": (82, 91),
        "eventID": "e1",
        "eiid": "ei50001",
        "tense": "NONE",
        "aspect": "NONE",
        "polarity": "POS",
        "pos": "NOUN"
    })


@pytest.fixture
def timex():
    return Timex({
        "tid": "t0",
        "type": "DATE",
        "value": "1998-03-03",
        "temporalFunction": "false",
        "functionInDocument": "CREATION_TIME"
    })


@pytest.fixture
def tlink(event, timex):
    return TLink(
        id="l70",
        source=event,
        target=timex,
        relation="BEFORE"
    )


class TestTimex:

    def test_id(self, timex):
        assert timex.id == "t0"

    def test_is_dct(self, timex):
        assert timex.is_dct


class TestEvent:

    def test_id(self, event):
        assert event.id == "e1"


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
