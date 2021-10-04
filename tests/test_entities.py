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


class TestTimex:

    def test_id(self, timex):
        assert timex.id == "t0"

    def test_is_dct(self, timex):
        assert timex.is_dct

    def test_le(self):
        t1 = Timex({"tid": "t1"})
        t2 = Timex({"tid": "t2"})
        assert t1 < t2


class TestEvent:

    def test_id(self, event):
        assert event.id == "e1"

    def test_le(self):
        e1 = Event({"eid": "e1"})
        e2 = Event({"eid": "e2"})
        assert e1 < e2
