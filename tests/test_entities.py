import pytest

from tieval.base import Event
from tieval.base import Timex


@pytest.fixture
def event():
    return Event(
        id="e1",
        class_="OCCURRENCE",
        text="expansion",
        offsets=(82, 91),
        eiid="ei50001",
        tense="NONE",
        aspect="NONE",
        polarity="POS",
        pos="NOUN"
    )


@pytest.fixture
def timex():
    return Timex(
        id="t0",
        type="DATE",
        value="1998-03-03",
        temporal_function="false",
        function_in_document="CREATION_TIME"
    )


class TestTimex:

    def test_id(self, timex):
        assert timex.id == "t0"

    def test_is_dct(self, timex):
        assert timex.is_dct

    def test_le(self):
        t1 = Timex(id="t1")
        t2 = Timex(id="t2")
        assert t1 < t2


class TestEvent:

    def test_id(self, event):
        assert event.id == "e1"

    def test_le(self):
        e1 = Event(id="e1")
        e2 = Event(id="e2")
        assert e1 < e2

    def test_is_dct(self, event):
        assert not event.is_dct
