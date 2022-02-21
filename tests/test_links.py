import pytest

from tieval.base import TLink


@pytest.fixture
def tlink():

    return TLink(
        id="l70",
        source="e1",
        target="t0",
        relation="BEFORE"
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

    def test_equal(self, tlink):
        assert tlink == tlink
        assert tlink == ~tlink
        assert tlink != TLink("e2", "e3", "INCLUDES")

    def test_inference(self):

        ab = TLink("A", "B", "after")
        ac = TLink("A", "C", "simultaneous")
        ba = TLink("B", "A", "before")
        ca = TLink("C", "A", "simultaneous")

        bc = TLink("B", "C", "before")

        assert ab & ac == bc
        assert ab & ca == bc
        assert ba & ac == bc
        assert ab & ca == bc

        # test the case where there is nothing to be inferred.
        assert ab & ba is None

        # test the case where the inferred relation is VAGUE
        assert ab & bc is None

        # specific case #1
        dh = TLink("D", "H", "ends")
        dg = TLink("D", "G", "before")
        hg = TLink("H", "G", "before")
        assert dh & dg == hg

        gh = TLink("G", "H", "after")
        assert dh & dg == gh

        # specific case #2
        ca = TLink("C", "A", "simultaneous")
        af = TLink("A", "F", "before")
        cf = TLink("C", "F", "before")
        assert ca & af == cf

    def test_hash(self):
        tl1 = TLink("A", "B", "before")
        tl2 = TLink("B", "A", "after")

        assert hash(tl1) == hash(tl2)
