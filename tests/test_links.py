import pytest

from text2timeline.base import TLink


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
        assert tlink != TLink("l6", "e2", "e3", "INCLUDES")

    def test_inference(self):

        ab = TLink("l0", "A", "B", "after")
        ac = TLink("l1", "A", "C", "simultaneous")
        ba = TLink("l2", "B", "A", "before")
        ca = TLink("l3", "C", "A", "simultaneous")

        bc = TLink("li1", "B", "C", "before")

        assert ab & ac == bc
        assert ab & ca == bc
        assert ba & ac == bc
        assert ab & ca == bc

        # test the case where there is nothing to be inferred.
        assert ab & ba is None

        # test the case where the inferred relation is VAGUE
        assert ab & bc is None

        # specific case #1
        dh = TLink("l2", "D", "H", "ends")
        dg = TLink("l3", "D", "G", "before")
        hg = TLink("l3", "H", "G", "before")
        assert dh & dg == hg

        gh = TLink("l3", "G", "H", "after")
        assert dh & dg == gh

        # specific case #2
        ca = TLink("l2", "C", "A", "simultaneous")
        af = TLink("l3", "A", "F", "before")
        cf = TLink("l3", "C", "F", "before")
        assert ca & af == cf

    def test_hash(self):
        tl1 = TLink("l1", "A", "B", "before")
        tl2 = TLink("l2", "B", "A", "after")

        assert hash(tl1) == hash(tl2)
