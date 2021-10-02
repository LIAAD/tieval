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


@pytest.fixture
def tlink0():

    return TLink(
        id="l6",
        source="e2",
        target="e3",
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

        ab = TLink("l0", "A", "B", "after")
        ac = TLink("l1", "A", "C", "simultaneous")
        ba = TLink("l2", "B", "A", "before")
        ca = TLink("l3", "C", "A", "simultaneous")

        bc = TLink("li1", "B", "C", "before")

        assert ab & ac == bc
        assert ab & ca == bc
        assert ba & ac == bc
        assert ab & ca == bc

        assert ab & ba is None

        # specific case #1
        dh = TLink("l2", "D", "H", "ends")
        dg = TLink("l3", "D", "G", "before")

        hg = TLink("l3", "H", "G", "before")
        assert dh & dg == hg

        gh = TLink("l3", "G", "H", "after")
        assert dh & dg == gh

        # specific case #2
        # C ---SIMULTANEOUS--> A A ---BEFORE--> F
        ca = TLink("l2", "C", "A", "simultaneous")
        af = TLink("l3", "A", "F", "before")

        cf = TLink("l3", "C", "F", "before")
        assert ca & af == cf

        # test the case where the inferred relation is VAGUE
        assert ab & bc is None




