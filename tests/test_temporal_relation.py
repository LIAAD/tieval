
from tieval.temporal_relation import PointRelation
from tieval.temporal_relation import TemporalRelation


class TestPointRelation:

    def test_relation(self):
        """Check if point relation inference is working."""

        assert PointRelation(xe_ys="<").relation == ["<", "<", "<", "<"]
        assert PointRelation(xs_ye=">").relation == [">", ">", ">", ">"]
        assert PointRelation(xe_ys="=").relation == ["<", "<", "=", "<"]
        assert PointRelation(xs_ye="=").relation == [">", "=", ">", ">"]
        assert PointRelation(xs_ys="<", xe_ye=">").relation == ["<", "<", ">", ">"]
        assert PointRelation(xs_ys=">", xe_ye="<").relation == [">", "<", ">", "<"]
        assert PointRelation(xs_ys="=").relation == ["=", "<", ">", None]
        assert PointRelation(xe_ye="=").relation == [None, "<", ">", "="]
        assert PointRelation(xs_ys="=", xe_ye="<").relation == ["=", "<", ">", "<"]
        assert PointRelation(xs_ys="=", xe_ye=">").relation == ["=", "<", ">", ">"]
        assert PointRelation(xs_ys=">", xe_ye="=").relation == [">", "<", ">", "="]
        assert PointRelation(xs_ys="<", xe_ye="=").relation == ["<", "<", ">", "="]
        assert PointRelation(xs_ys="=", xe_ye="=").relation == ["=", "<", ">", "="]
        assert PointRelation(xs_ys="<", xe_ys=">", xe_ye="<").relation == ["<", "<", ">", "<"]
        assert PointRelation().relation == [None, None, None, None]
        assert PointRelation(xs_ys="<", xe_ye="<").relation == ["<", "<", None, "<"]
        assert PointRelation(xs_ys=">", xe_ye=">").relation == [">", None, ">", ">"]

    def test_order(self):

        assert PointRelation(xe_ys="<").order == [[1, 2], [3, 4]]
        assert PointRelation(xs_ye=">").order == [[3, 4], [1, 2]]
        assert PointRelation(xe_ys="=").order == [[1, 2], [2, 4]]
        assert PointRelation(xs_ye="=").order == [[2, 4], [1, 2]]
        assert PointRelation(xs_ys="<", xe_ye=">").order == [[1, 4], [2, 3]]
        assert PointRelation(xs_ys=">", xe_ye="<").order == [[2, 3], [1, 4]]
        assert PointRelation(xs_ys="=", xe_ye="<").order == [[1, 3], [1, 4]]
        assert PointRelation(xs_ys="=", xe_ye=">").order == [[1, 4], [1, 3]]
        assert PointRelation(xs_ys=">", xe_ye="=").order == [[2, 3], [1, 3]]
        assert PointRelation(xs_ys="<", xe_ye="=").order == [[1, 3], [2, 3]]
        assert PointRelation(xs_ys="=", xe_ye="=").order == [[1, 3], [1, 3]]
        assert PointRelation(xs_ys="<", xe_ys=">", xe_ye="<").order == [[1, 3], [2, 4]]

        # TODO: what should be the order when one end point is not defined?
        # assert PointRelation(start_start="=").order == [[1, 3], [1, 3]]
        # assert PointRelation(end_end="=").order == [[1, 3], [1, 3]]
        # assert PointRelation().order == [[1, 2], [1, 2]]
        # assert PointRelation(start_start="<", end_end="<").order == [[], []]
        # assert PointRelation(start_start=">", end_end=">").order == [[], []]


class TestTemporalRelation:

    def test_input(self):

        assert TemporalRelation("before").interval == "BEFORE"
        assert TemporalRelation(["<", "<", "<", "<"]).interval == "BEFORE"
        assert TemporalRelation({"xe_ys": "<"}).interval == "BEFORE"
        assert TemporalRelation(PointRelation(xe_ys="<")).interval == "BEFORE"
        assert TemporalRelation(TemporalRelation("before")).interval == "BEFORE"

    def test_inference(self):

        r1 = TemporalRelation("before")
        r2 = TemporalRelation("before")

        assert r1 & r2 == TemporalRelation("before")

    def test_temporal_relation_inversion(self):

        assert (~TemporalRelation("AFTER")).interval == "BEFORE"
        assert (~TemporalRelation("BEFORE")).interval == "AFTER"
        assert (~TemporalRelation("BEGINS")).interval == "BEGUN_BY"
        assert (~TemporalRelation("BEGUN_BY")).interval == "BEGINS"
        # assert (~TemporalRelation("BEGINS-ON")).interval == "BEGINS-ON"
        assert (~TemporalRelation("ENDED_BY")).interval == "ENDS"
        assert (~TemporalRelation("ENDS")).interval == "ENDED_BY"
        # assert (~TemporalRelation("ENDS-ON")).interval == "ENDS-ON"
        assert (~TemporalRelation("IAFTER")).interval == "IBEFORE"
        assert (~TemporalRelation("IBEFORE")).interval == "IAFTER"
        assert (~TemporalRelation("INCLUDES")).interval == "IS_INCLUDED"
        assert (~TemporalRelation("IS_INCLUDED")).interval == "INCLUDES"
        assert (~TemporalRelation("SIMULTANEOUS")).interval == "SIMULTANEOUS"
        assert (~TemporalRelation("OVERLAP")).interval == "OVERLAPPED"
        assert (~TemporalRelation("OVERLAPPED")).interval == "OVERLAP"
        assert (~TemporalRelation("VAGUE")).interval == "VAGUE"


