from typing import Union

# map interval relations to unique names
_SETTLE_RELATION = {
    "OVERLAP": "OVERLAP",
    "OVERLAPS": "OVERLAP",
    "OVERLAPPED": "OVERLAPPED",
    "BEGINS": "BEGINS",
    "BEFORE": "BEFORE",
    "B": "BEFORE",
    "CONTAINS": "INCLUDES",
    "IDENTITY": "SIMULTANEOUS",
    "EQUAL": "SIMULTANEOUS",
    "AFTER": "AFTER",
    "A": "AFTER",
    "BEGINS-ON": "BEGINS",
    "SIMULTANEOUS": "SIMULTANEOUS",
    "S": "SIMULTANEOUS",
    "INCLUDES": "INCLUDES",
    "I": "INCLUDES",
    "DURING": "SIMULTANEOUS",  # According to TimeML 1.2.1 guidelines
    "ENDS-ON": "ENDS",
    "BEGUN_BY": "BEGUN_BY",
    "STARTED_BY": "BEGUN_BY",
    "ENDED_BY": "ENDED_BY",
    "DURING_INV": "SIMULTANEOUS",   # According to TimeML 1.2.1 guidelines
    "ENDS": "ENDS",
    "IS_INCLUDED": "IS_INCLUDED",
    "II": "IS_INCLUDED",
    "IBEFORE": "IBEFORE",
    "IAFTER": "IAFTER",
    "VAGUE": "VAGUE",
    "V": "VAGUE",
    "NONE": "VAGUE",
    "UNKNOWN": "VAGUE",
    "BEFORE-OR-OVERLAP": "BEFORE-OR-OVERLAP",
    "OVERLAP_BEFORE": "BEFORE-OR-OVERLAP",
    "OVERLAP-OR-AFTER": "OVERLAP-OR-AFTER",
    "OVERLAP_AFTER": "OVERLAP-OR-AFTER",
}

SUPPORTED_RELATIONS = list(_SETTLE_RELATION)

# transitivity table for point relations
POINT_TRANSITIONS = {
    "<": {"<": "<", "=": "<", ">": None, None: None},
    "=": {"<": "<", "=": "=", ">": ">", None: None},
    ">": {">": ">", "=": ">", "<": None, None: None},
    None: {">": None, "=": None, "<": None, None: None},
}

# inverse for each point relation
INVERSE_POINT_RELATION = {"<": ">", ">": "<", "=": "=", None: None}


class Point:
    def __init__(self, value: int = 1):
        self.value = value


class IncompleteRelationError(Exception):
    """Raised when the point relation is incomplete."""

    pass


class PointRelation:
    def __init__(
        self, xs_ys: str = None, xs_ye: str = None, xe_ys: str = None, xe_ye: str = None
    ) -> None:
        """Point relation.
        Every interval relation can be decomposed into four point relations between the entities offsets:
        start of entity x; end of entity x; start of entity y; end of entity y.

        :param xs_ys: Relation between start of x and start of y.
        :type xs_ys: str {"<", "=", ">"}
        :param xs_ye: Relation between start of x and end of y.
        :type xs_ye: str {"<", "=", ">"}
        :param xe_ys: Relation between end of x and start of y.
        :type xe_ys: str {"<", "=", ">"}
        :param xe_ye: Relation between end of x and end of y.
        :type xe_ye: str {"<", "=", ">"}
        """

        self.relation = [xs_ys, xs_ye, xe_ys, xe_ye]
        self.order = self._relative_position()

    def __hash__(self):
        return hash(tuple(self.relation))

    def __repr__(self):
        return f"PointRelation({self.relation})"

    def __str__(self):
        if self.order is None:
            raise IncompleteRelationError("The point relation is incomplete.")

        src_idx, tgt_idx = self.order
        source = "   ".join("*" if i in src_idx else " " for i in range(5))
        target = "   ".join("*" if i in tgt_idx else " " for i in range(5))

        return f"Source {source}\nTarget {target}"

    def __eq__(self, other):
        if isinstance(other, list):
            other = PointRelation(*other)

        return self.relation == other.relation

    def __invert__(self):
        ss, se, es, ee = self.relation

        inverse_relations = [
            INVERSE_POINT_RELATION[ss],
            INVERSE_POINT_RELATION[es],
            INVERSE_POINT_RELATION[se],
            INVERSE_POINT_RELATION[ee],
        ]

        return PointRelation(*inverse_relations)

    def __and__(self, other):
        r1, r2, r3, r4 = self.relation
        r5, r6, r7, r8 = other.relation

        ss = POINT_TRANSITIONS[r1][r5] or POINT_TRANSITIONS[r2][r7]
        se = POINT_TRANSITIONS[r1][r6] or POINT_TRANSITIONS[r2][r8]
        es = POINT_TRANSITIONS[r3][r5] or POINT_TRANSITIONS[r4][r7]
        ee = POINT_TRANSITIONS[r3][r6] or POINT_TRANSITIONS[r4][r8]

        return PointRelation(ss, se, es, ee)

    def __iter__(self):
        return self.relation.__iter__()

    @property
    def relation(self):
        return self._relation

    @relation.setter
    def relation(self, relations):
        # if the complete relation inferred a point relation different of the original
        # relations, the point relation is inconsistent a.k.a. not valid
        complete_relations = self._complete(*relations)
        for rel, inferred_rel in zip(relations, complete_relations):
            if rel and rel != inferred_rel:
                raise ValueError(f"Point relation {relations} is not valid.")

        self._relation = complete_relations

    @staticmethod
    def _complete(start_start, start_end, end_start, end_end):
        relations = [
            ("ss", start_start, "ts"),  # source start, target start
            ("ss", start_end, "te"),  # source start, target end
            ("se", end_start, "ts"),  # source end, target start
            ("se", end_end, "te"),  # source end, target end
            ("ss", "<", "se"),  # source start, source end
            ("ts", "<", "te"),  # target start, target end
        ]

        # creat a dictionary with the input relations
        relations_dict = {}
        for src, rel, tgt in relations:
            if rel:
                relations_dict[(src, tgt)] = rel
                relations_dict[(tgt, src)] = INVERSE_POINT_RELATION[rel]

        # infer the relation between all end points
        flag = True
        while flag:
            inferred_relations = {}
            # infer all possible relations
            for (p1, p2), rel12 in relations_dict.items():
                for (p3, p4), rel34 in relations_dict.items():
                    cond1 = p2 == p3 and p1 != p4
                    cond2 = (p1, p4) not in relations_dict
                    cond3 = (p4, p1) not in relations_dict
                    rel = POINT_TRANSITIONS[rel12][rel34]
                    if cond1 and cond2 and cond3 and rel:
                        inferred_relations[(p1, p4)] = rel
                        inferred_relations[(p4, p1)] = INVERSE_POINT_RELATION[rel]

            if not inferred_relations:
                flag = False

            relations_dict.update(inferred_relations)

        return [relations_dict.get((src, tgt)) for src, rel, tgt in relations[:4]]

    def _relative_position(self):
        """
        Compute the relative position of the points.
        """

        s_src, e_src = Point(), Point()
        s_tgt, e_tgt = Point(), Point()

        relations = [
            (s_src, self.relation[0], s_tgt),
            (s_src, self.relation[1], e_tgt),
            (e_src, self.relation[2], s_tgt),
            (e_src, self.relation[3], e_tgt),
            (s_src, "<", e_src),
            (s_tgt, "<", e_tgt),
        ]

        for src, rel, tgt in relations:
            if rel == "<":
                tgt.value += 1

            elif rel == ">":
                src.value += 1

            elif rel is None:
                return None

        return [[s_src.value, e_src.value], [s_tgt.value, e_tgt.value]]


# Mapping from interval relation names to point relations.
# For example, BEFORE means that the first interval"s end is before the second
# interval"s start
_INTERVAL_TO_POINT_RELATION = {
    "BEFORE": PointRelation(xe_ys="<"),
    "AFTER": PointRelation(xs_ye=">"),
    "IBEFORE": PointRelation(xe_ys="="),
    "IAFTER": PointRelation(xs_ye="="),
    "INCLUDES": PointRelation(xs_ys="<", xe_ye=">"),
    "IS_INCLUDED": PointRelation(xs_ys=">", xe_ye="<"),
    "BEGINS": PointRelation(xs_ys="=", xe_ye="<"),
    "BEGUN_BY": PointRelation(xs_ys="=", xe_ye=">"),
    "ENDS": PointRelation(xs_ys=">", xe_ye="="),
    "ENDED_BY": PointRelation(xs_ys="<", xe_ye="="),
    "SIMULTANEOUS": PointRelation(xs_ys="=", xe_ye="="),
    "OVERLAP": PointRelation(xs_ys="<", xe_ys=">", xe_ye="<"),
    "OVERLAPPED": PointRelation(xs_ys=">", xs_ye="<", xe_ye=">"),
    "VAGUE": PointRelation(),
    "BEFORE-OR-OVERLAP": PointRelation(xs_ys="<", xe_ye="<"),
    "OVERLAP-OR-AFTER": PointRelation(xs_ys=">", xe_ye=">"),
}


class TemporalRelation:
    def __init__(self, relation: Union[str, list, dict, PointRelation]):
        self.point = self._handle(relation)
        self._interval = None

    def __repr__(self) -> str:
        return f"TemporalRelation({self.interval})"

    def __str__(self) -> str:
        if self.interval:
            return f"{self.interval}"
        else:
            return f"{self.point}"

    def __invert__(self):
        return TemporalRelation(~self.point)

    def __and__(self, other):
        return TemporalRelation(self.point & other.point)

    def __eq__(self, other):
        return self.point == other.point

    def __hash__(self):
        return hash((self.point, self.interval))

    def is_complete(self):
        # A relation is considered complete if the point relation inferred can be mapped to one of the interval
        # relations.

        if self.interval:
            return True

        return False

    @property
    def interval(self):
        if self._interval is None:
            for itr, pnt in _INTERVAL_TO_POINT_RELATION.items():
                if pnt == self.point:
                    self._interval = itr

        return self._interval

    @staticmethod
    def _handle(relation):
        if isinstance(relation, str):
            interval = _SETTLE_RELATION.get(relation.upper())
            if interval is None:
                raise ValueError(f"Interval relation {relation} not supported.")

            point = _INTERVAL_TO_POINT_RELATION[interval]

        elif isinstance(relation, list):
            point = PointRelation(*relation)

        elif isinstance(relation, dict):
            point = PointRelation(**relation)

        elif isinstance(relation, PointRelation):
            point = relation

        elif isinstance(relation, TemporalRelation):
            point = relation.point

        else:
            raise TypeError("Argument type is not supported.")

        return point
