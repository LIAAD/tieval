from typing import Union

# map interval relations to unique names
_SETTLE_RELATION = {
    "OVERLAP": "OVERLAP",
    "OVERLAPPED": "OVERLAPPED",
    "BEGINS": "BEGINS",
    "BEFORE": "BEFORE",
    "B": "BEFORE",
    "CONTAINS": "INCLUDES",
    "IDENTITY": "SIMULTANEOUS",
    "EQUAL": "SIMULTANEOUS",
    "AFTER": "AFTER",
    "A": "AFTER",
    "BEGINS-ON": "BEGINS-ON",
    "SIMULTANEOUS": "SIMULTANEOUS",
    "S": "SIMULTANEOUS",
    "INCLUDES": "INCLUDES",
    "I": "INCLUDES",
    "DURING": "SIMULTANEOUS",
    "ENDS-ON": "ENDS-ON",
    "BEGUN_BY": "BEGUN_BY",
    "ENDED_BY": "ENDED_BY",
    "DURING_INV": "SIMULTANEOUS",
    "ENDS": "ENDS",
    "IS_INCLUDED": "IS_INCLUDED",
    "II": "IS_INCLUDED",
    "IBEFORE": "IBEFORE",
    "IAFTER": "IAFTER",
    "VAGUE": "VAGUE",
    "V": "VAGUE",
    "BEFORE-OR-OVERLAP": "BEFORE-OR-OVERLAP",
    "OVERLAP-OR-AFTER": "OVERLAP-OR-AFTER"
}

# transitivity table for point relations
_POINT_TRANSITIONS = {
    "<": {"<": "<", "=": "<", ">": None, None: None},
    "=": {"<": "<", "=": "=", ">": ">", None: None},
    ">": {">": ">", "=": ">", "<": None, None: None},
    None: {">": None, "=": None, "<": None, None: None}
}

# inverse for each point relation
_INVERSE_POINT_RELATION = {
    "<": ">",
    ">": "<",
    "=": "=",
    None: None
}


class PointRelation:

    def __init__(self,
                 start_start: str = None,
                 start_end: str = None,
                 end_start: str = None,
                 end_end: str = None) -> None:

        self.relation = [start_start, start_end, end_start, end_end]

    def __repr__(self):
        return f"PointRelation({self.relation})"

    def __eq__(self, other):
        return self.relation == other.relation

    def __invert__(self):

        ss, se, es, ee = self.relation

        inverse_relations = [
            _INVERSE_POINT_RELATION[ss],
            _INVERSE_POINT_RELATION[es],
            _INVERSE_POINT_RELATION[se],
            _INVERSE_POINT_RELATION[ee],
        ]

        return PointRelation(*inverse_relations)

    def __and__(self, other):
        result = [_POINT_TRANSITIONS[self_relation][other_relation]
                  for self_relation, other_relation in zip(self.relation, other.relation)]
        return PointRelation(*result)

    @property
    def relation(self):
        return self._relation

    @relation.setter
    def relation(self, relations):

        # if the complete relation inferred a point relation different than the original
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
                relations_dict[(tgt, src)] = _INVERSE_POINT_RELATION[rel]

        # infer the relation between all end points
        flag = True
        while flag:
            inferred_relations = {}
            # infer all possible relations
            for (p1, p2), rel12 in relations_dict.items():
                for (p3, p4), rel34 in relations_dict.items():
                    cond1 = (p2 == p3 and p1 != p4)
                    cond2 = (p1, p4) not in relations_dict
                    cond3 = (p4, p1) not in relations_dict
                    rel = _POINT_TRANSITIONS[rel12][rel34]
                    if cond1 and cond2 and cond3 and rel:
                        inferred_relations[(p1, p4)] = rel
                        inferred_relations[(p4, p1)] = _INVERSE_POINT_RELATION[rel]

            if not inferred_relations:
                flag = False

            relations_dict.update(inferred_relations)

        return [relations_dict.get((src, tgt)) for src, rel, tgt in relations[:4]]


class IntervalRelation:

    def __init__(self, relation: str) -> None:
        self.relation = relation

    def __eq__(self, other):
        return self.relation == other.relation

    @property
    def relation(self):
        return self._relation

    @relation.setter
    def relation(self, relation):

        inferred_relation = _SETTLE_RELATION.get(relation.upper())
        if inferred_relation is None:
            raise ValueError(f"Interval relation {relation} not supported.")

        self._relation = inferred_relation


# Mapping from interval relation names to point relations.
# For example, BEFORE means that the first interval"s end is before the second interval"s start
_INTERVAL_TO_POINT_RELATION = {
    "BEFORE": PointRelation(end_start="<"),
    "AFTER": PointRelation(start_end=">"),
    "IBEFORE": PointRelation(end_start="="),
    "IAFTER": PointRelation(start_end="="),
    "INCLUDES": PointRelation(start_start="<", end_end=">"),
    "IS_INCLUDED": PointRelation(start_start=">", end_end="<"),
    "BEGINS-ON": PointRelation(start_start="="),
    "ENDS-ON": PointRelation(end_end="="),
    "BEGINS": PointRelation(start_start="=", end_end="<"),
    "BEGUN_BY": PointRelation(start_start="=", end_end=">"),
    "ENDS": PointRelation(start_start=">", end_end="="),
    "ENDED_BY": PointRelation(start_start="<", end_end="="),
    "SIMULTANEOUS": PointRelation(start_start="=", end_end="="),
    "OVERLAP": PointRelation(start_start="<", end_start=">", end_end="<"),
    "OVERLAPPED": PointRelation(start_start=">", start_end="<", end_end=">"),
    "VAGUE": PointRelation(),
    "BEFORE-OR-OVERLAP": PointRelation(start_start="<", end_end="<"),
    "OVERLAP-OR-AFTER": PointRelation(start_start=">", end_end=">")
}


class RelationHandler:

    @staticmethod
    def handle(relation):

        if isinstance(relation, str):
            interval = IntervalRelation(relation)
            point = _INTERVAL_TO_POINT_RELATION[interval.relation]

        elif isinstance(relation, IntervalRelation):
            point = _INTERVAL_TO_POINT_RELATION[relation.relation]

        elif isinstance(relation, PointRelation):
            point = relation

        else:
            raise TypeError("Argument type is not supported.")

        return point


class TemporalRelation:

    def __init__(self, relation: Union[str, IntervalRelation, PointRelation]):
        self._point_relation = RelationHandler.handle(relation)

    def __repr__(self) -> str:
        return f"TemporalRelation({self.point})"

    def __str__(self) -> str:

        if self.interval:
            return self.interval

        else:
            return str(self.point)

    def __invert__(self):
        return TemporalRelation(~self.point)

    def __and__(self, other):
        return TemporalRelation(self.point & other.point)

    @property
    def interval(self) -> Union[str, PointRelation]:
        for itr, pnt in _INTERVAL_TO_POINT_RELATION.items():
            if pnt == self._point_relation:
                return itr

    @property
    def point(self):
        return self._point_relation
