from typing import Union

from text2timeline.temporal_relation import _POINT_TRANSITIONS
from text2timeline.temporal_relation import _INTERVAL_TO_POINT_COMPLETE
from text2timeline.temporal_relation import _INVERSE_INTERVAL_RELATION

from text2timeline.entities import Event, Timex
from text2timeline.temporal_relation import IntervalRelation, PointRelation


class TLink:
    """
    Object that represents a temporal link.

    ...

    Attributes
    -----------
    lid: str
        link id
    source: str
        source event/timex id that the link refers to
    target: str
        target event/timex id that the link refers to

    """

    def __init__(self,
                 id: str,
                 source: Union[Timex, Event],
                 target: Union[Timex, Event],
                 relation: Union[IntervalRelation, PointRelation]):

        self.id = id
        self.source = source
        self.target = target
        self.relation = relation

    def __str__(self):
        return f"{self.source} ---{self.relation}--> {self.target}"

    def __repr__(self):
        return f"TLink(id={self.id})"

    def __and__(self, other):
        """ Infer the relation between two TLinks.

        If a relation can be inferred it will return a Tlink between source of the first Tlink and target of the second
        Tlink.

        Example:
            tlink_1 = TLink({
                'id': 'l1',
                'source': 'e1',
                'target': 'e2',
                'relation': 'BEFORE'
            })

            tlink_2 = TLink({
                'id': 'l2',
                'source': 'e2',
                'target': 'e3',
                'relation': 'BEFORE'
            })

            tlink_1 & tlink_2

        :param other:
        :return:
        """

        # pair the relations of the first and second tlink
        paired_relations = zip(self.point_relation_complete, other.point_relation_complete)

        # get the relation between source of the first tlink and target of second tlink
        point_relation13 = [
            (relation12[0], _POINT_TRANSITIONS[relation12[1]][relation23[1]], relation23[2])
            for relation12, relation23 in paired_relations
        ]

        # search for a interval relation that matches the found point relation
        interval_relation = None
        for i_rel, p_rel in _INTERVAL_TO_POINT_COMPLETE.items():
            if p_rel == point_relation13:
                interval_relation = i_rel

        # if an interval relation was found it will return a TLink with it. otherwise it returns None
        if interval_relation:
            return TLink(
                id=f'il{self.source}{other.target}',
                source=self.source,
                target=other.target,
                relation=interval_relation,
            )

        else:
            return None

    def __invert__(self):
        """

        Invert TLink.
        Returns the symmetric tlink. For example, if A --Before--> B it will return a tlink with B --After--> A

        :return:
            A TLink symmetric to the current one.

        """

        return TLink(
            id=f'i{self.id}',
            source=self.target,
            target=self.source,
            relation=_INVERSE_INTERVAL_RELATION[self.relation]
        )

