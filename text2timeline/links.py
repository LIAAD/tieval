from typing import Union

from text2timeline.temporal_relation import TemporalRelation
from text2timeline.entities import Event, Timex


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
                 relation: TemporalRelation):

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

        return TLink(
            id=f'il{self.source}{other.target}',
            source=self.source,
            target=other.target,
            relation=self.relation & other.relation,
        )

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
            relation=~self.relation
        )
