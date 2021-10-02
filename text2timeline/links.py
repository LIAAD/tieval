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
                 source: Union[str, Timex, Event],
                 target: Union[str, Timex, Event],
                 relation: Union[str, list, dict, TemporalRelation]):

        self.id = id
        self.source = source
        self.target = target

        self._relation = None
        self.relation = relation

    def __str__(self):
        return f"{self.source} ---{self.relation}--> {self.target}"

    def __repr__(self):
        return f"TLink(id={self.id})"

    @staticmethod
    def _resolve_inference(self, other):

        if self.source == other.source:
            source = self.target
            target = other.target
            relation = (~self.relation) & other.relation

        elif self.source == other.target:
            source = self.target
            target = other.source
            relation = ~self.relation & ~other.relation

        elif self.target == other.source:
            source = self.source
            target = other.target
            relation = self.relation & other.relation

        elif self.target == other.target:
            source = self.source
            target = other.source
            relation = self.relation & ~other.relation

        return source, target, relation

    def __and__(self, other):
        """ Infer the relation between two TLinks.

        If a relation can be inferred it will return a Tlink between source of the first Tlink and target of the second
        Tlink.

        """

        # verify that there is one entity in common between self and other instances
        entities = {self.source, self.target, other.source, other.target}
        if len(entities) != 3:
            return None

        source, target, relation = self._resolve_inference(self, other)

        if relation.interval == "VAGUE":
            return None

        return TLink(
            id=f'il{self.source}{other.target}',
            source=source,
            target=target,
            relation=relation
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

    def __eq__(self, other):

        other_inv = ~other

        if (self.source == other.source) and \
                (self.target == other.target) and \
                (self.relation == other.relation):
            return True

        elif (self.source == other_inv.source) and \
                (self.target == other_inv.target) and \
                (self.relation == other_inv.relation):
            return True

        return False

    def __hash__(self):
        return hash((self.source, self.target, self.relation))

    @property
    def relation(self) -> TemporalRelation:
        return self._relation

    @relation.setter
    def relation(self, rel: str) -> None:
        self._relation = TemporalRelation(rel)

