from typing import Union

from tieval.entities import Entity
from tieval.temporal_relation import TemporalRelation


class TLink:
    """ Object that represents a temporal link.

    :param str id:  Link id
    :param Union[str, Timex, Event] source: Source event/timex id that the link refers to
    :param Union[str, Timex, Event] target: Target event/timex id that the link refers to
    :param Union[str, list, dict, TemporalRelation] relation: The temporal relation between source and target
    """

    def __init__(
            self,
            source: Union[str, Entity],
            target: Union[str, Entity],
            relation: Union[str, list, dict, TemporalRelation],
            id: str = None
    ):
        self.source = source
        self.target = target
        self.relation = TemporalRelation(relation)

        if id is None:

            # generate an id from object hash
            foo = hex(hash(self))
            foo = foo[foo.find("x") + 1:]
            self.id = foo

        else:
            self.id = id

    def __str__(self):
        return f"{self.source} ---{self.relation}--> {self.target}"

    def __repr__(self):
        return f"TLink({self.source} ---{self.relation}--> {self.target})"

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
        """ Invert TLink.
        Returns the symmetric tlink. For example, if A --Before--> B it will return a tlink with B --After--> A

        :return: A TLink symmetric to the current one.
        """

        return TLink(
            id=f'i{self.id}',
            source=self.target,
            target=self.source,
            relation=~self.relation
        )

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        """Define a unique encoding depending on source, target and relation.
         This encoding is necessary for hashing and equality.
         """
        entities = [self.source, self.target]
        entities.sort()
        src, tgt = entities

        if src == self.source:
            relation = self.relation.point
        else:
            relation = ~self.relation.point

        return hash((src, tgt, relation))

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

        else:  # self.target == other.target
            source = self.source
            target = other.source
            relation = self.relation & ~other.relation

        return source, target, relation

    @property
    def entities(self):
        return self.source, self.target

    @property
    def source_id(self):
        if isinstance(self.source, Entity):
            return self.source.id
        return self.source

    @property
    def target_id(self):
        if isinstance(self.target, Entity):
            return self.target.id
        return self.target
