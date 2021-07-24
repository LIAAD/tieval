from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

from text2timeline.constants import _POINT_TRANSITIONS
from text2timeline.constants import _INTERVAL_TO_POINT_COMPLETE
from text2timeline.constants import _INVERSE_INTERVAL_RELATION
from text2timeline.constants import _INTERVAL_RELATIONS
from text2timeline.constants import _SETTLE_RELATION
from text2timeline.constants import _INTERVAL_TO_POINT


PointRelation = List[Tuple[int, str, int]]
IntervalRelation = str


class Timex:
    """Object that represents a time expression."""

    def __init__(self, attributes: Dict):

        self.tid = attributes.get('tid')
        self.type = attributes.get('type')
        self.value = attributes.get('value')
        self.temporal_function = attributes.get('temporalFunction')
        self.function_in_document = attributes.get('functionInDocument')
        self.anchor_time_id = attributes.get('anchorTimeID')
        self.text = attributes.get('text')
        self.endpoints = attributes.get('endpoints')

    def __repr__(self):
        return f"Timex(tid={self.tid})"

    @property
    def id(self):
        return self.tid

    @property
    def is_dct(self):
        if self.function_in_document == 'CREATION_TIME':
            return True

        return False


class Event:
    """Object that represents an event."""

    def __init__(self, attributes: Dict):

        self.eid = attributes.get('eid')
        self.eiid = attributes.get('eiid')
        self.family = attributes.get('class')
        self.stem = attributes.get('stem')
        self.aspect = attributes.get('aspect')
        self.tense = attributes.get('tense')
        self.polarity = attributes.get('polarity')
        self.pos = attributes.get('pos')
        self.text = attributes.get('text')
        self.endpoints = attributes.get('endpoints')

    def __repr__(self):
        return f"Event(eid={self.eid})"

    @property
    def id(self):
        if self.eiid:
            return self.eiid

        return self.eid


# TODO: make TLink able to handel multiple interval relations
class TemporalRelation:
    """Descriptor for temporal relation.

    The relation can be passed as:
        - interval relation (ex: AFTER, BEFORE, ...)
        - point relation (ex: [(1, '<', 0)], )
        - complete point relation (ex: [(0, '<', 0), (1, '<', 0), (0, '<', 1), (1, '<', 1)])
    """

    def __get__(self, instance, owner):
        return self._relation

    def __set__(self, instance, relation):
        self._relation = self._validate(relation)
        if not self._relation:
            raise ValueError(f"{relation} is not valid")

    @staticmethod
    def _validate(relation):

        # if it is an interval relation
        if relation in _INTERVAL_RELATIONS:
            return _SETTLE_RELATION[relation]

        # if it is point relation
        elif relation in _INTERVAL_TO_POINT.values():
            interval_relation = [rel
                                 for rel, requirements in _INTERVAL_TO_POINT.items()
                                 if set(requirements).issubset(relation)]

            return [_SETTLE_RELATION[rel] for rel in interval_relation]

        # if it is a complete interval relation
        elif relation in _INTERVAL_TO_POINT_COMPLETE.values():
            [interval_relation] = [rel
                                   for rel, requirements in _INTERVAL_TO_POINT_COMPLETE.items()
                                   if set(requirements).issubset(relation)]

            return _SETTLE_RELATION[interval_relation]


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
    interval_relation: str
        interval relation between source and target. Ex: 'BEFORE', 'AFTER'
    point_relation: List
        List with the minimal point relation between the edges of source and target (minimal mining set os relations
        that defines the interval relation between source and target):
    """

    relation = TemporalRelation()

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

        If a relation can be infered it will return a Tlink between source of the first Tlink and target of the second
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

    @property
    def point_relation(self):
        return _INTERVAL_TO_POINT[self.relation]

    @property
    def point_relation_complete(self):
        return _INTERVAL_TO_POINT_COMPLETE[self.relation]


@dataclass
class Document:
    """An temporally annotated document."""

    name: str
    text: str
    events: List[Event]
    timexs: List[Timex]
    tlinks: List[TLink]

    def __repr__(self):
        return f'Document(name={self.name})'

    def __str__(self):
        return self.text.strip()

    @property
    def dct(self):
        """Extract document creation time"""
        for timex in self.timexs:
            if timex.is_dct:
                return timex

    def augment_tlinks(self, relations: List[str] = None) -> None:
        """ Augments the document tlinks by adding the symmetic relation of every tlink.
        For example if we have the tlink with A --BEFORE--> B the augmentation will add B --AFTER--> A to the document
        tlink list.

        :parameter:
            relation: a relation to limit the augmentation. If this argument is passed the method will only add the
            symmetric relation to tlink that have this relation in theis point_relation.

        :return: None
        """

        inv_tlinks = []
        for tlink in self.tlinks:

            if relations:
                cond_point_rel = [True for _, rel, _ in tlink.point_relation if rel in relations]
                cond_inter_rel = [tlink.relation in relations]
                cond = any(cond_point_rel + cond_inter_rel)

            else:
                cond = True

            if cond:
                inv_tlinks += [~tlink]

        self.tlinks += inv_tlinks

    def temporal_closure(self):
        pass


@dataclass
class Dataset:
    """A compilation of documents that have temporal annotations."""

    documents: List[Document]
