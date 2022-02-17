""" Objects that represent entities on TML.

Objects implemented:
    - Timex
    - Event

"""

from typing import Dict
from typing import Union
from typing import Iterable

from dataclasses import dataclass


@dataclass
class Timex:
    """Object that represents a time expression."""

    def __init__(
            self,
            id: str = None,
            text: str = None,
            value: str = None,
            endpoints: Iterable[int] = None,
            type_: str = None,
            function_in_document: str = None,
            anchor_time_id: str = None,
    ):

        self.id = id
        self.type = type_
        self.value = value
        self.function_in_document = function_in_document
        self.anchor_time_id = anchor_time_id
        self.text = text
        self.endpoints = endpoints

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"Timex(tid={self.id})"

    def __lt__(self, other):
        return self.id < other.id

    @property
    def is_dct(self):
        if self.function_in_document == 'CREATION_TIME':
            return True

        return False


class Event:
    """Object that represents an event."""

    def __init__(
            self,
            id: str = None,
            text: str = None,
            endpoints: Iterable[int] = None,
            family: str = None,
            stem: str = None,
            lemma: str = None,
            aspect: str = None,
            tense: str = None,
            polarity: str = None,
            pos: str = None,
            class_: str = None
    ):

        self.id = id
        self.family = family
        self.stem = stem
        self.lemma = lemma
        self.aspect = aspect
        self.tense = tense
        self.polarity = polarity
        self.pos = pos
        self.text = text
        self.endpoints = endpoints
        self.class_ = class_

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"Event(eid={self.id})"

    def __lt__(self, other):
        return self.id < other.id


Entity = Union[Timex, Event]
