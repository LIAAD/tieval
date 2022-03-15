"""Base objects.

Objects
-------
    - Document
    - Dataset

"""

from dataclasses import dataclass

from typing import Set, Optional, Union
from typing import Iterable, List

from tieval.entities import Event
from tieval.entities import Timex
from tieval.links import TLink
from tieval.closure import temporal_closure as _temporal_closure


class Document:
    """
    A document with temporal annotation.

    ...

    Attributes
    ----------
    name: str
        The name of the document
    text: str
        The raw test of the document
    events: Set[Event]
        The events annotated
    timexs: Set[Timex]
        The time expressions annotated
    tlinks: Set[TLink]
        The temporal links annotated

    Properties
    -------
    temporal_closure(sound=None)
        Prints the animals name and what sound it makes
    """

    def __init__(
            self,
            name: str,
            text: str,
            dct: Timex,
            entities: Set[Event],
            tlinks: Set[TLink],
            **kwargs
    ):
        self.name = name
        self.text = text
        self.dct = dct
        self.entities = entities
        self.tlinks = tlinks

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._closure = None

    def __repr__(self) -> str:
        return f'Document(name={self.name})'

    def __str__(self) -> str:
        return self.text.strip()

    def __getitem__(self, id: str) -> Optional[Union[Timex, Event, TLink]]:
        for entity in self.entities.union(self.tlinks):
            if entity.id == id:
                return entity

    @property
    def temporal_closure(self) -> Set[TLink]:
        """Compute temporal closure of the document.

        Temporal closure is the process of inferring new TLinks from the
        annotated TLinks.
        """

        if self._closure is None:
            self._closure = _temporal_closure(self.tlinks)

        return self._closure

    @property
    def timexs(self):
        return set(ent for ent in self.entities if isinstance(ent, Timex))

    @property
    def events(self):
        return set(ent for ent in self.entities if isinstance(ent, Event))

    @property
    def sentences(self):
        return self.text.split("\n")


@dataclass
class Dataset:
    """A compilation of documents that have temporal annotations."""

    name: str
    train: List[Document]
    test: List[Document] = None

    def __post_init__(self):

        if self.test is None:
            self.test = []

        self.documents = self.train + self.test

    def __repr__(self):
        return f"Dataset(name={self.name})"

    def __add__(self, other):

        name = f"{self.name}+{other._name}"
        train = self.train + other.train
        test = self.test + other.test

        return Dataset(name, train, test)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __getitem__(self, item):
        for doc in self.documents:
            if doc.name == item:
                return doc

    def __len__(self):
        return self.documents.__len__()
