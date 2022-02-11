"""Base objects.

Objects
-------
    - Document
    - Dataset

"""

from dataclasses import dataclass

from typing import Set, Optional, Union

from text2timeline.entities import Event
from text2timeline.entities import Timex
from text2timeline.links import TLink
from text2timeline.closure import temporal_closure as _temporal_closure


@dataclass
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

    name: str
    text: str
    events: Set[Event]
    timexs: Set[Timex]
    tlinks: Set[TLink]
    set: str = None

    def __post_init__(self):
        self._eiid2eid = {event.eiid: event.id for event in self.events}
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
    def dct(self):
        """Document creation time.

        Returns the Timex that is set as document creation time for this
        document.
        """

        for timex in self.timexs:
            if timex.is_dct:
                return timex

    @property
    def entities(self) -> Set:
        """ A set with the Events and Timexs."""

        return self.events.union(self.timexs)

    @property
    def temporal_closure(self) -> Set[TLink]:
        """Compute temporal closure of the document.

        Temporal closure is the process of inferring new TLinks from the
        annotated TLinks.
        """

        if self._closure is None:
            self._closure = _temporal_closure(self.tlinks)

        return self._closure


@dataclass
class Dataset:
    """A compilation of documents that have temporal annotations."""

    name: str
    documents: Set[Document]

    def __repr__(self):
        return f"Dataset(name={self.name})"

    def __add__(self, other):
        name = f"{self.name}+{other.name}"
        docs = self.documents.union(other.documents)
        return Dataset(name, docs)

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

    @property
    def train(self):
        return [doc for doc in self.documents if doc.set == "train"]

    @property
    def test(self):
        return [doc for doc in self.documents if doc.set == "test"]

    def evalaute(self, prediction, verbose=0):

        result = {
            "recall": None,
            "precision": None,
            "f1": None,
            "f_aware": None
        }

        # compute
        if verbose:
            print()

        return result
