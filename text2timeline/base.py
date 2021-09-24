from dataclasses import dataclass

from typing import Set, Iterable, Optional, Union

from text2timeline.entities import Event, Timex
from text2timeline.links import TLink
from text2timeline.closure import temporal_closure as _temporal_closure


@dataclass
class Document:
    """An temporally annotated document."""

    name: str
    text: str
    events: Set[Event]
    timexs: Set[Timex]
    tlinks: Set[TLink]

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
        """Extract document creation time"""
        for timex in self.timexs:
            if timex.is_dct:
                return timex

    @property
    def entities(self) -> Set:
        return self.events.union(self.timexs)

    @property
    def temporal_closure(self) -> Set[TLink]:

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
