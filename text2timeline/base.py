from dataclasses import dataclass

from typing import List

from text2timeline.entities import Event, Timex
from text2timeline.links import TLink


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

    def __getitem__(self, id: str):
        entities = self.events + self.timexs + self.tlinks
        for entity in entities:
            if entity.id == id:
                return entity

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

    name: str
    documents: List[Document]

    def __repr__(self):
        return f"Dataset(name={self.name})"

    def __add__(self, other):
        name = f"{self.name}+{other.name}"
        docs = self.documents + other.documents
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
