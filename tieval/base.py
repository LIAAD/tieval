"""Base objects.

Objects
-------
    - Document
    - Dataset

"""

from dataclasses import dataclass
from typing import Set, Optional, Union, List

import nltk

from tieval.entities import Event
from tieval.entities import Timex
from tieval.links import TLink
from tieval.closure import temporal_closure as _temporal_closure
from tieval import utils


class Token:

    def __init__(
            self,
            content: str,
            span: List[int]
    ) -> None:

        self.content = content
        self.span = span

    def __str__(self):
        return self.content

    def __repr__(self):
        return self.content

    def __call__(self):
        return self.content


class Sentence:

    def __init__(
            self,
            content: str,
            span: List[int],
    ):
        self.content = content
        self.span = span
        self._tokens = None

    def __str__(self):
        return self.content

    def __repr__(self):
        return self.content

    def __call__(self):
        return self.content

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return self.content.__len__()

    @property
    def tokens(self) -> List[List[str]]:

        if self._tokens is None:

            tkns = nltk.tokenize.word_tokenize(self.content)
            spans = utils.get_spans(self.content, tkns, self.span[0])

            self._tokens = [Token(tkn, span) for tkn, span in zip(tkns, spans)]

        return self._tokens


class Text:

    def __init__(
            self,
            content: str,
            language: str = "english"
    ) -> None:

        self.content = content
        self.language = language

        self.tokenizer = nltk.tokenize.sent_tokenize

        self._sents = None

    def __str__(self):
        return self.content

    @property
    def sentences(self) -> List[str]:

        if self._sents is None:

            sents = self.tokenizer(self.content, language=self.language)
            spans = utils.get_spans(self.content, sents)

            self._sents = [Sentence(sent, span) for sent, span in zip(sents, spans)]

        return self._sents


class Document:
    """A document with temporal annotation.

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
    ----------
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
            language: str = "english",
            **kwargs
    ) -> None:

        self.name = name
        self.text = Text(text, language)
        self.dct = dct
        self.entities = entities
        self.tlinks = tlinks

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._closure = None

    def __repr__(self) -> str:
        return f'Document(name={self.name})'

    def __str__(self) -> str:
        return self.text

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
    def timexs(self) -> Set[Timex]:
        return set(ent for ent in self.entities if isinstance(ent, Timex))

    @property
    def events(self) -> Set[Event]:
        return set(ent for ent in self.entities if isinstance(ent, Event))

    @property
    def sentences(self) -> List[str]:
        return self.text.sentences

    @property
    def tokens(self) -> List[str]:
        return [
            tkn
            for sent in self.text.sentences
            for tkn in sent.tokens
        ]


@dataclass
class Dataset:
    """A compilation of documents that have temporal annotations.

    Attributes
    ----------
    name : str
        The name of the dataset.
    train : list[Document]
        A list containing the documents of the training set.
    test : list[Document]
        A list containing the documents of the test set.
    """

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

        name = f"{self.name}+{other.name}"
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
