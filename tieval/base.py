from dataclasses import dataclass
from typing import Set, Optional, Union, List, Tuple

import nltk

from tieval import utils
from tieval.closure import temporal_closure as _temporal_closure
from tieval.entities import Entity, Event, Timex
from tieval.links import TLink


class Token:

    def __init__(
            self,
            content: str,
            offsets: List[int]
    ) -> None:
        self.content = content
        self.offsets = offsets

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
            offsets: Tuple[int, int],
    ):
        self.content = content
        self.offsets = offsets
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
            offsets = utils.get_offsets(self.content, tkns, self.offsets[0])

            self._tokens = [Token(tkn, offsets) for tkn, offsets in zip(tkns, offsets)]

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
            offsets = utils.get_offsets(self.content, sents)

            self._sents = [Sentence(sent, offsets) for sent, offsets in zip(sents, offsets)]

        return self._sents


class Document:
    """A document with temporal annotation.

    :param str name:  The name of the document
    :param str text: The raw test of the document
    :param Set[Event, Timex] entities: The events annotated
    :param Set[TLink] tlinks: The temporal links annotated
    """

    def __init__(
            self,
            name: str,
            text: str,
            dct: Timex,
            entities: Set[Entity],
            tlinks: Set[TLink],
            language: str = "english",
            **kwargs
    ) -> None:

        self.name = name
        self.text = text
        self._text = Text(text, language)
        self.dct = dct
        self.entities = [ent for ent in entities if not ent.is_dct]
        self.tlinks = tlinks

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._closure = None

    def __repr__(self) -> str:
        return f'Document(name={self.name})'

    def __str__(self) -> str:
        return self.text

    def __getitem__(self, id: str) -> Optional[Union[Entity, TLink]]:
        for entity in self.entities.union(self.tlinks):
            if entity.id == id:
                return entity

    @property
    def temporal_closure(self) -> Set[TLink]:
        """Compute temporal closure of the document. Temporal closure is the process of inferring new TLinks from the
        annotated TLinks.

        It will call the function :py:func:`tieval.closure.temporal_closure` on the set of temporal links of the
        current document.
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
        return self._text.sentences

    @property
    def tokens(self) -> List[str]:
        return [
            tkn
            for sent in self._text.sentences
            for tkn in sent.tokens
        ]


@dataclass
class Dataset:
    """A compilation of documents that have temporal annotations.

    :param str name:  The name of the dataset.
    :param list[Document] train:  A list containing the documents of the training set.
    :param list[Document] test:  A list containing the documents of the test set.
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
