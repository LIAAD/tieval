from typing import Union, Tuple


class Timex:
    """Object that represents a time expression."""

    def __init__(
            self,
            id: str = None,
            text: str = None,
            value: str = None,
            endpoints: Tuple[int, int] = None,
            type_: str = None,
            function_in_document: str = None,
            anchor_time_id: str = None,
            sent_idx: int = None,
            **kwargs
    ):

        self.id = id
        self.type = type_
        self.value = value
        self.function_in_document = function_in_document
        self.anchor_time_id = anchor_time_id
        self.text = text
        self.endpoints = endpoints
        self.sent_idx = sent_idx

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"Timex(\"{self.text}\")"

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
            endpoints: Tuple[int, int] = None,
            family: str = None,
            stem: str = None,
            lemma: str = None,
            aspect: str = None,
            tense: str = None,
            polarity: str = None,
            pos: str = None,
            class_: str = None,
            start_time: str = None,
            end_time: str = None,
            sent_idx: int = None,
            **kwargs
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
        self.start_time = start_time
        self.end_time = end_time
        self.sent_idx = sent_idx

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"Event(\"{self.text}\")"

    def __lt__(self, other):
        return self.id < other.id


Entity = Union[Timex, Event]
