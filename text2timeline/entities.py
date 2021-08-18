
from typing import Dict


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
