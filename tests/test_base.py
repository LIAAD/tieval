import unittest

import pytest

from text2timeline.base import Event
from text2timeline.base import Timex
from text2timeline.base import TLink


class TestTimex(unittest.TestCase):

    timex = Timex({
        'tid': 't0',
        'type': 'DATE',
        'value': '1998-03-03',
        'temporalFunction': 'false',
        'functionInDocument': 'CREATION_TIME'
    })

    def test_id(self):

        self.assertEqual(self.timex.id, 't0', 'Should be "t0".')

    def test_is_dct(self):

        self.assertTrue(self.timex.is_dct, 'Shoulb be True.')


class TestEvent(unittest.TestCase):

    event = Event({
        'eid': 'e1',
        'class': 'OCCURRENCE',
        'text': 'expansion',
        'endpoints': (82, 91),
        'eventID': 'e1',
        'eiid': 'ei50001',
        'tense': 'NONE',
        'aspect': 'NONE',
        'polarity': 'POS',
        'pos': 'NOUN'
    })

    def test_id(self):

        self.assertEqual(self.event.id, 'ei50001')


class TestTLink(unittest.TestCase):

    source = Event({
        'eid': 'e1',
        'class': 'OCCURRENCE',
        'text': 'expansion',
        'endpoints': (82, 91),
        'eventID': 'e1',
        'eiid': 'ei50001',
        'tense': 'NONE',
        'aspect': 'NONE',
        'polarity': 'POS',
        'pos': 'NOUN'
    })

    target = Timex({
        'tid': 't0',
        'type': 'DATE',
        'value': '1998-03-03',
        'temporalFunction': 'false',
        'functionInDocument': 'CREATION_TIME'
    })

    tlink = TLink(
        id='l70',
        source=source,
        target=target,
        relation='BEFORE'
    )

    def test_id(self):
        self.assertEqual(self.tlink.id, 'l70')

    def test_relation(self):

        self.assertEqual(self.tlink.relation, 'BEFORE')
        self.assertEqual(self.tlink.point_relation, [(1, '<', 0)])
        self.assertEqual(self.tlink.point_relation_complete, [(0, '<', 0), (0, '<', 1), (1, '<', 0), (1, '<', 1)])

    def test_invert(self):

        tlink_invert = ~self.tlink

        self.assertEqual(tlink_invert.relation, 'AFTER')
        self.assertEqual(tlink_invert.point_relation, [(0, '>', 1)])
        self.assertEqual(tlink_invert.point_relation_complete, [(0, '>', 0), (0, '>', 1), (1, '>', 0), (1, '>', 1)])


if __name__ == '__main__':

    unittest.main()
