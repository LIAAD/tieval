from xml.etree import ElementTree

import nltk

from typing import List
from typing import Tuple
from typing import Dict

import collections


class TLink:
    def __init__(self, source: str, target: str, point_relation: List = None, interval_relation: List = None):
        self.source = source
        self.target = target

        if interval_relation:
            interval_relation = self._assert_relation[interval_relation]
            self.interval_relation = interval_relation
            self.point_relation = self._get_point_relation()
        elif point_relation:
            self.point_relation = point_relation
            self.interval_relation = self._get_interval_relation()
        else:
            raise Exception("point_relation and interval_relation are both None. Must provide one of them.")

    def _get_point_relation(self):
        if self.interval_relation:
            return self._interval_to_point[self.interval_relation]
        else:
            raise Exception("interval_relation not defined.")

    def _get_interval_relation(self):
        result = [relation
                  for relation, requirements in self._interval_to_point.items()
                  if set(requirements).issubset(self.point_relation)]
        if not result:
            return None
        return result

    # constants representing the start point and end point of an interval
    _start = 0
    _end = 1
    
    # Mapping from interval relation names to point relations.
    # For example, BEFORE means that the first interval's end is before the second interval's start
    _interval_to_point = {
        "BEFORE": [(_end, "<", _start)],
        "AFTER": [(_start, '>', _end)],
        "IBEFORE": [(_end, "=", _start)],
        "IAFTER": [(_start, "=", _end)],
        # "CONTAINS": [(_start, "<", _start), (_end, "<", _end)],
        "INCLUDES": [(_start, "<", _start), (_end, '>', _end)],
        "IS_INCLUDED": [(_start, '>', _start), (_end, "<", _end)],
        "BEGINS-ON": [(_start, "=", _start)],
        "ENDS-ON": [(_end, "=", _end)],
        "BEGINS": [(_start, "=", _start), (_end, "<", _end)],
        "BEGUN_BY": [(_start, "=", _start), (_end, '>', _end)],
        "ENDS": [(_start, '>', _start), (_end, "=", _end)],
        "ENDED_BY": [(_start, "<", _start), (_end, "=", _end)],
        "SIMULTANEOUS": [(_start, "=", _start), (_end, "=", _end)],
        # "IDENTITY": [(_start, "=", _start), (_end, "=", _end)],
        # "DURING": [(_start, "=", _start), (_end, "=", _end)],
        # "DURING_INV": [(_start, "=", _start), (_end, "=", _end)],
        "OVERLAP": [(_start, "<", _end), (_end, '>', _start)],
    }
    
    _interval_to_point_complete = {
        "BEFORE": [(_start, "<", _start),
                   (_start, "<", _end),
                   (_end, "<", _start),
                   (_end, "<", _end)],
        "AFTER": [(_start, ">", _start),
                  (_start, ">", _end),
                  (_end, ">", _start),
                  (_end, ">", _end)],
        "IBEFORE": [(_start, "<", _start),
                    (_start, "=", _end),
                    (_end, "<", _start),
                    (_end, "<", _end)],
        "IAFTER": [(_start, ">", _start),
                   (_start, "=", _end),
                   (_end, ">", _start),
                   (_end, ">", _end)],
        "CONTAINS": [(_start, "<", _start),
                     (_start, "<", _end),
                     (_end, ">", _start),
                     (_end, ">", _end)],
        "INCLUDES": [(_start, "<", _start),
                     (_start, "<", _end),
                     (_end, ">", _start),
                     (_end, ">", _end)],
        "IS_INCLUDED": [(_start, ">", _start),
                        (_start, "<", _end),
                        (_end, ">", _start),
                        (_end, "<", _end)],
        "BEGINS-ON": [(_start, "=", _start),
                      (_start, "<", _end),
                      (_end, ">", _start),
                      (_end, None, _end)],
        "ENDS-ON": [(_start, None, _start),
                    (_start, "<", _end),
                    (_end, ">", _start),
                    (_end, "=", _end)],
        "BEGINS": [(_start, "=", _start),
                   (_start, "<", _end),
                   (_end, ">", _start),
                   (_end, "<", _end)],
        "BEGUN_BY": [(_start, "=", _start),
                     (_start, "<", _end),
                     (_end, ">", _start),
                     (_end, ">", _end)],
        "ENDS": [(_start, ">", _start),
                 (_start, "<", _end),
                 (_end, ">", _start),
                 (_end, "=", _end)],
        "ENDED_BY": [(_start, "<", _start),
                     (_start, "<", _end),
                     (_end, ">", _start),
                     (_end, "=", _end)],

        "SIMULTANEOUS": [(_start, "=", _start),
                         (_start, "<", _end),
                         (_end, ">", _start),
                         (_end, "=", _end)],
        "IDENTITY": [(_start, "=", _start),
                     (_start, "<", _end),
                     (_end, ">", _start),
                     (_end, "=", _end)],
        "DURING": [(_start, "=", _start),
                   (_start, "<", _end),
                   (_end, ">", _start),
                   (_end, "=", _end)],
        "DURING_INV": [(_start, "=", _start),
                       (_start, "<", _end),
                       (_end, ">", _start),
                       (_end, "=", _end)],
        "OVERLAP": [(_start, "<", _start),
                    (_start, "<", _end),
                    (_end, ">", _start),
                    (_end, "<", _end)]
    }

    # Map relations to the standard names.
    _assert_relation = {
        'OVERLAP': 'OVERLAP',
        'BEGINS': 'BEGINS',
        'BEFORE': 'BEFORE',
        'CONTAINS': 'INCLUDES',
        'IDENTITY': 'SIMULTANEOUS',
        'AFTER': 'AFTER',
        'BEGINS-ON': 'BEGINS-ON',
        'SIMULTANEOUS': 'SIMULTANEOUS',
        'INCLUDES': 'INCLUDES',
        'DURING': 'SIMULTANEOUS',
        'ENDS-ON': 'ENDS-ON',
        'BEGUN_BY': 'BEGUN_BY',
        'ENDED_BY': 'ENDED_BY',
        'DURING_INV': 'SIMULTANEOUS',
        'ENDS': 'ENDS',
        'IS_INCLUDED': 'IS_INCLUDED',
        'IBEFORE': 'IBEFORE',
        'IAFTER': 'IAFTER'
    }


class Timex:
    def __init__(self, attributes: Dict):
        self.type = attributes['type']
        self.type = attributes['type']
        self.text = attributes['text']
        self.value = attributes['value']
        self.endpoints = attributes['endpoints']
        self.function_in_document = attributes['functionInDocument']
        self.temporal_function = attributes['temporalFunction']


class Event:
    def __init__(self, attributes: dict):
        self.class_ = attributes['class']
        self.text = attributes['text']
        self.endpoints = attributes['endpoints']
        self.aspect = attributes['aspect']
        self.eiid = attributes['eiid']
        self.polarity = attributes['polarity']
        self.pos = attributes['pos']
        self.tense = attributes['tense']


class Document:
    def __init__(self, path: str):
        self.tokenizer = nltk.tokenize.WordPunctTokenizer()
        self.xml_root = ElementTree.parse(path).getroot()
        self.name = self.xml_root.findtext('.//DOCID')

        self.text = self._get_text()
        self.sentences = self._get_sentences()
        self.tokens = self._get_tokens()

        self._expression_idxs = self._expression_indexes()
        self.timexs = self._get_timexs()
        self.events = self._get_events()

        self.tlinks = self._get_tlinks()

    def _get_text(self) -> str:
        """
        Returns the raw text of the document
        :return:
        """
        text_root = self.xml_root.find('.//TEXT')
        text = ''.join(list(text_root.itertext()))
        return text

    def _get_sentences(self) -> List[Tuple]:
        sentences = nltk.sent_tokenize(self.text)
        sentence_idxs = list()
        for sent in sentences:
            start = self.text.find(sent)
            end = start + len(sent)
            sentence_idxs.append((start, end, sent))
        return sentence_idxs

    def _get_tokens(self) -> List[Tuple]:
        """
        Returns the tokens with their respective start and end positions.
        :return:
        """
        spans = self.tokenizer.span_tokenize(self.text)
        tokens = [(start, end, self.text[start:end]) for start, end in spans]
        return tokens

    def _expression_indexes(self) -> dict:
        """
        Finds start and end indexes of each expression (EVENT or TIMEX).

        :return:
        """
        text_root = self.xml_root.find('.//TEXT')

        # Find indexes of the expressions.
        text_blocks = list()
        start = 0
        for txt in text_root.itertext():
            end = start + len(txt)
            text_blocks.append((start, end, txt))
            start = end

        # Get the tags of each expression.
        text_tags = list()
        for txt in text_root.iter():
            if txt.items():
                text_tags.append((txt.text, txt.items()[0][1]))

        # Join the indexes with the tags.
        expression_idxs = {'t0': (-1, -1, None)}  # Initialize with the position of DCT.
        for txt, tag_id in text_tags:
            for start, end, txt_sch in text_blocks:
                if txt == txt_sch:
                    expression_idxs[tag_id] = (start, end)
        return expression_idxs

    def _get_make_instance(self) -> dict:
        make_insts = dict()
        for make_inst in self.xml_root.findall('.//MAKEINSTANCE'):
            make_insts[make_inst.attrib.pop('eventID')] = make_inst.attrib
        return make_insts

    def _get_events(self) -> dict:
        # Most of event attributes are in <MAKEINSTACE> tag.
        make_insts = self._get_make_instance()

        events = dict()
        for event in self.xml_root.findall('.//EVENT'):
            event_id = event.attrib.pop('eid')
            event.attrib['text'] = event.text
            event.attrib['endpoints'] = self._expression_idxs[event_id]
            event.attrib.update(make_insts[event_id])
            events[event_id] = Event(event.attrib)
        return events

    def _get_timexs(self) -> dict:
        # Add text to timex attributes.
        timexs = dict()
        for timex in self.xml_root.findall('.//TIMEX3'):
            time_id = timex.attrib['tid']
            timex.attrib['text'] = timex.text
            timex.attrib['endpoints'] = self._expression_idxs[time_id]
            timexs[time_id] = Timex(timex.attrib)
        return timexs

    def _get_tlinks(self) -> dict:
        tlinks = dict()
        for tlink in self.xml_root.findall('.//TLINK'):
            attrib = tlink.attrib
            keys = tlink.keys()
            tlinks[attrib['lid']] = TLink(
                source=attrib[keys[2]],
                target=attrib[keys[3]],
                interval_relation=attrib['relType']
            )
        return tlinks

    def temporal_closure(self) -> Dict:
        # convert interval relations to point relations
        new_relations = {((tlink.source, scr_ep), r, (tlink.target, tgt_ep))
                         for lid, tlink in self.tlinks.items()
                         for scr_ep, r, tgt_ep in tlink.point_relation}

        # repeatedly apply point transitivity rules until no new relations can be inferred
        point_relations = set()
        point_relations_index = collections.defaultdict(set)
        while new_relations:

            # update the result and the index with any new relations found on the last iteration
            point_relations.update(new_relations)
            for point_relation in new_relations:
                point_relations_index[point_relation[0]].add(point_relation)

            # infer any new transitive relations, e.g., if A < B and B < C then A < C
            new_relations = set()
            for point1, relation12, point2 in point_relations:
                for _, relation23, point3 in point_relations_index[point2]:
                    relation13 = self._point_transitions[relation12][relation23]
                    if not relation13:
                        continue
                    new_relation = (point1, relation13, point3)
                    if new_relation not in point_relations:
                        new_relations.add(new_relation)

        # Combine point relations.
        combined_point_relations = collections.defaultdict(list)
        for ((scr, scr_ep), rel, (tgt, tgt_ep)) in point_relations:
            combined_point_relations[(scr, tgt)] += [(scr_ep, rel, tgt_ep)]

        # Creat and store the valid temporal links.
        tlinks = []
        for (scr, tgt), point_relation in combined_point_relations.items():
            tlink = TLink(scr, tgt, point_relation=point_relation)
            if tlink.interval_relation:
                tlinks.append(tlink)

        # Generate indexes for tlinks.
        return {f'tc{idx}': tlink for idx, tlink in enumerate(tlinks)}

    # transitivity table for point relations
    _point_transitions = {
        '<': {'<': '<', '=': '<', '>': None},
        '=': {'<': '<', '=': '=', '>': '>'},
        '>': {'>': '>', '=': '>', '<': None}
    }
