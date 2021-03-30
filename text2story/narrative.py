import collections
from typing import Dict
from typing import List
from typing import Tuple
from xml.etree import ElementTree
import copy

import nltk

from pprint import pprint

# constants representing the start point and end point of an interval
_START = 0
_END = 1

# Mapping from interval relation names to point relations.
# For example, BEFORE means that the first interval's end is before the second interval's start
_INTERVAL_TO_POINT = {
    "BEFORE": [(_END, "<", _START)],
    "AFTER": [(_START, '>', _END)],
    "IBEFORE": [(_END, "=", _START)],
    "IAFTER": [(_START, "=", _END)],
    # "CONTAINS": [(_START, "<", _START), (_END, "<", _END)],
    "INCLUDES": [(_START, "<", _START), (_END, '>', _END)],
    "IS_INCLUDED": [(_START, '>', _START), (_END, "<", _END)],
    "BEGINS-ON": [(_START, "=", _START)],
    "ENDS-ON": [(_END, "=", _END)],
    "BEGINS": [(_START, "=", _START), (_END, "<", _END)],
    "BEGUN_BY": [(_START, "=", _START), (_END, '>', _END)],
    "ENDS": [(_START, '>', _START), (_END, "=", _END)],
    "ENDED_BY": [(_START, "<", _START), (_END, "=", _END)],
    "SIMULTANEOUS": [(_START, "=", _START), (_END, "=", _END)],
    # "IDENTITY": [(_START, "=", _START), (_END, "=", _END)],
    # "DURING": [(_START, "=", _START), (_END, "=", _END)],
    # "DURING_INV": [(_START, "=", _START), (_END, "=", _END)],
    "OVERLAP": [(_START, "<", _END), (_END, '>', _START)],
}

_INTERVAL_TO_POINT_COMPLETE = {
    "BEFORE": [(_START, "<", _START),
               (_START, "<", _END),
               (_END, "<", _START),
               (_END, "<", _END)],
    "AFTER": [(_START, ">", _START),
              (_START, ">", _END),
              (_END, ">", _START),
              (_END, ">", _END)],
    "IBEFORE": [(_START, "<", _START),
                (_START, "=", _END),
                (_END, "<", _START),
                (_END, "<", _END)],
    "IAFTER": [(_START, ">", _START),
               (_START, "=", _END),
               (_END, ">", _START),
               (_END, ">", _END)],
    "CONTAINS": [(_START, "<", _START),
                 (_START, "<", _END),
                 (_END, ">", _START),
                 (_END, ">", _END)],
    "INCLUDES": [(_START, "<", _START),
                 (_START, "<", _END),
                 (_END, ">", _START),
                 (_END, ">", _END)],
    "IS_INCLUDED": [(_START, ">", _START),
                    (_START, "<", _END),
                    (_END, ">", _START),
                    (_END, "<", _END)],
    "BEGINS-ON": [(_START, "=", _START),
                  (_START, "<", _END),
                  (_END, ">", _START),
                  (_END, None, _END)],
    "ENDS-ON": [(_START, None, _START),
                (_START, "<", _END),
                (_END, ">", _START),
                (_END, "=", _END)],
    "BEGINS": [(_START, "=", _START),
               (_START, "<", _END),
               (_END, ">", _START),
               (_END, "<", _END)],
    "BEGUN_BY": [(_START, "=", _START),
                 (_START, "<", _END),
                 (_END, ">", _START),
                 (_END, ">", _END)],
    "ENDS": [(_START, ">", _START),
             (_START, "<", _END),
             (_END, ">", _START),
             (_END, "=", _END)],
    "ENDED_BY": [(_START, "<", _START),
                 (_START, "<", _END),
                 (_END, ">", _START),
                 (_END, "=", _END)],

    "SIMULTANEOUS": [(_START, "=", _START),
                     (_START, "<", _END),
                     (_END, ">", _START),
                     (_END, "=", _END)],
    "IDENTITY": [(_START, "=", _START),
                 (_START, "<", _END),
                 (_END, ">", _START),
                 (_END, "=", _END)],
    "DURING": [(_START, "=", _START),
               (_START, "<", _END),
               (_END, ">", _START),
               (_END, "=", _END)],
    "DURING_INV": [(_START, "=", _START),
                   (_START, "<", _END),
                   (_END, ">", _START),
                   (_END, "=", _END)],
    "OVERLAP": [(_START, "<", _START),
                (_START, "<", _END),
                (_END, ">", _START),
                (_END, "<", _END)]
}

# transitivity table for point relations
_POINT_TRANSITIONS = {
    '<': {'<': '<', '=': '<', '>': None},
    '=': {'<': '<', '=': '=', '>': '>'},
    '>': {'>': '>', '=': '>', '<': None}
}

_INVERSE_POINT_RELATION = {
    '<': '>',
    '>': '<',
    '=': '='
}

_INVERSE_INTERVAL_RELATION = {
    'AFTER': 'BEFORE',
    'BEFORE': 'AFTER',
    'BEGINS': 'BEGUN_BY',
    'BEGINS-ON': 'BEGINS-ON',
    'BEGUN_BY': 'BEGINS',
    'ENDED_BY': 'ENDS',
    'ENDS': 'ENDED_BY',
    'ENDS-ON': 'ENDS-ON',
    'IAFTER': 'IBEFORE',
    'IBEFORE': 'IAFTER',
    'INCLUDES': 'IS_INCLUDED',
    'IS_INCLUDED': 'INCLUDES',
    'SIMULTANEOUS': 'SIMULTANEOUS'
}


# Map relations to the standard names.
_ASSERT_RELATION = {
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


class TLink:
    """
    Class that represents a temporal link.

    ...

    Attributes
    -----------
    lid: str
        link id
    source: str
        source event/timex id that the link refers to
    target: str
        target event/timex id that the link refers to
    interval_relation: str
        interval relation between source and target. Ex: 'BEFORE', 'AFTER'
    point_relation: List
        List with the minimal point relation between the edges of source and target (minimal mining set os relations
        that defines the interval relation between source and target):
    """
    def __init__(self, attributes: dict):
        self.lid = attributes['lid']
        self.source = attributes['source']
        self.target = attributes['target']
        self.task = self._infer_task()

        if 'interval_relation' in attributes:
            interval_relation = attributes['interval_relation']
            interval_relation = _ASSERT_RELATION[interval_relation]
            self.interval_relation = interval_relation
            self.point_relation = self._get_point_relation()
        elif 'point_relation' in attributes:
            self.point_relation = attributes['point_relation']
            self.interval_relation = self._get_interval_relation()
        else:
            raise Exception("point_relation and interval_relation are both None. Must provide one of them.")

    def __repr__(self):
        return f"{self.source} ---{self.interval_relation}--> {self.target}"

    def _infer_task(self):
        """ Infer the task based on source and target id.
        The task ontology is as follows:
            task A: event <-> event relations
            task B: timex <-> timex relations
            task C: event <-> timex relations
            task D: dct <-> event/timex relations

        :return:
        """
        scr, tgt = self.source, self.target
        if (scr == 't0') or (tgt == 't0'):
            return 'D'
        elif (scr[0] == 'e') and (tgt[0] == 'e'):
            return 'A'
        elif (scr[0] == 't') and (tgt[0] == 't'):
            return 'B'
        else:
            return 'C'

    def _get_point_relation(self):
        if self.interval_relation:
            return _INTERVAL_TO_POINT[self.interval_relation]
        else:
            raise Exception("interval_relation not defined.")

    def _get_interval_relation(self):
        result = [relation
                  for relation, requirements in _INTERVAL_TO_POINT.items()
                  if set(requirements).issubset(self.point_relation)]
        if not result:
            return None
        return result

    def complete_point_relation(self):
        if self.interval_relation:
            return _INTERVAL_TO_POINT_COMPLETE[self.interval_relation]
        else:
            raise Exception("interval_relation not defined.")

    def invert(self):
        """ Inverts the Tlink. For example if the Tlink is A ---Before---> B this method transforms it into
         B ---AFTER--> A.

        :return:
        """
        self.lid = 'a' + self.lid
        self.source, self.target = self.target, self.source
        self.interval_relation = _INVERSE_INTERVAL_RELATION[self.interval_relation]
        self.point_relation = self._get_point_relation()


class Timex:
    def __init__(self, attributes: Dict):
        for key, value in attributes.items():
            setattr(self, key, value)


class Event:
    def __init__(self, attributes: dict):
        for key, value in attributes.items():
            setattr(self, key, value)


class Document:
    def __init__(self, path: str):
        self.path = path

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
            if txt.attrib and txt.tag == 'EVENT':
                text_tags.append((txt.text, txt.attrib['eid']))
            elif txt.attrib and txt.tag == 'TIMEX3':
                text_tags.append((txt.text, txt.attrib['tid']))

        # Join the indexes with the tags.
        expression_idxs = {'t0': (-1, -1, None)}  # Initialize with the position of DCT.
        while text_tags:
            txt, tag_id = text_tags.pop(0)
            for idx, (start, end, txt_sch) in enumerate(text_blocks):
                if txt == txt_sch:
                    expression_idxs[tag_id] = (start, end)
                    # remove the items that were found.
                    text_blocks = text_blocks[idx + 1:]
                    break

        return expression_idxs

    def _get_make_instance(self) -> dict:
        make_insts = collections.defaultdict(list)
        for make_inst in self.xml_root.findall('.//MAKEINSTANCE'):
            eid = make_inst.attrib['eventID']
            make_insts[eid].append(make_inst.attrib)

        # there are some MAKEINSTANCE entries that refer to the same eid.
        # in these cases we join those items we ignore the remaining differences and join the two eiid.
        for eid, make_inst in make_insts.items():
            eiids = [entry['eiid'] for entry in make_inst]
            make_insts[eid] = make_insts[eid][0]
            make_insts[eid]['eiid'] = eiids
        return make_insts

    def _get_events(self) -> dict:
        # Most of event attributes are in <MAKEINSTACE> tag.
        make_insts = self._get_make_instance()

        events = list()
        for event in self.xml_root.findall('.//TEXT//EVENT'):
            attrib = event.attrib.copy()
            event_id = attrib['eid']
            attrib['text'] = event.text
            attrib['endpoints'] = self._expression_idxs[event_id]
            if event_id in make_insts:
                attrib.update(make_insts[event_id])
            events.append(Event(attrib))
        return events

    def _get_timexs(self) -> dict:
        timexs = list()
        for timex in self.xml_root.findall('.//TIMEX3'):
            attrib = timex.attrib.copy()
            time_id = attrib['tid']
            attrib['text'] = timex.text
            attrib['endpoints'] = self._expression_idxs[time_id]
            timexs.append(Timex(attrib))
        return timexs

    def _get_tlinks(self) -> dict:
        tlinks = list()
        for tlink in self.xml_root.findall('.//TLINK'):
            attrib = dict(
                lid=tlink.attrib['lid'],
                source=tlink.attrib['timeID'] if 'timeID' in tlink.attrib else tlink.attrib['eventInstanceID'],
                target=tlink.attrib['relatedToEventInstance'] if 'relatedToEventInstance' in tlink.attrib
                                                              else tlink.attrib['relatedToTime'],
                interval_relation=tlink.attrib['relType']
            )

            tlinks.append(TLink(attrib))
        return tlinks

    def augment_tlinks(self, relation: str = None):
        """ Augments the document tlinks by adding the symmetic relation of every tlink.
        For example if we have the tlink with A --BEFORE--> B the augmentation will add B --AFTER--> A to the document
        tlink list.

        :parameter:
            relation: a relation to limit the augmentation. If this argument is passed the method will only add the
            symmetric relation to tlink that have this relation in theis point_relation.

        :return: None
        """
        for tlink in self.tlinks:
            if relation:
                cond = [True for _, rel, _ in tlink.point_relation if rel == relation]
            else:
                cond = True

            if any(cond):
                inv_tlink = copy.copy(tlink)
                inv_tlink.invert()
                self.tlinks.append(inv_tlink)
                break

    def limit_task(self, tasks):
        """Limits the document to have only tlinks corresponding to the tasks in the task list.

        :param tasks:
        :return:
        """
        self.tlinks = [tlink for tlink in self.tlinks if tlink.task in tasks]



    def temporal_closure(self, tlinks) -> Dict:
        # TODO: Keep the original labels when the inferred are more ambiguous.
        # Remove duplicate temporal links.
        lids = set()
        relations = set()
        for lid, tlink in tlinks.items():
            scr = tlink.source
            tgt = tlink.target
            if (scr, tgt) not in relations and (tgt, scr) not in relations:
                lids.add(lid)
                relations.add((scr, tgt))

        tlinks = {lid: tlink for lid, tlink in tlinks.items() if lid in lids}

        # convert interval relations to point relations
        new_relations = {((tlink.source, scr_ep), r, (tlink.target, tgt_ep))
                         for lid, tlink in tlinks.items()
                         for scr_ep, r, tgt_ep in tlink.point_relation}

        # Add the symmetric of each relation. A < B ---> B > A
        {(tgt, _INVERSE_POINT_RELATION[rel], scr) for scr, rel, tgt in new_relations if rel != '='}


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
                    relation13 = _POINT_TRANSITIONS[relation12][relation23]
                    if not relation13:
                        continue
                    new_relation = (point1, relation13, point3)
                    if new_relation not in point_relations:
                        new_relations.add(new_relation)

        # Combine point relations.
        combined_point_relations = collections.defaultdict(list)
        for ((scr, scr_ep), rel, (tgt, tgt_ep)) in point_relations:
            if scr == tgt:
                continue
            combined_point_relations[(scr, tgt)] += [(scr_ep, rel, tgt_ep)]

        # Creat and store the valid temporal links.
        tlinks = []
        for (scr, tgt), point_relation in combined_point_relations.items():
            tlink = TLink(scr, tgt, point_relation=point_relation)
            if tlink.interval_relation:
                tlinks.append(tlink)

        # Generate indexes for tlinks.
        return {f'lc{idx}': tlink for idx, tlink in enumerate(tlinks)}
