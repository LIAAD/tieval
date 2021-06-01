import collections
import re
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from xml.etree import ElementTree as ET
import copy

import nltk

from pprint import pprint

import numpy as np

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
    "INCLUDES": [(_START, "<", _START), (_END, '>', _END)],
    "IS_INCLUDED": [(_START, '>', _START), (_END, "<", _END)],
    "BEGINS-ON": [(_START, "=", _START)],
    "ENDS-ON": [(_END, "=", _END)],
    "BEGINS": [(_START, "=", _START), (_END, "<", _END)],
    "BEGUN_BY": [(_START, "=", _START), (_END, '>', _END)],
    "ENDS": [(_START, '>', _START), (_END, "=", _END)],
    "ENDED_BY": [(_START, "<", _START), (_END, "=", _END)],
    "SIMULTANEOUS": [(_START, "=", _START), (_END, "=", _END)],
    "OVERLAP": [(_START, "<", _END), (_END, '>', _START)],
    "VAGUE": [],
    "CONTAINS": [(_START, "<", _START), (_END, "<", _END)],
    "IDENTITY": [(_START, "=", _START), (_END, "=", _END)],
    "DURING": [(_START, "=", _START), (_END, "=", _END)],
    "DURING_INV": [(_START, "=", _START), (_END, "=", _END)],
    'BEFORE-OR-OVERLAP': [(_START, '<', _START), (_END, '<', _END)],
    'OVERLAP-OR-AFTER': [(_START, '>', _START), (_END, '>', _END)]
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
                (_END, "<", _END)],
    "VAGUE": [(_START, None, _START),
              (_START, None, _END),
              (_END, None, _START),
              (_END, None, _END)],
    'BEFORE-OR-OVERLAP': [(_START, '<', _START),
                          (_START, '<', _END),
                          (_END, None, _START),
                          (_END, '<', _END)],
    'OVERLAP-OR-AFTER': [(_START, '>', _START),
                          (_START, None, _END),
                          (_END, '>', _START),
                          (_END, '>', _END)]
}

# transitivity table for point relations
_POINT_TRANSITIONS = {
    '<': {'<': '<', '=': '<', '>': None},
    '=': {'<': '<', '=': '=', '>': '>'},
    '>': {'>': '>', '=': '>', '<': None}
}

_POINT_RELATIONS = list(_INTERVAL_TO_POINT.values()) + \
                   list(_INTERVAL_TO_POINT_COMPLETE.values())

_INTERVAL_RELATIONS = list(_INTERVAL_TO_POINT_COMPLETE.keys())

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
    'SIMULTANEOUS': 'SIMULTANEOUS',
    'OVERLAP': 'OVERLAP',
    'VAGUE': 'VAGUE'
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
    'IAFTER': 'IAFTER',

    'VAGUE': 'VAGUE',
    'BEFORE-OR-OVERLAP': 'BEFORE-OR-OVERLAP'
}

PointRelation = List[Tuple[int, str, int]]
IntervalRelation = str


class Timex:

    def __init__(self, attributes: Dict):
        attr = collections.defaultdict(lambda: None, attributes)

        self._tid = attr['tid']
        self.type = attr['type']
        self.value = attr['value']
        self.temporal_function = attr['temporalFunction']
        self.function_in_document = attr['functionInDocument']
        self.anchor_time_id = attr['anchorTimeID']
        self.text = attr['text']
        self.endpoints = attr['endpoints']

    def __repr__(self):
        return f"Timex(tid={self.id})"

    @property
    def id(self):
        return self._tid

    @property
    def is_dct(self):
        if self.function_in_document == 'CREATION_TIME':
            return True

        return False


class Event:
    def __init__(self, attributes: dict):
        attr = collections.defaultdict(lambda: None, attributes)

        self.eid = attr['eid']
        self.eiid = attr['eiid']
        self.family = attr['class']
        self.stem = attr['stem']
        self.aspect = attr['aspect']
        self.tense = attr['tense']
        self.polarity = attr['polarity']
        self.pos = attr['pos']
        self.text = attr['text']
        self.endpoints = attr['endpoints']

    def __repr__(self):
        return f"Event(eid={self.eid})"

    @property
    def id(self):
        if self.eiid:
            return self.eiid

        return self.eid

    @property
    def is_dct(self):
        return False


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

    def __init__(self,
                 id: str,
                 source: Union[Timex, Event],
                 target: Union[Timex, Event],
                 relation: Union[IntervalRelation, PointRelation],
                 **kwargs):

        self.id = id
        self.source = source
        self.target = target

        self._relation = None
        self.relation = relation

        for key, value in kwargs.items():
            if key not in self.__dict__:
                setattr(self, key, value)

    def __str__(self):
        return f"{self.source} ---{self.relation}--> {self.target}"

    def __repr__(self):
        return f"TLink(id={self.id})"

    def __and__(self, other):
        """ Infer the relation between two TLINKS.

        If a relation can be infered it will return a Tlink between source of the first Tlink and target of the second
        Tlink.

        Example:
            tlink_1 = TLink({
                'id': 'l1',
                'source': 'e1',
                'target': 'e2',
                'relation': 'BEFORE'
            })

            tlink_2 = TLink({
                'id': 'l2',
                'source': 'e2',
                'target': 'e3',
                'relation': 'BEFORE'
            })

            tlink_1 & tlink_2

        :param other:
        :return:
        """

        # pair the relations of the first and second tlink
        paired_relations = zip(self.point_relation_complete, other.point_relation_complete)

        # get the relation between source of the first tlink and target of second tlink
        point_relation13 = [
            (relation12[0], _POINT_TRANSITIONS[relation12[1]][relation23[1]], relation23[2])
            for relation12, relation23 in paired_relations
        ]

        # search for a interval relation that matches the found point relation
        interval_relation = None
        for i_rel, p_rel in _INTERVAL_TO_POINT_COMPLETE.items():
            if p_rel == point_relation13:
                interval_relation = i_rel

        # if an interval relation was found it will return a TLink with it. otherwise it returns None
        if interval_relation:
            return TLink(
                id=f'il{self.source}{other.target}',
                source=self.source,
                target=other.target,
                relation=interval_relation,
            )

        else:
            return None

    def __invert__(self):
        """

        Invert TLink.
        Returns the symmetric tlink. For example, if A --Before--> B it will return a tlink with B --After--> A

        :return:
            A TLink symmetric to the current one.

        """

        return TLink(
            id=f'i{self.id}',
            source=self.target,
            target=self.source,
            relation=_INVERSE_INTERVAL_RELATION[self.relation]
        )

    @property
    def relation(self):
        return self._relation

    @relation.setter
    def relation(self, relation):

        if relation in _INTERVAL_RELATIONS:
            self._relation = relation

        elif relation in _INTERVAL_TO_POINT.values():
            self._relation = [rel
                              for rel, requirements in _INTERVAL_TO_POINT.items()
                              if set(requirements).issubset(relation)]

        elif relation in _INTERVAL_TO_POINT_COMPLETE.values():
            self._relation = [rel
                              for rel, requirements in _INTERVAL_TO_POINT_COMPLETE.items()
                              if set(requirements).issubset(relation)][0]

        else:
            raise ValueError(f"{relation} is not a valid relation.")

    @property
    def point_relation(self):
        return _INTERVAL_TO_POINT[self.relation]

    @property
    def point_relation_complete(self):
        return _INTERVAL_TO_POINT_COMPLETE[self.relation]

    def normalize_relation(self):
        """This method is usefull to remove redundent relations. For instances "OVERLAP" and "SIMULTANIUES" are the same
        temporal relation but there are datasets that use both.
        """
        self.relation = _ASSERT_RELATION[self.relation]

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
        if scr.is_dct or tgt.is_dct:
            return 'D'
        elif (scr.id[0] == 'e') and (tgt.id[0] == 'e'):
            return 'A'
        elif (scr.id[0] == 't') and (tgt.id[0] == 't'):
            return 'B'
        else:
            return 'C'


class Document:
    """

    A .tml document.

    Attributes:

        - path

    """

    def __init__(self, path: str):
        self.path = path

        self.tokenizer = nltk.tokenize.WordPunctTokenizer()
        self.xml_root = ET.parse(path).getroot()
        self.name = self._get_name()

        self.text = self._get_text()
        self.sentences = self._get_sentences()
        self.tokens = self._get_tokens()

        self.dct = self._get_dct()

        self._expression_idxs = self._expression_indexes()
        self.timexs = self._get_timexs()
        self.events = self._get_events()
        self.expressions = {exp.id: exp for exp in self.timexs + self.events}

        self.tlinks = self._get_tlinks()

    def __repr__(self):
        return f'Document(name={self.name})'

    def __str__(self):
        return self.text.strip()

    def _get_name(self):
        name = self.path.split('/')[-1].strip('.tml')
        return name

    def _get_dct(self):
        """Extract document creation time"""

        # TODO: improve this syntax

        # dct is always the first TIMEX3 element
        dct = self.xml_root.find(".//TIMEX3[@functionInDocument='CREATION_TIME']")

        if dct is None:
            dct = self.xml_root.find(".//TIMEX3[@functionInDocument='PUBLICATION_TIME']")

        return Timex(dct.attrib)

    def _remove_xml_tags(self, root: ET.Element, tags2keep: List[str] = ['TIMEX3', 'EVENT']) -> ET.Element:
        """ Removes tags all tags in xml_root that are not on tags2keep list.

        :param root:
        :param tags2keep:
        :return:
        """
        raw_root = ET.tostring(root, encoding='unicode')

        tags2remove = set(elem.tag for elem in root.findall('.//*') if elem.tag not in tags2keep)
        for tag in tags2remove:
            raw_root = re.sub(f'</?{tag}>|</?{tag}\s.*?>', '', raw_root)

        return ET.fromstring(raw_root)

    def _get_text(self) -> str:
        """
        Returns the raw text of the document
        :return:
        """
        text = ''.join(list(self.xml_root.itertext()))
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

        root = copy.deepcopy(self.xml_root)

        # remove unnecessary tags.
        root = self._remove_xml_tags(root)

        # Find indexes of the expressions.
        text_blocks = list()
        start = 0
        for txt in root.itertext():
            end = start + len(txt)
            text_blocks.append((start, end, txt))
            start = end

        # Get the tags of each expression.
        text_tags = list()
        elements = [elem for elem in list(root.iterfind('.//*'))]

        for element in elements:

            # there are cases where there is a nested tag <EVENT><NUMEX>example</NUMEX></EVENT>
            text = ' '.join(list(element.itertext()))

            if element.attrib and element.tag == 'EVENT':
                text_tags.append((text, element.attrib['eid']))

            elif element.attrib and element.tag == 'TIMEX3':
                text_tags.append((text, element.attrib['tid']))

        # Join the indexes with the tags.
        expression_idxs = {self.dct.id: (-1, -1)}  # Initialize with the position of DCT.
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

        return make_insts

    def _get_events(self) -> dict:
        # Most of event attributes are in <MAKEINSTACE> tag.
        make_insts = self._get_make_instance()

        events = list()
        for event in self.xml_root.findall('.//EVENT'):
            attrib = event.attrib.copy()
            event_id = attrib['eid']
            attrib['text'] = event.text
            attrib['endpoints'] = self._expression_idxs[event_id]

            if event_id in make_insts:

                attribs = []
                for make_inst in make_insts[event_id]:
                     attrib_copy = attrib.copy()
                     attrib_copy.update(make_inst)
                     attribs += [attrib_copy]

            else:
                attribs = [attrib]

            events += [Event(attrib) for attrib in attribs]
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
        """
        Get keys for each dataset

        np.unique([key for tlink in self.xml_root.findall('.//TLINK') for key in tlink.attrib])

        :return:
        """

        tlinks = list()

        for tlink in self.xml_root.findall('.//TLINK'):

            src_id, tgt_id = self._get_source_and_target_exp(tlink)

            # find Event/ Timex with those ids
            source = self.expressions[src_id]
            target = self.expressions[tgt_id]

            tlink = TLink(
                id=tlink.attrib['lid'],
                source=source,
                target=target,
                relation=tlink.attrib['relType'],
                **tlink.attrib
            )

            tlinks += [tlink]

        return tlinks

    def _get_source_and_target_exp(self, tlink):

        # scr_id
        src_id = tlink.attrib['eventID']

        # tgt_id
        if 'relatedToEvent' in tlink.attrib:
            tgt_id = tlink.attrib['relatedToEvent']
        else:
            tgt_id = tlink.attrib['relatedToTime']

        return src_id, tgt_id

    def augment_tlinks(self, relations: List[str] = None):
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
            if tlink.relation:
                tlinks.append(tlink)

        # Generate indexes for tlinks.
        return {f'lc{idx}': tlink for idx, tlink in enumerate(tlinks)}


TimeBankDocument = Document
AquaintDocument = Document
TempEval3Document = Document
TimeBankPTDocument = Document


class TimeBank12Document(Document):

    def _get_source_and_target_exp(self, tlink):

        # scr_id
        if 'eventInstanceID' in tlink.attrib:
            src_id = tlink.attrib['eventInstanceID']
        else:
            src_id = tlink.attrib['timeID']

        # tgt_id
        if 'relatedToEventInstance' in tlink.attrib:
            tgt_id = tlink.attrib['relatedToEventInstance']
        else:
            tgt_id = tlink.attrib['relatedToTime']

        return src_id, tgt_id

