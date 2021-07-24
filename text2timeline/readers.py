import nltk
import collections
import warnings
import re
import copy
from xml.etree import ElementTree as ET

from text2timeline.base import Timex, Event, TLink
from typing import List, Tuple


class DocumentReader:
    """

    A .tml document.

    Attributes:

        - paths

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

        self.expressions_uid = {exp.id: exp for exp in self.timexs + self.events}
        self.expressions_id = {**{exp.id: exp for exp in self.timexs}, **{exp.eid: exp for exp in self.events}}

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
        sentence_idxs = []
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

        # Find indexes of the expressions_uid.
        text_blocks = []
        start = 0
        for txt in root.itertext():
            end = start + len(txt)
            text_blocks.append((start, end, txt))
            start = end

        # Get the tags of each expression.
        text_tags = []
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

        events = []
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

        timexs = []
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

        tlinks = []
        for tlink in self.xml_root.findall('.//TLINK'):

            src_id, tgt_id = self._get_source_and_target_exp(tlink)

            # find Event/ Timex with those ids
            if src_id not in self.expressions_uid:
                msg = f"Expression {src_id} of tlink {tlink.attrib['lid']} from doc in {self.path} was not found"
                warnings.warn(msg)
                continue

            elif tgt_id not in self.expressions_uid:
                msg = f"Expression {tgt_id} of tlink {tlink.attrib['lid']} from doc in {self.path} was not found"
                warnings.warn(msg)
                continue

            source = self.expressions_uid[src_id]
            target = self.expressions_uid[tgt_id]

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


class TempEval3Document(DocumentReader):

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


TimeBankDocument = DocumentReader
TimeBankPTDocument = DocumentReader

AquaintDocument = TempEval3Document
PlatinumDocument = TempEval3Document
TimeBank12Document = TempEval3Document
