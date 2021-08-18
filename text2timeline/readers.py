
from typing import List, Tuple

import nltk
import collections
import warnings
import copy

from text2timeline.base import Document, Dataset
from text2timeline.entities import Timex, Event
from text2timeline.links import TLink
from text2timeline.utils import XMLHandler

# TODO: clean the readers interface
# Move the responsabilities to handel the xml file to the XMLHandler
# Favor composition over inherentence


class TMLReader:
    """

    A .tml document.

    Attributes:

        - paths

    """

    def __init__(self):

        self.tokenizer = nltk.tokenize.WordPunctTokenizer()

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
        elements = list(root.iterfind('.//*'))

        for element in elements:

            # there are cases where there is a nested tag <EVENT><NUMEX>example</NUMEX></EVENT>
            text = ' '.join(element.itertext())

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

    def read(self, path: str) -> Document:

        return Document(name, text, events, timexs, tlinks)






class TempEval3Document:

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


# TimeBankDocument = DocumentReader
# TimeBankPTDocument = DocumentReader

AquaintDocument = TempEval3Document
PlatinumDocument = TempEval3Document
TimeBank12Document = TempEval3Document
