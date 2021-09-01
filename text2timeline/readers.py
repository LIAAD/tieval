
from pprint import pprint

from typing import List, Tuple, Union

import nltk
import collections
import warnings
import copy

from pathlib import Path

from text2timeline.base import Document, Dataset
from text2timeline.entities import Timex, Event
from text2timeline.links import TLink
from text2timeline.utils import XMLHandler

# TODO: clean the readers interface
# Move the responsibilities to handle the xml file to the XMLHandler


class TMLDocumentReader:
    """

    A .tml document.

    Attributes:

        - paths

    """

    def __init__(self):

        self.tokenizer = nltk.tokenize.WordPunctTokenizer()

    def get_events(self, xml: XMLHandler) -> List[Event]:
        """Retrive Events from tml file."""

        event_tags = xml.get_tag("EVENT")

        # complementary information of each event is given on MAKEINSTANCE tags
        minst_tags = xml.get_tag("MAKEINSTANCE")
        minst_attrib = {mit.attrib['eventID']: mit.attrib for mit in minst_tags}

        events = []
        for et in event_tags:

            attrib = et.attrib
            event_id = attrib['eid']

            # add MAKEINSTANCE info
            attrib.update(minst_attrib.get(event_id))
            events += [Event(attrib)]

        return events

    def get_timexs(self, xml: XMLHandler) -> List[Timex]:
        return [Timex(element.attrib) for element in xml.get_tag('TIMEX3')]

    def get_tlinks(self, xml: XMLHandler, events: List[Event], timexs: List[Timex]) -> List[TLink]:
        """
        Get keys for each dataset

        np.unique([key for tlink in self.xml_root.findall('.//TLINK') for key in tlink.attrib])

        :return:
        """

        entities = events + timexs

        tlinks = []
        for tlink in xml.get_tag("TLINK"):

            # retrive source and target id.
            attrib = tlink.attrib
            src_id = attrib.get("eventID")
            tgt_id = attrib.get("relatedToEvent") or attrib.get("relatedToTime")

            source = None
            target = self.expressions_uid[tgt_id]

            tlink = TLink(
                id=tlink.attrib['lid'],
                source=source,
                target=target,
                relation=tlink.attrib['relType']
            )

            tlinks += [tlink]

        return tlinks

    def read(self, path: Union[str, Path]) -> Document:

        if not isinstance(path, Path):
            path = Path(path)

        tml = XMLHandler(path)

        name = path.name.replace('.tml', '')
        text = tml.text

        events, timexs = self.get_events(tml), self.get_timexs(tml)
        tlinks = self.get_tlinks(tml, events, timexs)

        return Document(name, text, events, timexs, tlinks)


def _read_matres_tlink(self, line: list, doc: str) -> TLink:
    """Read a temporal link from the MATRES dataset."""

    src, tgt = f'ei{line[2]}', f'ei{line[3]}'

    source = self._find_expressions(src, doc)
    target = self._find_expressions(tgt, doc)

    if source is None or target is None:
        return None, None

    tlink = TLink(
        id=f'l{self._tlink_id}',
        source=source,
        target=target,
        relation=line[4]
    )

    self._tlink_id += 1
    return tlink


def _read_tddiscourse_tlink(self, line: list, doc: str):
    """Read a temporal link from the TDDiscourse dataset."""

    src, tgt = line[0], line[1]

    source = self._find_expressions(src, doc)
    target = self._find_expressions(tgt, doc)

    if source is None or target is None:
        return None, None

    tlink = TLink(
        id=f'l{self._tlink_id}',
        source=source,
        target=target,
        relation=line[2]
    )

    self._tlink_id += 1
    return tlink


def _read_tbdense_tlink(self, line: list, doc: str):
    """Read a temporal link from the Timebank Dense dataset."""

    return self._tddiscourse(line, doc)


# mapping between the name of the dataset and the respective tlink reader
TLINK_READER = {
    'matres': _read_matres_tlink,
    'tddiscourse': _read_tddiscourse_tlink,
    'tbdense': _read_tbdense_tlink,

}


class TableDatasetReader:
    """Read temporal annotated files that the annotation is given on tables.
    As is the case of: MATRES, TDDiscourse and TimeBank-Dense."""

    def __init__(self, dataset_name: str):

        self.dataset = dataset_name
        self.tlink_reader = TLINK_READER[dataset_name]

    def read(self, table):

        with open(table, 'r') as f:
            content = [line.split() for line in f.read().split('\n')]

            tlinks_by_doc = collections.defaultdict(list)
            for line in content:
                if line:
                    tlinks_by_doc[line[0]] += [line[1:]]  # doc: tlink

            tlinks = {}
            for doc, lines in tlinks_by_doc.items():

                if doc not in self.dataset.doc_names:
                    warnings.warn(f"Document {doc} not found on the source dataset")
                    continue

                tlinks[doc] = [self.tlink_reader(line, doc) for line in lines]

        return tlinks

    def _find_expressions(self, exp_id: str, doc: str):

        if exp_id in self.dataset[doc].expressions_uid:
            return self.dataset[doc].expressions_uid[exp_id]

        elif exp_id in self.dataset[doc].expressions_id:
            return self.dataset[doc].expressions_id[exp_id]

        else:
            warnings.warn(f"Expression with id {exp_id} was not found in document {doc}")
            return None


class DatasetReader:
    """Handles the process of reading any temporally annotated dataset."""
    pass


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
