
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

"""
Document Readers.
"""


class TMLDocumentReader:
    """

    A .tml document.

    Attributes:

        - paths

    """

    def __init__(self):

        self.tokenizer = nltk.tokenize.WordPunctTokenizer()

    @staticmethod
    def get_events(xml: XMLHandler) -> List[Event]:
        """Retrive Events from tml file."""

        event_tags = xml.get_tag("EVENT")

        # complementary information of each event is given on MAKEINSTANCE tags
        minst_tags = xml.get_tag("MAKEINSTANCE")
        minst_attribs = {mit.attrib['eventID']: mit.attrib for mit in minst_tags}

        if len(minst_tags) != len(minst_attribs):
            warnings.warn(f"There might be multiple MAKEINSTANCE entries with the same event id in {xml.path}. "
                          f"Only the last will be used.")

        events = []
        for event_tag in event_tags:

            attrib = event_tag.attrib
            event_id = attrib['eid']

            # add MAKEINSTANCE info
            minst_attrib = minst_attribs.get(event_id)
            if minst_attrib:
                attrib.update(minst_attrib)

            events += [Event(attrib)]

        return events

    @staticmethod
    def get_timexs(xml: XMLHandler) -> List[Timex]:
        return [Timex(element.attrib) for element in xml.get_tag('TIMEX3')]

    @staticmethod
    def get_tlinks(xml: XMLHandler, events: List[Event], timexs: List[Timex]) -> List[TLink]:
        """Get Tlink's of the document"""

        entities = {entity.id: entity for entity in events + timexs}

        tlinks = []
        for tlink in xml.get_tag("TLINK"):

            # retrieve source and target id.
            attrib = tlink.attrib
            src_id = attrib.get("eventInstanceID") or attrib.get("timeID")
            tgt_id = attrib.get("relatedToEventInstance") or attrib.get("relatedToTime")

            source, target = entities.get(src_id), entities.get(tgt_id)
            if source and target:
                tlinks += [TLink(
                    id=tlink.attrib['lid'],
                    source=source,
                    target=entities[tgt_id],
                    relation=tlink.attrib['relType']
                )]

        return tlinks

    def read(self, path: Union[str, Path]) -> Document:
        """Read the tml file on the provided path."""

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


"""
Dataset Readers.
"""


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


class TMLDatasetReader:
    """Handles the process of reading any temporally annotated dataset."""

    def __init__(self):
        self.document_reader = TMLDocumentReader()

    def read(self, path: str) -> Dataset:

        path = Path(path)

        documents = [self.document_reader.read(tml_file)
                     for tml_file in path.glob("*.tml")]

        return Dataset(documents)

