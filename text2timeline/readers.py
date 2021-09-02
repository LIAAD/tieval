
from typing import List, Tuple, Union

import nltk
import warnings

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

            src_id = attrib.get("eventInstanceID") or \
                     attrib.get("timeID") or \
                     attrib.get("eventID")

            tgt_id = attrib.get("relatedToEventInstance") or \
                     attrib.get("relatedToTime") or \
                     attrib.get("relatedToEvent")

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


"""
Dataset Readers.
"""

DocName = str
SourceID, TargetID = str, str


class TableDatasetReader:
    """Read temporal annotated files that the annotation is given on tables.
    As is the case of: MATRES, TDDiscourse and TimeBank-Dense."""

    def __init__(self, metadata: dict) -> None:

        self._extension = metadata["extension"]
        self._column_idxs = (
            metadata["columns"].index("doc"),
            metadata["columns"].index("src"),
            metadata["columns"].index("tgt"),
        )
        self._base_datasets = metadata['base']

    def read(self, path: str) -> Dataset:

        path = Path(path)

        for path in path.glob(self._extension):

            with open(path, 'r') as f:

                content = f.read()
                lines = [line.split() for line in content.split('\n')]

                # create a dictionary with docs as keys and a list of tlinks as values.
                for line in lines:
                    if line:
                        doc_name, src, tgt = [line[idx] for idx in self._column_idxs]

        # TODO: where should the base dataset should be read?
        return None


class TMLDatasetReader:
    """Handles the process of reading any temporally annotated dataset."""

    def __init__(self, metadata: dict) -> None:
        self.document_reader = TMLDocumentReader()

    def read(self, path: str) -> Dataset:

        path = Path(path)

        documents = [self.document_reader.read(tml_file)
                     for tml_file in path.glob("*.tml")]

        return Dataset(path.name, documents)

