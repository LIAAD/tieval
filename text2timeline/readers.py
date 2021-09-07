import collections
from typing import List, Union, Tuple

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
        event_attrs = {event_tag.attrib["eid"]: event_tag.attrib
                       for event_tag in event_tags}

        # complementary information of each event is given on MAKEINSTANCE tags
        events = []
        for mnist_tag in xml.get_tag("MAKEINSTANCE"):

            attrib = mnist_tag.attrib
            event_id = attrib["eventID"]

            # add MAKEINSTANCE info
            event_attr = event_attrs.get(event_id)
            if event_attr:
                attrib.update(event_attr)

            events += [Event(attrib)]

        return events

    @staticmethod
    def get_timexs(xml: XMLHandler) -> List[Timex]:
        return [Timex(element.attrib) for element in xml.get_tag('TIMEX3')]

    def get_tlinks(self, xml: XMLHandler, events: List[Event], timexs: List[Timex]) -> List[TLink]:
        """Get Tlink's of the document"""

        entities = {entity.id: entity for entity in events + timexs}

        # tlinks have eiid but our reference is eid.
        # the map between eiid to eid is on the MAKEINSTANCE elements
        eiid2eid = {
            mit.attrib['eiid']: mit.attrib['eventID']
            for mit in xml.get_tag("MAKEINSTANCE")
        }

        tlinks = []
        for tlink in xml.get_tag("TLINK"):

            # retrieve source and target id.
            src_id, tgt_id = self._src_tgt_id_tlink(tlink.attrib)

            # map eiid to eid
            if src_id in eiid2eid:
                src_id = eiid2eid[src_id]

            if tgt_id in eiid2eid:
                tgt_id = eiid2eid[tgt_id]

            # build tlink
            source, target = entities.get(src_id), entities.get(tgt_id)
            if source and target:
                tlinks += [TLink(
                    id=tlink.attrib['lid'],
                    source=source,
                    target=target,
                    relation=tlink.attrib['relType']
                )]

        return tlinks

    def _src_tgt_id_tlink(self, attrib):

        src_id = attrib.get("eventInstanceID") or \
                 attrib.get("timeID") or \
                 attrib.get("eventID")

        tgt_id = attrib.get("relatedToEventInstance") or \
                 attrib.get("relatedToTime") or \
                 attrib.get("relatedToEvent")

        return src_id, tgt_id

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
Entity = Union[Event, Timex]


class TableDatasetReader:
    """Read temporal annotated files that the annotation is given on tables.
    As is the case of: MATRES, TDDiscourse and TimeBank-Dense."""

    def __init__(self, metadata: dict, base_dataset: Dataset) -> None:

        self._extension = metadata["extension"]
        self._event_id = metadata["index"]

        self._metadata = metadata

        self._base_dataset = base_dataset

    def read(self, path: str) -> Dataset:

        path = Path(path)

        documents = []
        for path in path.glob(self._extension):

            with open(path, 'r') as f:

                content = f.read()
                lines = [line.split() for line in content.split('\n')]

                # create a dictionary with docs as keys and a list of tlinks as values.
                lines_by_doc = collections.defaultdict(list)
                for line in lines:
                    if line:
                        doc_name = line[self._metadata["columns"].index("doc")]
                        lines_by_doc[doc_name] += [line]

                for doc_name, lines in lines_by_doc.items():

                    doc = self._retrieve_base_document(doc_name)

                    tlinks = []
                    for idx, line in enumerate(lines):

                        src, tgt, relation = self._decode_line(line)
                        source, target = self._get_source_target(src, tgt, doc)

                        if source and target:
                            tlinks += [
                                TLink(
                                    id=f"l{idx}",
                                    source=source,
                                    target=target,
                                    relation=relation
                                )
                            ]

                    doc.tlinks = tlinks
                    documents += [doc]

        return Dataset(path.name, documents)

    def _decode_line(self, line: List[str]):

        column_idxs = (
            self._metadata["columns"].index("src"),
            self._metadata["columns"].index("tgt"),
            self._metadata["columns"].index("relation"),
        )

        src, tgt, relation = [line[idx] for idx in column_idxs]

        if src.isdigit():
            if self._event_id == "eiid":
                src, tgt = f"ei{src}", f"ei{tgt}"

            elif self._event_id == "eid":
                src, tgt = f"e{src}", f"e{tgt}"

        return src, tgt, relation

    def _transform_eiid2eid(self, id: str, eiid2eid: dict):

        if self._event_id == "eiid":
            eiid = eiid2eid.get(id)

            if eiid:
                return eiid

        return id

    def _retrieve_base_document(self, doc_name):

        # retrieve document from the original dataset.
        doc = self._base_dataset[doc_name]

        if doc is None:
            warnings.warn(f"Document {doc_name} was not found on the source dataset "
                          f"{self._base_dataset.name}")

        return doc

    def _get_source_target(self, src: str, tgt: str, doc: Document) -> Tuple[Entity]:

        source = doc[self._transform_eiid2eid(src, doc.eiid2eid)]
        target = doc[self._transform_eiid2eid(tgt, doc.eiid2eid)]

        if source is None:
            warnings.warn(f"{src} not found on the original document {doc.name}.")

        elif target is None:
            warnings.warn(f"{tgt} not found on the original document {doc.name}.")

        return source, target


class TMLDatasetReader:
    """Handles the process of reading any temporally annotated dataset."""

    def __init__(self, metadata: dict = None) -> None:
        self.document_reader = TMLDocumentReader()

    def read(self, path: str) -> Dataset:

        path = Path(path)

        documents = [self.document_reader.read(tml_file)
                     for tml_file in path.glob("*.tml")]

        return Dataset(path.name, documents)

