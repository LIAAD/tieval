import collections
from typing import List, Union, Tuple

import nltk
import warnings

from pathlib import Path

from text2timeline.base import Document, Dataset
from text2timeline.entities import Timex, Event
from text2timeline.links import TLink
from text2timeline.datasets.utils import TMLHandler
from text2timeline.datasets.utils import XMLHandler

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
        pass

    @staticmethod
    def get_events(xml: TMLHandler) -> List[Event]:
        """Retrive Events from tml file."""

        event_tags = xml.get_tag("EVENT")
        event_attrs = {event_tag.attrib["eid"]: event_tag.attrib
                       for event_tag in event_tags}

        # complementary information of each event is given on MAKEINSTANCE tags
        events = set()
        for mnist_tag in xml.get_tag("MAKEINSTANCE"):

            attrib = mnist_tag.attrib
            event_id = attrib["eventID"]

            # add MAKEINSTANCE info
            event_attr = event_attrs.get(event_id)
            if event_attr:
                attrib.update(event_attr)

            events.add(Event(attrib))

        return events

    @staticmethod
    def get_timexs(xml: TMLHandler) -> List[Timex]:
        return set(Timex(element.attrib) for element in xml.get_tag('TIMEX3'))

    def get_tlinks(self, xml: TMLHandler, events: List[Event], timexs: List[Timex]) -> List[TLink]:
        """Get Tlink's of the document"""

        entities = {entity.id: entity for entity in events.union(timexs)}

        # tlinks have eiid but our reference is eid.
        # the map between eiid to eid is on the MAKEINSTANCE elements
        eiid2eid = {
            mit.attrib['eiid']: mit.attrib['eventID']
            for mit in xml.get_tag("MAKEINSTANCE")
        }

        tlinks = set()
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
                tlinks.add(
                    TLink(
                        id=tlink.attrib['lid'],
                        source=source,
                        target=target,
                        relation=tlink.attrib['relType']
                    )
                )

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

        tml = TMLHandler(path)

        name = path.name.replace('.tml', '')
        text = tml.text

        events, timexs = self.get_events(tml), self.get_timexs(tml)
        tlinks = self.get_tlinks(tml, events, timexs)

        # train or test document
        if "train" in path.parts:
            set_ = "train"
        else:
            set_ = "test"

        return Document(name, text, events, timexs, tlinks, set_)


class MeanTimeDocumentReader:
    """

    A .tml document.

    Attributes:

        - paths

    """

    def __init__(self):
        pass

    @staticmethod
    def get_events(xml: XMLHandler) -> List[Event]:
        """Retrive Events from tml file."""

        event_tags = xml.get_tag("EVENT_MENTION")
        for event_tag in event_tags:
            print(event_tag.attrib["text"])
            break
        event_attrs = {["eid"]: event_tag.attrib
                       }

        # complementary information of each event is given on MAKEINSTANCE tags
        events = set()
        for mnist_tag in xml.get_tag("MAKEINSTANCE"):

            attrib = mnist_tag.attrib
            event_id = attrib["eventID"]

            # add MAKEINSTANCE info
            event_attr = event_attrs.get(event_id)
            if event_attr:
                attrib.update(event_attr)

            events.add(Event(attrib))

        return events

    @staticmethod
    def get_timexs(xml: TMLHandler) -> List[Timex]:
        return set(Timex(element.attrib) for element in xml.get_tag('TIMEX3'))

    def get_tlinks(self, xml: TMLHandler, events: List[Event], timexs: List[Timex]) -> List[TLink]:
        """Get Tlink's of the document"""

        entities = {entity.id: entity for entity in events.union(timexs)}

        # tlinks have eiid but our reference is eid.
        # the map between eiid to eid is on the MAKEINSTANCE elements
        eiid2eid = {
            mit.attrib['eiid']: mit.attrib['eventID']
            for mit in xml.get_tag("MAKEINSTANCE")
        }

        tlinks = set()
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
                tlinks.add(
                    TLink(
                        id=tlink.attrib['lid'],
                        source=source,
                        target=target,
                        relation=tlink.attrib['relType']
                    )
                )

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

        xml = XMLHandler(path)

        name = path.name.replace('.xml', '')

        # reconstruct original text
        text = xml.text

        events, timexs = self.get_events(xml), self.get_timexs(xml)
        tlinks = self.get_tlinks(xml, events, timexs)

        # train or test document
        if "train" in path.parts:
            set_ = "train"
        else:
            set_ = "test"

        return Document(name, text, events, timexs, tlinks, set_)


"""
Dataset Readers.
"""

DocName = str
SourceID, TargetID = str, str
Entity = Union[Event, Timex]


class TableDatasetReader:
    """Read temporal annotated files that the annotation is given on tables.
    As is the case of: MATRES, TDDiscourse and TimeBank-Dense."""

    def __init__(self, metadata, base_dataset: Dataset) -> None:
        self._metadata = metadata
        self._base_dataset = base_dataset

    def read(self, path: str) -> Dataset:

        path = Path(path)

        documents = set()
        with open(path, 'r') as f:

            content = f.read()
            lines = [line.split() for line in content.split('\n')]

            # create a dictionary with docs as keys and a list of tlinks as values.
            lines_by_doc = collections.defaultdict(list)
            for line in lines:
                if line:
                    doc_name = line[self._metadata.columns.index("doc")]
                    lines_by_doc[doc_name] += [line]

            # add tlinks from table to the base document
            for doc_name, lines in lines_by_doc.items():

                doc = self._retrieve_base_document(doc_name)

                tlinks = set()
                for idx, line in enumerate(lines):

                    src, tgt, relation = self._decode_line(line)
                    source, target = self._get_source_target(src, tgt, doc)

                    if source and target:
                        tlinks.add(
                            TLink(
                                id=f"l{idx}",
                                source=source,
                                target=target,
                                relation=relation
                            )
                        )

                doc.tlinks = tlinks
                documents.add(doc)

        return Dataset(path.name, documents)

    def _decode_line(self, line: List[str]):

        column_idxs = (
            self._metadata.columns.index("src"),
            self._metadata.columns.index("tgt"),
            self._metadata.columns.index("relation"),
        )

        src, tgt, relation = [line[idx] for idx in column_idxs]

        return src, tgt, relation

    def _resolve_id(self, id, doc):

        # to address the problem of timebank-dense where t0 is referred but it is not defined on timebank.
        if id == "t0":
            return doc.dct.id

        # MATRES only has the id number (ex: "e105" appears as 105)
        if id.isdigit():
            if self._metadata.event_index == "eiid":
                eiid = f"ei{id}"
                eid = doc._eiid2eid.get(eiid)
                return eid

            elif self._metadata.event_index == "eid":
                eid = f"e{id}"
                return eid

        return id

    def _retrieve_base_document(self, doc_name):

        # retrieve document from the original dataset.
        doc = self._base_dataset[doc_name]

        if doc is None:
            warnings.warn(f"Document {doc_name} was not found on the source dataset "
                          f"{self._base_dataset.name}")

        return doc

    def _get_source_target(self, src: str, tgt: str, doc: Document) -> Tuple[Entity]:

        source = doc[self._resolve_id(src, doc)]
        target = doc[self._resolve_id(tgt, doc)]

        if source is None:
            warnings.warn(f"{src} not found on the original document {doc.name}.")

        elif target is None:
            warnings.warn(f"{tgt} not found on the original document {doc.name}.")

        return source, target


class XMLDatasetReader:
    """Handles the process of reading any temporally annotated dataset stored with .tml or .xml extension."""

    def __init__(self, doc_reader) -> None:
        self.document_reader = doc_reader()

    def read(self, path: str) -> Dataset:

        path = Path(path)

        if not path.is_dir():
            raise IOError(f"The dataset being load have not been downloaded yet.")

        # documents = [self.document_reader.read(file)
        #              for file in path.glob("**/*.[tx]ml")]

        docs = []
        for file in path.glob("**/*.[tx]ml"):
            doc = self.document_reader.read(file)
            docs += [doc]

        return Dataset(path.name, docs)
