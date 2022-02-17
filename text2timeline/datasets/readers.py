from typing import List, Tuple
from typing import Iterable

import abc
import warnings
import collections

from tqdm import tqdm
import string

from pathlib import Path

from text2timeline.base import Document, Dataset
from text2timeline.entities import Timex, Event, Entity
from text2timeline.links import TLink
from text2timeline.datasets.utils import xml2dict
from text2timeline.temporal_relation import SUPPORTED_RELATIONS


def _detokenize(tokens):
    text = [
        " " + tkn
        if not tkn.startswith("'") and tkn not in string.punctuation
        else tkn
        for tkn in tokens
    ]

    return "".join(text).strip()


"""
Document Readers.
"""


class BaseDocumentReader:

    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        self.path = path
        self.content = xml2dict(self.path)

    @property
    @abc.abstractmethod
    def _name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def _dct(self) -> Timex:
        pass

    @property
    @abc.abstractmethod
    def _text(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def _entities(self) -> Iterable[Entity]:
        pass

    @property
    @abc.abstractmethod
    def _tlinks(self) -> Iterable[TLink]:
        pass

    def read(self) -> Document:
        return Document(
            name=self._name,
            dct=self._dct,
            text=self._text,
            entities=self._entities,
            tlinks=self._tlinks
        )


class TempEval3DocumentReader(BaseDocumentReader):

    @property
    def _name(self) -> str:
        return self.path.parts[-1]

    @property
    def _text(self) -> str:
        return self.content["TimeML"]["TEXT"]["text"]

    @property
    def _entities(self) -> Iterable[Entity]:

        def assert_list(entities):

            if entities and not isinstance(entities, list):
                return [entities]

            return entities

        root = self.content["TimeML"]

        events = assert_list(root["TEXT"].get("EVENT"))
        timexs = assert_list(root["TEXT"].get("TIMEX3"))
        mkinsts = assert_list(root.get("MAKEINSTANCE"))

        result = set()
        events_dict = {event["eid"]: event for event in events}
        if mkinsts:

            # add makeintance information to events
            for mkinst in mkinsts:

                event = events_dict.get(mkinst["eid"])
                if event:
                    mkinst.update(event)

                result.add(Event(
                    aspect=mkinst['aspect'],
                    class_=mkinst['class'],
                    id=mkinst['eiid'],
                    polarity=mkinst['polarity'],
                    pos=mkinst['pos'],
                    tense=mkinst['tense'],
                    text=mkinst['text'],
                ))

        # add timexs to entities
        if timexs:
            for timex in timexs:
                result.add(Timex(
                    function_in_document=timex.get("functionInDocument"),
                    text=timex["text"],
                    id=timex["tid"],
                    type_=timex["type"],
                    value=timex["value"]
                ))

        return result

    @property
    def _dct(self) -> Timex:
        attrib = self.content["TimeML"]["TIMEX3"]
        return Timex(
            function_in_document=attrib["functionInDocument"],
            text=attrib["text"],
            id=attrib["tid"],
            type_=attrib["type"],
            value=attrib["value"]
        )

    @property
    def _tlinks(self) -> Iterable[TLink]:

        entities = set.union(self._entities, {self._dct})
        entities_dict = {ent.id: ent for ent in entities}

        tlinks = self.content["TimeML"].get("TLINK")

        result = set()

        for tlink in tlinks:
            result.add(
                TLink(
                    id=tlink["lid"],
                    source=entities_dict[tlink["from"]],
                    target=entities_dict[tlink["to"]],
                    relation=tlink["relType"]
                )
            )

        return result


class MeanTimeDocumentReader(BaseDocumentReader):

    @property
    def _name(self) -> str:
        return self.path.parts[-1]

    @property
    def _text(self) -> str:
        tokens = self.content["Document"]["token"]
        sent_n = 0
        text, sent = [], []
        for token in tokens:

            if int(token["sentence"]) != sent_n:
                text += [_detokenize(sent)]
                sent = []
                sent_n += 1

            sent += [token["text"]]

        return "\n".join(text)

    @property
    def _entities(self) -> Iterable[Entity]:

        def get_text(token_anchor, tokens):

            if not isinstance(token_anchor, list):
                token_anchor = [token_anchor]

            text = [tokens[tkn["t_id"]] for tkn in token_anchor]

            return " ".join(text)

        tokens = {
            tkn["t_id"]: tkn["text"]
            for tkn in self.content["Document"]["token"]
        }

        entities = set()

        # events
        events = self.content["Document"]["Markables"]["EVENT_MENTION"]
        for event in events:

            # retrieve text
            if event.get("token_anchor"):
                text = get_text(event["token_anchor"], tokens)
                event["text"] = text

            entities.add(Event(
                id=event["m_id"],
                tense=event.get("tense"),
                pos=event.get("pos"),
                text=event["text"],
            ))

        # timex
        timexs = self.content["Document"]["Markables"]["TIMEX3"]

        if not isinstance(timexs, list):
            timexs = [timexs]

        for timex in timexs:

            is_dct = timex["functionInDocument"] == "CREATION_TIME"
            is_descriptor = "TAG_DESCRIPTOR" in timex
            if is_dct or is_descriptor:
                continue

            # retrieve text
            if timex.get("token_anchor"):
                text = get_text(timex["token_anchor"], tokens)
                timex["text"] = text

            entities.add(Timex(id=timex["m_id"], text=timex["text"], type_=timex["type"],
                               function_in_document=timex["functionInDocument"]))

        return entities

    @property
    def _dct(self) -> Timex:

        timexs = self.content["Document"]["Markables"]["TIMEX3"]
        if not isinstance(timexs, list):
            timexs = [timexs]

        for timex in timexs:
            if timex["functionInDocument"] == 'CREATION_TIME':
                return Timex(id=timex["m_id"], text=timex["value"], value=timex["value"],
                             function_in_document=timex["functionInDocument"])

    @property
    def _tlinks(self) -> Iterable[TLink]:

        tlinks = self.content["Document"]["Relations"]["TLINK"]

        entities_dict = {entity.id: entity for entity in self._entities}
        entities_dict[self._dct.id] = self._dct

        result = set()
        for tlink in tlinks:

            id = tlink.get("r_id")
            relation = tlink.get("reltype")
            source = tlink.get("source")
            target = tlink.get("target")

            # proceed in case any of the required information is missing.
            if not all([relation, source, target, id]):
                continue

            # ensure that the relation is supported by the framework
            if relation in SUPPORTED_RELATIONS:

                result.add(
                    TLink(
                        id=id,
                        source=source["m_id"],
                        target=target["m_id"],
                        relation=relation
                    )
                )

        return result

    def read(self) -> Document:

        return Document(
            name=self._name,
            dct=self._dct,
            text=self._text,
            entities=self._entities,
            tlinks=self._tlinks,
            language=self.content["Document"]["lang"]
        )


# TODO: add support for EXTRA corpus
class GraphEveDocumentReader(BaseDocumentReader):

    @property
    def _root(self):
        if "Article" in self.content:
            return self.content["Article"]

        return self.content["ProcessedArticle"]

    @property
    def _name(self) -> str:
        return self.path.parts[-1]

    @property
    def _dct(self) -> Timex:
        return None

    @property
    def _text(self) -> str:
        self.content.values()
        return self._root["Text"]

    @property
    def _entities(self) -> Iterable[Entity]:

        entities = set()

        # events
        events = self._root["MicroEvents"]["MicroEvent"]
        for event in events:

            entities.add(Event(
                id=event["ID"],
                text=event["EventCarrier"],
                stem=event["EventCarrierTaggedWord"]["Stem"],
                pos=event["EventCarrierTaggedWord"]["POSTag"],
                lemma=event["EventCarrierTaggedWord"]["Lemma"],
                endpoints=(
                    int(event["EventCarrierTaggedWord"]["DocumentStartPosition"]),
                    int(event["EventCarrierTaggedWord"]["DocumentStartPosition"]) + len(event["EventCarrier"])
                )
            ))

        # timexs
        timexs = []
        sentences = self._root["Sentences"]["Sentence"]
        for sentence in sentences:
            if sentence["TemporalExpressions"]:
                timex = sentence["TemporalExpressions"]["TemporalExpression"]
                if isinstance(timex, list):
                    timexs += timex
                else:
                    timexs += [timex]

        for timex in timexs:
            entities.add(Timex(
                id=timex["TimexID"],
                value=timex.get("Value"),
                text=timex["Text"],
                type_=timex["Type"]
            ))

        return entities

    @property
    def _tlinks(self) -> Iterable[TLink]:

        result = set()

        entities_dict = {ent.id: ent for ent in self._entities}

        annotations = self._root["Relations"]["Relation"]
        for annot in annotations:

            relation = annot["RelationType"]

            if relation.startswith("Temporal"):

                relation = relation.replace("Temporal", "")
                source = entities_dict[annot["FirstEvent"]["ID"]]
                target = entities_dict[annot["SecondEvent"]["ID"]]

                result.add(TLink(
                    source=source,
                    target=target,
                    relation=relation
                ))

        return result


"""
Dataset Readers.
"""

DocName = str
SourceID, TargetID = str, str


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
            return doc._dct.id

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
        self.document_reader = doc_reader

    def read(self, path: str) -> Dataset:

        path = Path(path)

        if not path.is_dir():
            raise IOError(f"The dataset being load have not been downloaded yet.")

        train, test = [], []
        files = list(path.glob("**/*.[tx]ml"))
        for file in tqdm(files):
            reader = self.document_reader(file)
            document = reader.read()

            if "test" in file.parts:
                test += [document]

            else:
                train += [document]

        return Dataset(path.name, train, test)
