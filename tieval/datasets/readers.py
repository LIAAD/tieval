from typing import List, Tuple
from typing import Iterable

import abc
import json
import warnings
import collections

import re

from tqdm import tqdm
import string

from pathlib import Path

from xml.etree import ElementTree as ET

from tieval.base import Document, Dataset
from tieval.entities import Timex, Event, Entity
from tieval.links import TLink
from tieval.datasets.utils import xml2dict
from tieval.temporal_relation import SUPPORTED_RELATIONS

from pprint import pprint



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

    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        self.path = path
        self.content = xml2dict(self.path)

        self.xml = ET.parse(self.path)

    @property
    def _name(self) -> str:
        return self.path.parts[-1]

    @property
    def _text(self) -> str:
        text_tag = self.xml.find("TEXT")
        text_blocks = list(text_tag.itertext())
        return "".join(text_blocks)

    @property
    def _entities(self) -> Iterable[Entity]:

        def assert_list(entities):

            if entities and not isinstance(entities, list):
                return [entities]

            return entities

        entities = set()

        events = assert_list(self.content["TimeML"]["TEXT"].get("EVENT"))
        timexs = assert_list(self.content["TimeML"]["TEXT"].get("TIMEX3"))

        # events
        events_dict = {event["eid"]: event for event in events}
        mkinsts = assert_list(self.content["TimeML"].get("MAKEINSTANCE"))
        if mkinsts:

            # add makeintance information to events
            for mkinst in mkinsts:

                event = events_dict.get(mkinst["eid"])
                if event:
                    mkinst.update(event)

                s, e = mkinst["endpoints"].split()
                entities.add(Event(
                    aspect=mkinst['aspect'],
                    class_=mkinst['class'],
                    id=mkinst['eiid'],
                    polarity=mkinst['polarity'],
                    pos=mkinst['pos'],
                    tense=mkinst['tense'],
                    text=mkinst['text'],
                    endpoints=(int(s), int(e)),
                    sent_idx=int(mkinst.get("sent_idx"))
                ))

        # timexs
        if timexs:
            for timex in timexs:

                s, e = timex["endpoints"].split()

                entities.add(Timex(
                    function_in_document=timex.get("functionInDocument"),
                    text=timex["text"],
                    id=timex["tid"],
                    type_=timex["type"],
                    value=timex["value"],
                    endpoints=(int(s), int(e)),
                    sent_idx=int(timex.get("sent_idx"))
                ))

        return entities

    @property
    def _dct(self) -> Timex:
        attrib = self.content["TimeML"]["TIMEX3"]
        return Timex(
            function_in_document=attrib["functionInDocument"],
            text=attrib["text"],
            id=attrib["tid"],
            type_=attrib["type"],
            value=attrib["value"],
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

    def __init__(self, path):

        if not isinstance(path, Path):
            path = Path(path)

        self.path = path
        self.content = xml2dict(self.path)

    @property
    def _name(self) -> str:
        return self.path.parts[-1]

    @property
    def _text(self) -> str:

        text = self.content["Document"]["raw"]

        # add endpoints to tokens
        idx = 0
        running_text = text
        for tkn in self.content["Document"]["token"]:
            offset = running_text.find(tkn["text"])
            idx += offset
            tkn["endpoints"] = (idx, idx + len(tkn["text"]))
            idx += len(tkn["text"])
            running_text = running_text[offset + len(tkn["text"]):]

        return text

    @property
    def _entities(self) -> Iterable[Entity]:

        def get_endpoints(token_anchor, tokens):

            if not isinstance(token_anchor, list):
                token_anchor = [token_anchor]

            endpoints = [
                endpoint
                for tkn in token_anchor
                for endpoint in tokens[tkn["t_id"]]["endpoints"]
            ]

            return endpoints[0], endpoints[-1]

        tokens = {
            tkn["t_id"]: tkn
            for tkn in self.content["Document"]["token"]
        }

        entities = set()

        # events
        events = self.content["Document"]["Markables"]["EVENT_MENTION"]
        for event in events:

            # retrieve text
            s, e = get_endpoints(event["token_anchor"], tokens)
            text = self._text[s: e]

            entities.add(Event(
                id=event["m_id"],
                tense=event.get("tense"),
                pos=event.get("pos"),
                text=text,
                endpoints=(s, e)
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

            # retrieve endpoints and text
            s, e = get_endpoints(timex["token_anchor"], tokens)
            text = self._text[s: e]

            entities.add(Timex(
                id=timex["m_id"],
                text=text,
                endpoints=(s, e),
                type_=timex["type"],
                function_in_document=timex["functionInDocument"]),
            )

        return entities

    @property
    def _dct(self) -> Timex:

        timexs = self.content["Document"]["Markables"]["TIMEX3"]
        if not isinstance(timexs, list):
            timexs = [timexs]

        for timex in timexs:
            if timex["functionInDocument"] == 'CREATION_TIME':
                return Timex(
                    id=timex["m_id"],
                    text=timex["value"],
                    value=timex["value"],
                    function_in_document=timex["functionInDocument"]
                )

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

    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        self.path = path
        self.content = xml2dict(self.path)

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


class TempEval2DocumentReader(BaseDocumentReader):

    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        self.path = path
        self.content = json.load(self.path.open())

    @property
    def _name(self) -> str:
        return self.content["name"]

    @property
    def _dct(self) -> Timex:

        timexs = self.content["entities"]["timexs"]
        for timex in timexs:
            if "function_in_document" in timex:
                return Timex(
                    id=timex["id"],
                    sent_idx=timex["sent_idx"],
                    tkn_idx=timex["tkn_idx"],
                    value=timex["val"],
                    function_in_document=timex["function_in_document"]
                )

    @property
    def _text(self) -> str:
        return self.content["raw"]

    @property
    def _entities(self) -> Iterable[Entity]:

        entities = set()

        token_dict = {
            f"s{tkn['sent_idx']}t{tkn['tkn_idx']}": tkn
            for tkn in self.content["tokens"]
        }

        # events
        for event in self.content["entities"]["events"]:

            # find entity endpoints
            endpoints = []
            for idx in event["tkn_idx"]:
                key = f"s{event['sent_idx']}t{idx}"
                token = token_dict[key]
                endpoints += [*token["endpoints"]]
            s, e = endpoints[0], endpoints[-1]

            event["endpoints"] = [s, e]
            event["text"] = self._text[s: e]

            entities.add(Event(
                id=event["id"],
                sent_idx=event["sent_idx"],
                tkn_idx=event["tkn_idx"],
                aspect=event.get("aspect"),
                mood=event.get("mood"),
                tense=event.get("tense"),
                polarity=event.get("polarity"),
                endpoints=event.get("endpoints"),
                text=event.get("text")
            ))

        # timexs
        for timex in self.content["entities"]["timexs"]:

            # find entity endpoints
            if timex["sent_idx"] != 0 and timex["tkn_idx"] != [0]:  # ignore dct as it is not explicit in raw text
                endpoints = []
                for idx in timex["tkn_idx"]:
                    key = f"s{timex['sent_idx']}t{idx}"
                    token = token_dict[key]
                    endpoints += [*token["endpoints"]]
                s, e = endpoints[0], endpoints[-1]

                timex["endpoints"] = [s, e]
                timex["text"] = self._text[s: e]

            entities.add(Timex(
                id=timex["id"],
                sent_idx=timex["sent_idx"],
                tkn_idx=timex["tkn_idx"],
                type=timex["id"],
                value=timex.get("val"),
                function_in_document=timex.get("function_in_document"),
                endpoints=timex.get("endpoints"),
                text=timex.get("text")
            ))

        return entities

    @property
    def _tlinks(self) -> Iterable[TLink]:

        tlinks = set()

        entities_dict = {ent.id: ent for ent in self._entities}
        for tlink in self.content["tlinks"]:

            source = entities_dict[tlink["from"]]
            target = entities_dict[tlink["to"]]

            tlinks.add(TLink(
                source=source,
                target=target,
                relation=tlink["relation"]
            ))

        return tlinks

    def _get_endpoints(self, entity):

        endpoints = []
        for token in self.content["tokens"]:
            cond1 = token["sent_idx"] == entity["sent_idx"]
            cond2 = token["tkn_idx"] in entity["tkn_idx"]
            if cond1 and cond2:
                endpoints += [*token["endpoints"]]

        return endpoints[0], endpoints[-1]


class TempEval2FrenchDocumentReader(BaseDocumentReader):

    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)

        self.path = path
        self.content = xml2dict(self.path)

        self.xml = ET.parse(self.path)

    @property
    def _name(self) -> str:
        return self.path.name.replace(".xml", "")

    @property
    def _dct(self) -> Timex:

        timexs = self.content["TimeML"]["TEXT"]["TIMEX3"]
        if not isinstance(timexs, list):
            timexs = [timexs]

        for timex in timexs:
            if timex.get("functionInDocument") == "CREATION_TIME":
                return Timex(
                    id=timex["tid"],
                    function_in_document=timex["functionInDocument"],
                    value=timex["value"],
                    type=timex["type"],
                )

    @property
    def _text(self) -> str:

        root = self.xml.find("TEXT")
        return "".join(element for element in root.itertext())

    @property
    def _entities(self) -> Iterable[Entity]:

        entities = set()

        # events
        events = self.content["TimeML"]["TEXT"]["EVENT"]
        events_dict = {event["eid"]: event for event in events}
        mkinsts = self.content["TimeML"].get("MAKEINSTANCE")
        if mkinsts:
            for mkinst in mkinsts:

                # add makeintance information to events
                event = events_dict.get(mkinst["eid"])
                if event:
                    mkinst.update(event)

                entities.add(Event(
                    id=mkinst['eiid'],
                    text=mkinst['text'],
                    aspect=mkinst.get('aspect'),
                    class_=mkinst.get('class'),
                    modality=mkinst.get('modality'),
                    polarity=mkinst.get('polarity'),
                    cardinality=mkinst.get('cardinality'),
                    signal_id=mkinst.get('signalID'),
                    pos=mkinst.get('pos'),
                    tense=mkinst.get('tense'),
                ))

        else:

            for event in events:

                entities.add(Event(
                    id=event['eid'],
                    text=event['text'],
                    aspect=event.get('aspect'),
                    class_=event.get('class'),
                    modality=event.get('modality'),
                    polarity=event.get('polarity'),
                    cardinality=event.get('cardinality'),
                    signal_id=event.get('signalID'),
                    pos=event.get('pos'),
                    tense=event.get('tense'),
                ))

        timexs = self.content["TimeML"]["TEXT"]["TIMEX3"]
        if not isinstance(timexs, list):
            timexs = [timexs]

        for timex in timexs:
            entities.add(Timex(
                id=timex["tid"],
                text=timex.get("text"),
                value=timex.get("value"),
                type=timex.get("type"),
                function_in_document=timex.get("functionInDocument"),
                anchor_time_id=timex.get("anchorTimeID"),
                temporal_function=timex.get("temporalFunction"),
                value_from_function=timex.get("valueFromFunction"),
            ))

        return entities
    
    @property
    def _tlinks(self) -> Iterable[TLink]:

        result = set()

        entities_dict = {ent.id: ent for ent in self._entities}

        tlinks = self.content["TimeML"].get("TLINK")
        if tlinks:
            for tlink in tlinks:
                result.add(TLink(
                    id=tlink["lid"],
                    source=entities_dict[tlink["from"]],
                    target=entities_dict[tlink["to"]],
                    relation=tlink["relType"]
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
            print(file)
            reader = self.document_reader(file)
            document = reader.read()

            if "test" in file.parts:
                test += [document]

            else:
                train += [document]

        return Dataset(path.name, train, test)


class JSONDatasetReader:

    def __init__(self, doc_reader) -> None:
        self.document_reader = doc_reader

    def read(self, path) -> Dataset:

        path = Path(path)
        if not path.is_dir():
            raise IOError(f"The dataset being load have not been downloaded yet.")

        train, test = [], []
        files = list(path.glob("**/*.json"))
        for file in tqdm(files):
            print(file)
            reader = self.document_reader(file)
            document = reader.read()

            if "test" in file.parts:
                test += [document]

            else:
                train += [document]

        return Dataset(path.name, train, test)
