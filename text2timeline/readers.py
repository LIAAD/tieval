
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

    def _get_events_from_tags(self, xml: XMLHandler) -> List[Event]:

        event_tags = xml.get_tag("EVENT")

        # complementary information of each event is given on MAKEINSTANCE tags
        minst_tags = xml.get_tag("MAKEINSTANCE")
        minst_attrib = {mit.attrib['eventID']: mit.attrib for mit in minst_tags}

        events = []
        for et in event_tags:

            attrib = et.attrib
            event_id = attrib['eid']

            attrib['text'] = ''.join(list(et.itertext()))
            # attrib['endpoints'] = self._expression_idxs[event_id]

            # add MAKEINSTANCE info
            attrib.update(minst_attrib.get(event_id))
            events += [Event(attrib)]

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

    def read(self, path: Union[str, Path]) -> Document:

        if not isinstance(path, Path):
            path = Path(path)

        tml = XMLHandler(path)

        name = path.name.replace('.tml', '')
        text = tml.text

        events = self._get_events_from_tags(tml)
        tags = tml.get_tag("EVENT"), tml.get_tag("TIMEX3"), tml.get_tag("TLINK")
        event_tags, timex_tags, tlink_tags = tags

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
