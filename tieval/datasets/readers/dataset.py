import collections
from pathlib import Path
from typing import Union

from tqdm import tqdm

from tieval.base import Dataset
from tieval.datasets.readers.document import DocumentReaders
from tieval.links import TLink

DocName = str


class XMLDatasetReader:
    """Handles the process of reading any temporally annotated dataset stored with .tml or .xml extension."""

    def __init__(self, doc_reader: DocumentReaders) -> None:
        self.document_reader = doc_reader

    def read(self, path: Union[str, Path]) -> Dataset:

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


class JSONDatasetReader:

    def __init__(self, doc_reader: DocumentReaders) -> None:
        self.document_reader = doc_reader

    def read(self, path) -> Dataset:

        path = Path(path)
        if not path.is_dir():
            raise IOError(f"The dataset being load have not been downloaded yet.")

        train, test = [], []
        files = list(path.glob("**/*.json"))
        for file in tqdm(files):
            reader = self.document_reader(file)
            document = reader.read()

            if "test" in file.parts:
                test += [document]

            else:
                train += [document]

        return Dataset(path.name, train, test)


class EventTimeDatasetReader:

    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def read(self, path: Union[str, Path]) -> Dataset:

        path = Path(path)

        events_table_path = path / "train" / "event-times_normalized.tab"
        with open(events_table_path, 'r') as fin:
            docs = collections.defaultdict(list)
            for line in fin.readlines():
                doc, sent_idx, tkn_idx, entity_type, id_, _, type_, value = line.split()
                docs[doc] += [{
                    "id": id_,
                    "sent_idx": sent_idx,
                    "tkn_idx": tkn_idx,
                    "type": type_,
                    "value": value,
                }]

        documents = []
        for doc_name, events in docs.items():
            document = self.base_dataset[doc_name]

            update_events = []
            document.tlinks = set()
            original_events_dict = {event.eid: event for event in document.events}
            for event in events:
                original_event = original_events_dict.get(event["id"])
                original_event.value = event["value"]
                original_event.type = event["type"]
                update_events += [original_event]
            document.entities = set(update_events)

            documents += [document]

        return Dataset(path.name, train=documents)


class MATRESDatasetReader:

    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def read(self, path: Union[str, Path]) -> Dataset:

        path = Path(path)

        docs = {
            "train": collections.defaultdict(list),
            "test": collections.defaultdict(list)
        }
        for filepath in path.glob("**/*.txt"):

            split = filepath.parts[-2]
            with open(filepath, 'r') as fin:

                for line in fin.readlines():
                    doc, src_token, tgt_token, src_id, tgt_id, relation = line.split()
                    src_id = "ei" + src_id
                    tgt_id = "ei" + tgt_id

                    document = self.base_dataset[doc]
                    if document is None:
                        continue
                    entities_dict = {ent.id: ent for ent in document.entities}

                    src = entities_dict[src_id]
                    tgt = entities_dict[tgt_id]

                    docs[split][doc] += [TLink(
                        source=src,
                        target=tgt,
                        relation=relation
                    )]

        documents = {
            "train": [],
            "test": []
        }
        for split in docs:
            for doc_name, tlinks in docs[split].items():

                document = self.base_dataset[doc_name]

                entities = []
                for tlink in tlinks:
                    entities += [tlink.source, tlink.target]

                document.tlinks = tlinks
                document.entities = set(entities)

                documents[split] += [document]

        return Dataset(path.name, train=documents["train"], test=documents["test"])


class TDDiscourseDatasetReader:

    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def read(self, path: Union[str, Path]) -> Dataset:

        path = Path(path)

        docs = {
            "train": collections.defaultdict(list),
            "test": collections.defaultdict(list)
        }
        for filepath in path.glob("**/*.tsv"):

            split = filepath.parts[-2]
            with open(filepath, 'r') as fin:

                for line in fin.readlines():
                    doc, src_id, tgt_id, relation = line.split()

                    document = self.base_dataset[doc]
                    entities_dict = {ent.eid: ent for ent in document.events}

                    src = entities_dict[src_id]
                    tgt = entities_dict[tgt_id]

                    docs[split][doc] += [TLink(
                        source=src,
                        target=tgt,
                        relation=relation
                    )]

        documents = {
            "train": [],
            "test": []
        }
        for split in docs:
            for doc_name, tlinks in docs[split].items():

                document = self.base_dataset[doc_name]

                entities = []
                for tlink in tlinks:
                    entities += [tlink.source, tlink.target]

                document.tlinks = tlinks
                document.entities = set(entities)

                documents[split] += [document]

        return Dataset(path.name, train=documents["train"], test=documents["test"])


class TimeBankDenseDatasetReader:

    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def read(self, path: Union[str, Path]) -> Dataset:

        path = Path(path)

        docs = {
            "train": collections.defaultdict(list),
            "test": collections.defaultdict(list)
        }
        for filepath in path.glob("**/*.txt"):
            print(filepath)

            split = filepath.parts[-2]
            with open(filepath, 'r') as fin:

                for line in fin.readlines():

                    doc, src_id, tgt_id, relation = line.split()

                    document = self.base_dataset[doc]
                    entities_dict = {"t0": document.dct}
                    entities_dict.update({timex.id: timex for timex in document.timexs})
                    entities_dict.update({event.eid: event for event in document.events})

                    src = entities_dict[src_id]
                    tgt = entities_dict[tgt_id]

                    docs[split][doc] += [TLink(
                        source=src,
                        target=tgt,
                        relation=relation
                    )]

        documents = {
            "train": [],
            "test": []
        }
        for split in docs:
            for doc_name, tlinks in docs[split].items():

                document = self.base_dataset[doc_name]

                entities = []
                for tlink in tlinks:
                    entities += [tlink.source, tlink.target]

                document.tlinks = tlinks
                document.entities = set(entities)

                documents[split] += [document]

        return Dataset(path.name, train=documents["train"], test=documents["test"])


DatasetReaders = Union[
    XMLDatasetReader,
    JSONDatasetReader,
    EventTimeDatasetReader,
    MATRESDatasetReader,
    TDDiscourseDatasetReader,
    TimeBankDenseDatasetReader
]
