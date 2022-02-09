""" Store metadata about datasets.

Objects
-------
    - DatasetMetadata

Constants
---------
    - DATASETS_METADATA

"""

from dataclasses import dataclass
import pathlib
from typing import Union, Tuple, Iterable

from text2timeline.datasets.readers import XMLDatasetReader
from text2timeline.datasets.readers import TableDatasetReader

from text2timeline.datasets.readers import MeanTimeDocumentReader
from text2timeline.datasets.readers import TMLDocumentReader

from text2timeline import DATA_PATH


DatasetReaders = Union[XMLDatasetReader, TableDatasetReader]
DocumentReaders = Union[TMLDocumentReader, MeanTimeDocumentReader]


@dataclass
class DatasetMetadata:
    """ Store dataset metadata."""

    name: str
    reader: DatasetReaders = None
    doc_reader: DocumentReaders = None
    url: str = None
    repo: str = None
    reference: str = None
    base: Iterable[str] = None
    extension: str = None
    columns: Tuple = None
    event_index: str = None
    files: [str] = None

    def __post_init__(self):
        self.path = pathlib.Path(f"{DATA_PATH}/{self.name.lower()}")

    @property
    def description(self):

        readme_path = self.path / "README.md"

        with open(readme_path) as f:
            readme = f.read()

        return readme


# datasets that only have a table with the temporal links
DATASETS_METADATA = {
    "aquaint": DatasetMetadata(
        name="AQUAINT",
        reader=XMLDatasetReader,
        doc_reader=TMLDocumentReader,
        url="https://drive.inesctec.pt/s/sW8Acr9wxGzmn74/download",
    ),

    "eventtime": DatasetMetadata(
        name="EventTime",
        url="https://drive.inesctec.pt/s/TT6M7TXTbo93kKb/download",
        reader=TableDatasetReader,
        files=[
            "event-times_normalized.tab"
        ],
        base=("timebank",),
        columns=("doc", "sentence_number", "token_number", "tag_name", "tag_id", "instance_id", "attribute_name", "attribute_value"),
    ),

    "grapheve": DatasetMetadata(
        name="GraphEve",
        url="https://drive.inesctec.pt/s/2oYDkRTwTSpyY4N/download"
    ),

    "matres": DatasetMetadata(
        name="MATRES",
        url="https://drive.inesctec.pt/s/bNkyKYLog9TLTfp/download",
        base=("tempeval-3",),
        files=[
          "aquaint.txt", "platinum.txt", "timebank,txt"
        ],
        columns=("doc", "src_token", "tgt_token", "src", "tgt", "relation"),
        event_index="eiid",
        repo="https://github.com/qiangning/MATRES",
        reader=TableDatasetReader,
    ),

    "mctaco": DatasetMetadata(
        name="MCTaco",
        url="https://drive.inesctec.pt/s/dNkSnFKsyjjYRCQ/download"
    ),

    "meantime": DatasetMetadata(
        name="MeanTime",
        url="https://drive.inesctec.pt/s/YskTKn5H7Bj7Dtt/download",
        reader=XMLDatasetReader,
        doc_reader=MeanTimeDocumentReader
    ),

    "platinum": DatasetMetadata(
        name="Platinum",
        url="https://drive.inesctec.pt/s/X7PdbFF466ffMsC/download",
        reader=XMLDatasetReader,
        doc_reader=TMLDocumentReader,
    ),

    "tcr": DatasetMetadata(
        name="TCR",
        url="https://drive.inesctec.pt/s/H9xi7mRpEkYL6ws/download",
        reader=XMLDatasetReader,
        doc_reader=TMLDocumentReader,
    ),

    "tddiscourse": DatasetMetadata(
        name="TDDiscourse",
        url="https://drive.inesctec.pt/s/reaoG7LJD4sjdaN/download",
        base={"timebank-1.2"},
        columns=("doc", "src", "tgt", "relation"),
        event_index="eid",
        reader=TableDatasetReader,
    ),

    "tempeval-2": DatasetMetadata(
        name="TempEval-2",
        url="https://drive.inesctec.pt/s/ZD6wzmrgysr8mne/download"
    ),

    "tempeval-3": DatasetMetadata(
        name="TempEval-3",
        url="https://drive.inesctec.pt/s/K457eTAo656gqMw/download",
        reader=XMLDatasetReader,
        doc_reader=TMLDocumentReader,
    ),

    "tempqa": DatasetMetadata(
        name="TempQA",
        url="https://drive.inesctec.pt/s/q6jntQeoa2EZntD/download"
    ),

    "tempquestions": DatasetMetadata(
        name="TempQuestions",
        url="https://drive.inesctec.pt/s/KgdXNKz35DW2opN/download"
    ),

    "timebank-1.2": DatasetMetadata(
        name="TimeBank-1.2",
        url="https://drive.inesctec.pt/s/2yTsSCRsmJ9nGKH/download",
        reader=XMLDatasetReader,
        doc_reader=TMLDocumentReader,

    ),

    "timebank-dense": DatasetMetadata(
        name="TimeBank-Dense",
        url="https://drive.inesctec.pt/s/8AK3LYJpayn39px/download",
        base={"timebank-1.2"},
        columns=("doc", "src", "tgt", "relation"),
        event_index="eid",
        reader=TableDatasetReader,

    ),

    "timebankpt": DatasetMetadata(
        name="TimeBankPT",
        url="https://drive.inesctec.pt/s/kxk9yMHAiK4XdKS/download",
        reader=XMLDatasetReader,
        doc_reader=TMLDocumentReader,
    ),

    "timebank": DatasetMetadata(
        name="TimeBank",
        url="https://drive.inesctec.pt/s/i57gZQjZs5XiDpZ/download",
        reader=XMLDatasetReader,
        doc_reader=TMLDocumentReader,
    ),

    "torque": DatasetMetadata(
        name="TORQUE",
        url="https://drive.inesctec.pt/s/yLLiJtTW99XLpZE/download"
    ),

    "traint3": DatasetMetadata(
        name="Traint3",
        url="https://drive.inesctec.pt/s/4mF6NmZa47zyX2R/download",
        reader=XMLDatasetReader,
        doc_reader=TMLDocumentReader
    ),

    "uds-t": DatasetMetadata(
        name="UDS-T",
        url="https://drive.inesctec.pt/s/db5dNCCyG3szWej/download"
    )
}
