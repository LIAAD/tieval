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

from tieval.datasets.readers import \
    XMLDatasetReader, \
    JSONDatasetReader, \
    EventTimeDatasetReader, \
    MATRESDatasetReader, \
    UDSTDatasetReader, \
    TDDiscourseDatasetReader, \
    MCTacoDatasetReader, \
    MeanTimeDocumentReader, \
    TimeBankDenseDatasetReader


from tieval.datasets.readers import \
    TempEval3DocumentReader, \
    GraphEveDocumentReader, \
    TempEval2DocumentReader, \
    TempEval2FrenchDocumentReader, \
    TimeBank12DocumentReader

from tieval import DATA_PATH


DatasetReaders = Union[
    XMLDatasetReader,
    JSONDatasetReader,
    EventTimeDatasetReader,
    MATRESDatasetReader,
    UDSTDatasetReader,
    TDDiscourseDatasetReader,
    MCTacoDatasetReader,
    MeanTimeDocumentReader,
    TimeBankDenseDatasetReader
]

DocumentReaders = Union[
    TempEval3DocumentReader,
    GraphEveDocumentReader,
    TempEval2DocumentReader,
    TempEval2FrenchDocumentReader
]

TMLDocumentReader = None


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
        doc_reader=TempEval3DocumentReader,
        url="https://drive.inesctec.pt/s/fxRfPfLcKKJ74Dn/download",
    ),

    "eventtime": DatasetMetadata(
        name="EventTime",
        url="https://drive.inesctec.pt/s/wbGtJceye6ntkiS/download",
        reader=EventTimeDatasetReader,
        base=["timebank"]
    ),

    "grapheve": DatasetMetadata(
        name="GraphEve",
        url="https://drive.inesctec.pt/s/eKSHKB6gMozCP4Q/download",
        reader=XMLDatasetReader,
        doc_reader=GraphEveDocumentReader
    ),

    "matres": DatasetMetadata(
        name="matres",
        url="https://drive.inesctec.pt/s/7g68GBTECiD2XYK/download",
        base=["tempeval_3"],
        repo="https://github.com/qiangning/MATRES",
        reader=MATRESDatasetReader,
    ),

    "mctaco": DatasetMetadata(
        name="MCTaco",
        url="https://drive.inesctec.pt/s/q54BizkCwK9egEL/download",
        reader=MCTacoDatasetReader
    ),

    "meantime_english": DatasetMetadata(
        name="meantime_english",
        url="https://drive.inesctec.pt/s/QoLwqgdTLkfia7L/download",
        reader=XMLDatasetReader,
        doc_reader=MeanTimeDocumentReader
    ),

    "meantime_spanish": DatasetMetadata(
        name="meantime_spanish",
        url="https://drive.inesctec.pt/s/xEPGnqygP6FEFkP/download",
        reader=XMLDatasetReader,
        doc_reader=MeanTimeDocumentReader
    ),

    "meantime_dutch": DatasetMetadata(
        name="meantime_dutch",
        url="https://drive.inesctec.pt/s/HyFfxEwryj6Fffq/download",
        reader=XMLDatasetReader,
        doc_reader=MeanTimeDocumentReader
    ),

    "meantime_italian": DatasetMetadata(
        name="meantime_italian",
        url="https://drive.inesctec.pt/s/kabifjEcQboKbBA/download",
        reader=XMLDatasetReader,
        doc_reader=MeanTimeDocumentReader
    ),

    "platinum": DatasetMetadata(
        name="Platinum",
        url="https://drive.inesctec.pt/s/ppCdTWijAYFbRiL/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader,
    ),

    "tcr": DatasetMetadata(
        name="TCR",
        url="https://drive.inesctec.pt/s/mSGaNyYSiMRTfGH/download",
        reader=XMLDatasetReader,
        doc_reader=TMLDocumentReader,
    ),

    "tddiscourse": DatasetMetadata(
        name="TDDiscourse",
        url="https://drive.inesctec.pt/s/9nXDNqt3Sa8bkDk/download",
        base=["timebank_1.2"],
        columns=("doc", "src", "tgt", "relation"),
        event_index="eid",
        reader=TDDiscourseDatasetReader,
    ),

    "tempeval_2_chinese": DatasetMetadata(
        name="tempeval_2_chinese",
        url="https://drive.inesctec.pt/s/s4HMWnntet8z2bS/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),


    "tempeval_2_english": DatasetMetadata(
        name="tempeval_2_english",
        url="https://drive.inesctec.pt/s/Z2q5oEYf4cAM2ji/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),


    "tempeval_2_french": DatasetMetadata(
        name="tempeval_2_french",
        url="https://drive.inesctec.pt/s/mNoo2YFWG4X8tD4/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval2FrenchDocumentReader,
    ),


    "tempeval_2_italian": DatasetMetadata(
        name="tempeval_2_italian",
        url="https://drive.inesctec.pt/s/PkADwaWEogapSWW/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),


    "tempeval_2_korean": DatasetMetadata(
        name="tempeval_2_korean",
        url="https://drive.inesctec.pt/s/RwMLseDt4GnnfKr/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),


    "tempeval_2_spanish": DatasetMetadata(
        name="tempeval_2_spanish",
        url="https://drive.inesctec.pt/s/H7otpwJCFCsjM9r/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),

    "tempeval_3": DatasetMetadata(
        name="tempeval_3",
        url="https://drive.inesctec.pt/s/ebp27ZjfCgDTxwG/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader,
    ),

    "tempqa": DatasetMetadata(
        name="TempQA",
        url="https://drive.inesctec.pt/s/PQJD7KfEgKTYMXR/download"
    ),

    "tempquestions": DatasetMetadata(
        name="TempQuestions",
        url="https://drive.inesctec.pt/s/RCYzmwQapHJMaa4/download"
    ),

    "timebank_1.2": DatasetMetadata(
        name="TimeBank_1.2",
        url="https://drive.inesctec.pt/s/QHiBgZmi45B72AB/download",
        reader=XMLDatasetReader,
        doc_reader=TimeBank12DocumentReader,

    ),

    "timebank_dense": DatasetMetadata(
        name="TimeBank_Dense",
        url="https://drive.inesctec.pt/s/dtztXXBpPPXyzLX/download",
        base=["timebank_1.2"],
        columns=("doc", "src", "tgt", "relation"),
        event_index="eid",
        reader=TimeBankDenseDatasetReader,

    ),

    "timebankpt": DatasetMetadata(
        name="TimeBankPT",
        url="https://drive.inesctec.pt/s/jCcpQGXzLdnL9Tx/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader,
    ),

    "timebank": DatasetMetadata(
        name="TimeBank",
        url="https://drive.inesctec.pt/s/KmeTs6LqnmzRr2s/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader,
    ),

    "torque": DatasetMetadata(
        name="TORQUE",
        url="https://drive.inesctec.pt/s/EfJ2YeB7wQKxdjM/download",
    ),

    "traint3": DatasetMetadata(
        name="Traint3",
        url="https://drive.inesctec.pt/s/SaPzJxD2b9PzxY4/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader
    ),

    "uds_t": DatasetMetadata(
        name="UDS_T",
        url="https://drive.inesctec.pt/s/JLRoMczLXcgpYKg/download",
        reader=UDSTDatasetReader
    )
}
