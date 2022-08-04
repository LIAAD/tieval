import pathlib
from dataclasses import dataclass
from typing import Union, Tuple, Iterable

from tieval.datasets.readers import (
    XMLDatasetReader,
    JSONDatasetReader,
    EventTimeDatasetReader,
    MATRESDatasetReader,
    TDDiscourseDatasetReader,
    MeanTimeDocumentReader,
    TimeBankDenseDatasetReader,
    TimeBankPTDocumentReader,
    KRAUTSDocumentReader
)
from tieval.datasets.readers import (
    TempEval3DocumentReader,
    GraphEveDocumentReader,
    TempEval2DocumentReader,
    TempEval2FrenchDocumentReader,
    TimeBank12DocumentReader,
    TCRDocumentReader
)


DatasetReaders = Union[
    XMLDatasetReader,
    JSONDatasetReader,
    EventTimeDatasetReader,
    MATRESDatasetReader,
    TDDiscourseDatasetReader,
    MeanTimeDocumentReader,
    TimeBankDenseDatasetReader
]

DocumentReaders = Union[
    TempEval3DocumentReader,
    GraphEveDocumentReader,
    TempEval2DocumentReader,
    TempEval2FrenchDocumentReader,
    KRAUTSDocumentReader
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
    language: str = None
    reference: str = None
    base: Iterable[str] = None
    extension: str = None
    columns: Tuple = None
    event_index: str = None
    files: [str] = None
    data_path: str = "data"

    @property
    def path(self):
        return pathlib.Path(f"{self.data_path}/{self.name.lower()}")

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
        language="english",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader,
        url="https://drive.inesctec.pt/s/fxRfPfLcKKJ74Dn/download",
    ),

    "eventtime": DatasetMetadata(
        name="EventTime",
        language="english",
        url="https://drive.inesctec.pt/s/wbGtJceye6ntkiS/download",
        reader=EventTimeDatasetReader,
        base=["timebank"]
    ),

    "grapheve": DatasetMetadata(
        name="GraphEve",
        language="english",
        url="https://drive.inesctec.pt/s/eKSHKB6gMozCP4Q/download",
        reader=XMLDatasetReader,
        doc_reader=GraphEveDocumentReader
    ),

    "krauts": DatasetMetadata(
        name="KRAUTS",
        language="german",
        url="https://drive.inesctec.pt/s/cFNiJ2CLnBFYr5a/download",
        reader=XMLDatasetReader,
        doc_reader=KRAUTSDocumentReader
    ),

    "matres": DatasetMetadata(
        name="matres",
        language="english",
        url="https://drive.inesctec.pt/s/7g68GBTECiD2XYK/download",
        base=["tempeval_3"],
        repo="https://github.com/qiangning/MATRES",
        reader=MATRESDatasetReader,
    ),

    "meantime_english": DatasetMetadata(
        name="meantime_english",
        language="english",
        url="https://drive.inesctec.pt/s/QoLwqgdTLkfia7L/download",
        reader=XMLDatasetReader,
        doc_reader=MeanTimeDocumentReader
    ),

    "meantime_spanish": DatasetMetadata(
        name="meantime_spanish",
        language="spanish",
        url="https://drive.inesctec.pt/s/xEPGnqygP6FEFkP/download",
        reader=XMLDatasetReader,
        doc_reader=MeanTimeDocumentReader
    ),

    "meantime_dutch": DatasetMetadata(
        name="meantime_dutch",
        language="dutch",
        url="https://drive.inesctec.pt/s/HyFfxEwryj6Fffq/download",
        reader=XMLDatasetReader,
        doc_reader=MeanTimeDocumentReader
    ),

    "meantime_italian": DatasetMetadata(
        name="meantime_italian",
        language="italian",
        url="https://drive.inesctec.pt/s/kabifjEcQboKbBA/download",
        reader=XMLDatasetReader,
        doc_reader=MeanTimeDocumentReader
    ),

    "platinum": DatasetMetadata(
        name="Platinum",
        language="english",
        url="https://drive.inesctec.pt/s/ppCdTWijAYFbRiL/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader,
    ),

    "tcr": DatasetMetadata(
        name="TCR",
        language="english",
        url="https://drive.inesctec.pt/s/mSGaNyYSiMRTfGH/download",
        reader=XMLDatasetReader,
        doc_reader=TCRDocumentReader,
    ),

    "tddiscourse": DatasetMetadata(
        name="TDDiscourse",
        language="english",
        url="https://drive.inesctec.pt/s/9nXDNqt3Sa8bkDk/download",
        base=["timebank_1.2"],
        reader=TDDiscourseDatasetReader,
    ),

    "tempeval_2_chinese": DatasetMetadata(
        name="tempeval_2_chinese",
        language="chinese",
        url="https://drive.inesctec.pt/s/s4HMWnntet8z2bS/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),


    "tempeval_2_english": DatasetMetadata(
        name="tempeval_2_english",
        language="english",
        url="https://drive.inesctec.pt/s/Z2q5oEYf4cAM2ji/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),


    "tempeval_2_french": DatasetMetadata(
        name="tempeval_2_french",
        language="french",
        url="https://drive.inesctec.pt/s/mNoo2YFWG4X8tD4/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval2FrenchDocumentReader,
    ),


    "tempeval_2_italian": DatasetMetadata(
        name="tempeval_2_italian",
        language="italian",
        url="https://drive.inesctec.pt/s/PkADwaWEogapSWW/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),


    "tempeval_2_korean": DatasetMetadata(
        name="tempeval_2_korean",
        language="korean",
        url="https://drive.inesctec.pt/s/RwMLseDt4GnnfKr/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),


    "tempeval_2_spanish": DatasetMetadata(
        name="tempeval_2_spanish",
        language="spanish",
        url="https://drive.inesctec.pt/s/H7otpwJCFCsjM9r/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),

    "tempeval_3": DatasetMetadata(
        name="tempeval_3",
        language="english",
        url="https://drive.inesctec.pt/s/ebp27ZjfCgDTxwG/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader,
    ),

    "timebank_1.2": DatasetMetadata(
        name="TimeBank_1.2",
        language="english",
        url="https://drive.inesctec.pt/s/QHiBgZmi45B72AB/download",
        reader=XMLDatasetReader,
        doc_reader=TimeBank12DocumentReader,

    ),

    "timebank_dense": DatasetMetadata(
        name="TimeBank_Dense",
        language="english",
        url="https://drive.inesctec.pt/s/dtztXXBpPPXyzLX/download",
        base=["timebank_1.2"],
        reader=TimeBankDenseDatasetReader,

    ),

    "timebankpt": DatasetMetadata(
        name="TimeBankPT",
        language="portuguese",
        url="https://drive.inesctec.pt/s/jCcpQGXzLdnL9Tx/download",
        reader=XMLDatasetReader,
        doc_reader=TimeBankPTDocumentReader,
    ),

    "timebank": DatasetMetadata(
        name="TimeBank",
        language="english",
        url="https://drive.inesctec.pt/s/KmeTs6LqnmzRr2s/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader,
    ),

    "traint3": DatasetMetadata(
        name="Traint3",
        language="english",
        url="https://drive.inesctec.pt/s/SaPzJxD2b9PzxY4/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader
    ),
}
