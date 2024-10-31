from dataclasses import dataclass
from typing import Tuple, Iterable, Union

from tieval.datasets.readers import (
    ProfessorHeidelTimeDocumentReader,
    TempEval3DocumentReader,
    GraphEveDocumentReader,
    TempEval2DocumentReader,
    TempEval2FrenchDocumentReader,
    TimeBank12DocumentReader,
    TCRDocumentReader,
    DatasetReaders,
    AncientTimeDocumentReader,
    XMLDatasetReader,
    JSONDatasetReader,
    EventTimeDatasetReader,
    MATRESDatasetReader,
    TDDiscourseDatasetReader,
    MeanTimeDocumentReader,
    TimeBankDenseDatasetReader,
    TimeBankPTDocumentReader,
    KRAUTSDocumentReader,
    NarrativeContainerDocumentReader,
    WikiWarsDocumentReader,
    FRTimeBankDocumentReader,
    DocumentReaders
)


@dataclass
class DatasetMetadata:
    """ Store dataset metadata."""

    name: str
    reader: Union[DatasetReaders] = None
    doc_reader: Union[DocumentReaders] = None
    url: str = None
    repo: str = None
    language: str = None
    reference: str = None
    base: Iterable[str] = None
    extension: str = None
    columns: Tuple = None
    event_index: str = None
    files: [str] = None


# datasets that only have a table with the temporal links
DATASETS_METADATA = {
    "ancient_time_arabic": DatasetMetadata(
        name="ancient_time_arabic",
        language="arabic",
        reader=XMLDatasetReader,
        doc_reader=AncientTimeDocumentReader,
        url="https://drive.inesctec.pt/s/LTZbYe6jFWcZXyj/download",
    ),

    "ancient_time_dutch": DatasetMetadata(
        name="ancient_time_dutch",
        language="dutch",
        reader=XMLDatasetReader,
        doc_reader=AncientTimeDocumentReader,
        url="https://drive.inesctec.pt/s/SR5CWgjfgJaY22B/download",
    ),

    "ancient_time_english": DatasetMetadata(
        name="ancient_time_english",
        language="english",
        reader=XMLDatasetReader,
        doc_reader=AncientTimeDocumentReader,
        url="https://drive.inesctec.pt/s/FjHtY3wBYPwciSL/download",
    ),

    "ancient_time_french": DatasetMetadata(
        name="ancient_time_french",
        language="french",
        reader=XMLDatasetReader,
        doc_reader=AncientTimeDocumentReader,
        url="https://drive.inesctec.pt/s/ggpACrLRzwSz5ks/download",
    ),

    "ancient_time_german": DatasetMetadata(
        name="ancient_time_german",
        language="german",
        reader=XMLDatasetReader,
        doc_reader=AncientTimeDocumentReader,
        url="https://drive.inesctec.pt/s/r6Z6KwwKgjmLNJR/download",
    ),

    "ancient_time_italian": DatasetMetadata(
        name="ancient_time_italian",
        language="italian",
        reader=XMLDatasetReader,
        doc_reader=AncientTimeDocumentReader,
        url="https://drive.inesctec.pt/s/JbcHfaDqQgiW59T/download",
    ),

    "ancient_time_spanish": DatasetMetadata(
        name="ancient_time_spanish",
        language="spanish",
        reader=XMLDatasetReader,
        doc_reader=AncientTimeDocumentReader,
        url="https://drive.inesctec.pt/s/C3bcfp8g7C23fPg/download",
    ),

    "ancient_time_vietnamese": DatasetMetadata(
        name="ancient_time_vietnamese",
        language="vietnamese",
        reader=XMLDatasetReader,
        doc_reader=AncientTimeDocumentReader,
        url="https://drive.inesctec.pt/s/LyMLHoRQbL9zndt/download",
    ),

    "aquaint": DatasetMetadata(
        name="AQUAINT",
        language="english",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader,
        url="https://drive.inesctec.pt/s/EsCsL6bayxsg5Co/download",
    ),

    "eventtime": DatasetMetadata(
        name="EventTime",
        language="english",
        url="https://drive.inesctec.pt/s/8HyHGnLE7QbWRCF/download",
        reader=EventTimeDatasetReader,
        base=["timebank"]
    ),

    "fr_timebank": DatasetMetadata(
        name="FR_Timebank",
        language="french",
        url="https://drive.inesctec.pt/s/crZ8cTrZRobKsLK/download",
        reader=XMLDatasetReader,
        doc_reader=FRTimeBankDocumentReader,
    ),

    "grapheve": DatasetMetadata(
        name="GraphEve",
        language="english",
        url="https://drive.inesctec.pt/s/ctPKsYbt5terqJS/download",
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

    "krauts_diezeit": DatasetMetadata(
        name="krauts_diezeit",
        language="german",
        url="https://drive.inesctec.pt/s/f98ZDGLqEjqCXEp/download",
        reader=XMLDatasetReader,
        doc_reader=KRAUTSDocumentReader
    ),

    "krauts_dolomiten_42": DatasetMetadata(
        name="krauts_dolomiten_42",
        language="german",
        url="https://drive.inesctec.pt/s/AEwAod8Wn2L9E9G/download",
        reader=XMLDatasetReader,
        doc_reader=KRAUTSDocumentReader
    ),

    "krauts_dolomiten_100": DatasetMetadata(
        name="krauts_dolomiten_100",
        language="german",
        url="https://drive.inesctec.pt/s/x3N7Kf446RFeDse/download",
        reader=XMLDatasetReader,
        doc_reader=KRAUTSDocumentReader
    ),

    "matres": DatasetMetadata(
        name="matres",
        language="english",
        url="https://drive.inesctec.pt/s/9kpP6oy8y4jZPyN/download",
        base=["tempeval_3"],
        repo="https://github.com/qiangning/MATRES",
        reader=MATRESDatasetReader,
    ),

    "meantime_english": DatasetMetadata(
        name="meantime_english",
        language="english",
        url="https://drive.inesctec.pt/s/jbHfZYpb3Y3Cs5T/download",
        reader=XMLDatasetReader,
        doc_reader=MeanTimeDocumentReader
    ),

    "meantime_spanish": DatasetMetadata(
        name="meantime_spanish",
        language="spanish",
        url="https://drive.inesctec.pt/s/HE7qxH6BP6nPEkg/download",
        reader=XMLDatasetReader,
        doc_reader=MeanTimeDocumentReader
    ),

    "meantime_dutch": DatasetMetadata(
        name="meantime_dutch",
        language="dutch",
        url="https://drive.inesctec.pt/s/pQA3RcKiko2GcMX/download",
        reader=XMLDatasetReader,
        doc_reader=MeanTimeDocumentReader
    ),

    "meantime_italian": DatasetMetadata(
        name="meantime_italian",
        language="italian",
        url="https://drive.inesctec.pt/s/m4jckQLrYSxmTEq/download",
        reader=XMLDatasetReader,
        doc_reader=MeanTimeDocumentReader
    ),

    "narrative_container": DatasetMetadata(
        name="narrative_container",
        language="italian",
        url="https://drive.inesctec.pt/s/mKZsejEmLESCPCQ/download",
        reader=XMLDatasetReader,
        doc_reader=NarrativeContainerDocumentReader
    ),

    "ph_english": DatasetMetadata(
        name="ph_english",
        language="english",
        url="https://drive.inesctec.pt/s/ZnPeygpJT3X4pMX/download",
        reader=JSONDatasetReader,
        doc_reader=ProfessorHeidelTimeDocumentReader,
    ),

    "ph_french": DatasetMetadata(
        name="ph_french",
        language="french",
        url="https://drive.inesctec.pt/s/85npxdnKPc9Bomk/download",
        reader=JSONDatasetReader,
        doc_reader=ProfessorHeidelTimeDocumentReader,
    ),

    "ph_german": DatasetMetadata(
        name="ph_german",
        language="german",
        url="https://drive.inesctec.pt/s/NYsjP4yepifzZaF/download",
        reader=JSONDatasetReader,
        doc_reader=ProfessorHeidelTimeDocumentReader,
    ),

    "ph_italian": DatasetMetadata(
        name="ph_italian",
        language="italian",
        url="https://drive.inesctec.pt/s/qMrMQGPLTeEJb5A/download",
        reader=JSONDatasetReader,
        doc_reader=ProfessorHeidelTimeDocumentReader,
    ),

    "ph_portuguese": DatasetMetadata(
        name="ph_portuguese",
        language="portuguese",
        url="https://drive.inesctec.pt/s/p9dNZ58YQ4enQNa/download",
        reader=JSONDatasetReader,
        doc_reader=ProfessorHeidelTimeDocumentReader,
    ),

    "ph_spanish": DatasetMetadata(
        name="ph_spanish",
        language="spanish",
        url="https://drive.inesctec.pt/s/jFqpWdTj9r3RAXP/download",
        reader=JSONDatasetReader,
        doc_reader=ProfessorHeidelTimeDocumentReader,
    ),

    "platinum": DatasetMetadata(
        name="Platinum",
        language="english",
        url="https://drive.inesctec.pt/s/jpj3iwBDqiW2rWM/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader,
    ),

    "spanish_timebank": DatasetMetadata(
        name="spanish_timebank",
        language="spanish",
        url="https://drive.inesctec.pt/s/9jYWY44Nokd6axn/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),

    "tcr": DatasetMetadata(
        name="TCR",
        language="english",
        url="https://drive.inesctec.pt/s/WCFxDPc9JeQxG3e/download",
        reader=XMLDatasetReader,
        doc_reader=TCRDocumentReader,
    ),

    "tddiscourse": DatasetMetadata(
        name="TDDiscourse",
        language="english",
        url="https://drive.inesctec.pt/s/fqjsffnrk8TrCLg/download",
        base=["timebank_1.2"],
        reader=TDDiscourseDatasetReader,
    ),

    "tempeval_2_chinese": DatasetMetadata(
        name="tempeval_2_chinese",
        language="chinese",
        url="https://drive.inesctec.pt/s/L7TqZeJC4cG7yfs/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),

    "tempeval_2_english": DatasetMetadata(
        name="tempeval_2_english",
        language="english",
        url="https://drive.inesctec.pt/s/sgGpwHYjHNXqjmp/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),

    "tempeval_2_french": DatasetMetadata(
        name="tempeval_2_french",
        language="french",
        url="https://drive.inesctec.pt/s/nNeYpe9BCcj6aCg/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval2FrenchDocumentReader,
    ),

    "tempeval_2_italian": DatasetMetadata(
        name="tempeval_2_italian",
        language="italian",
        url="https://drive.inesctec.pt/s/roFgne7kTCdgpoj/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),

    "tempeval_2_korean": DatasetMetadata(
        name="tempeval_2_korean",
        language="korean",
        url="https://drive.inesctec.pt/s/HL9ie4nqebpLf8Q/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),

    "tempeval_2_spanish": DatasetMetadata(
        name="tempeval_2_spanish",
        language="spanish",
        url="https://drive.inesctec.pt/s/MNBoR9w29kZTo9T/download",
        reader=JSONDatasetReader,
        doc_reader=TempEval2DocumentReader,
    ),

    "tempeval_3": DatasetMetadata(
        name="tempeval_3",
        language="english",
        url="https://drive.inesctec.pt/s/yyDxHXpqkCZA6Cn/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader,
    ),

    "timebank_1.2": DatasetMetadata(
        name="TimeBank_1.2",
        language="english",
        url="https://drive.inesctec.pt/s/AJKjcCwEED6X9ae/download",
        reader=XMLDatasetReader,
        doc_reader=TimeBank12DocumentReader,

    ),

    "timebank_dense": DatasetMetadata(
        name="TimeBank_Dense",
        language="english",
        url="https://drive.inesctec.pt/s/4AbX8n25Dc3MyGy/download",
        base=["timebank_1.2"],
        reader=TimeBankDenseDatasetReader,

    ),

    "timebankpt": DatasetMetadata(
        name="TimeBankPT",
        language="portuguese",
        url="https://drive.inesctec.pt/s/BBQmStdTCL5FSMA/download",
        reader=XMLDatasetReader,
        doc_reader=TimeBankPTDocumentReader,
    ),

    "timebank": DatasetMetadata(
        name="TimeBank",
        language="english",
        url="https://drive.inesctec.pt/s/ndNDjnKq9CTdR8Q/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader,
    ),

    "traint3": DatasetMetadata(
        name="Traint3",
        language="spanish",
        url="https://drive.inesctec.pt/s/CFwAqZ8T8ZArHyC/download",
        reader=XMLDatasetReader,
        doc_reader=TempEval3DocumentReader
    ),

    "wikiwars": DatasetMetadata(
        name="wikiwars",
        language="english",
        url="https://drive.inesctec.pt/s/8ZPnNPfofwyyLT9/download",
        reader=XMLDatasetReader,
        doc_reader=WikiWarsDocumentReader
    ),

    "wikiwars_de": DatasetMetadata(
        name="wikiwars_de",
        language="german",
        url="https://drive.inesctec.pt/s/ysw4DGrpX9RT6ec/download",
        reader=XMLDatasetReader,
        doc_reader=WikiWarsDocumentReader
    ),
}
