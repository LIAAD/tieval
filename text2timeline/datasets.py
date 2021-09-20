from dataclasses import dataclass
from typing import Union, Tuple, Set

from text2timeline.base import Dataset
from text2timeline.readers import TMLDatasetReader, TableDatasetReader

DATASET_READER = {
    "tml": TMLDatasetReader,
    "table": TableDatasetReader
}


@dataclass
class DatasetMetadata:
    name: str
    reader: Union[TMLDatasetReader, TableDatasetReader]
    path: Tuple[str]
    url: str = None
    base: Set[str] = None
    extension: str = None
    columns: Tuple = None
    event_index: str = None
    description: str = None


# datasets that only have a table with the temporal links
DATASETS_METADATA = {

    "timebank": DatasetMetadata(
        name="timebank",
        reader=TMLDatasetReader,
        path=("data/TempEval-3/Train/TBAQ-cleaned/TimeBank",)
    ),

    "timebank-1.2": DatasetMetadata(
        name="timebank-1.2",
        reader=TMLDatasetReader,
        path=("data/TimeBank-1.2/data/timeml",),
        description=
        "The TimeBank Corpus contains 183 news articles that have been annotated with temporal information, "
        "adding events, times and temporal links between events and times. "
    ),

    "aquaint": DatasetMetadata(
        name="aquaint",
        reader=TMLDatasetReader,
        path=('data/TempEval-3/Train/TBAQ-cleaned/AQUAINT',)
    ),

    "timebank-pt": DatasetMetadata(
        name="timebank-pt",
        url="nlx-server.di.fc.ul.pt/~fcosta/TimeBankPT/TimeBankPTv1.0.zip",
        description=
        "TimeBankPT was obtained by translating the English data used in the first TempEval competition ("
        "http://timeml.org/tempeval).\nTimeBankPT can be found at http://nlx.di.fc.ul.pt/~fcosta/TimeBankPT.\nThat "
        "page contains some information about the corpus and a link to the release.",
        reader=TMLDatasetReader,
        path=(
            'data/TimeBankPT/train',
            'data/TimeBankPT/tests'
        )
    ),

    "tempeval-3": DatasetMetadata(
        name="tempeval-3",
        reader=TMLDatasetReader,
        path=(
            'data/TempEval-3/Train/TBAQ-cleaned/TimeBank',
            'data/TempEval-3/Train/TBAQ-cleaned/AQUAINT',
            # 'data/TempEval-3/Train/trainT3',
            'data/TempEval-3/Test/TempEval-3-Platinum'
        )
    ),

    "platinum": DatasetMetadata(
        name="platinum",
        reader=TMLDatasetReader,
        path=(
            'data/TempEval-3/Test/TempEval-3-Platinum',
        )
    ),

    "matres": DatasetMetadata(
        name="matres",
        base=("tempeval-3",),
        columns=("doc", "src_token", "tgt_token", "src", "tgt", "relation"),
        event_index="eiid",
        url="https://github.com/qiangning/MATRES",
        description="",
        reader=TableDatasetReader,
        path=(
            'data/MATRES/aquaint.txt',
            'data/MATRES/timebank.txt',
            'data/MATRES/platinum.txt',
        )
    ),

    "tddiscourse": DatasetMetadata(
        name="tddiscourse",
        base={"timebank-1.2"},
        columns=("doc", "src", "tgt", "relation"),
        event_index="eid",
        url="https://github.com/aakanksha19/TDDiscourse",
        description=
        "TDDiscourse is a dataset for temporal ordering of events, which specifically focuses on event pairs that "
        "are more than one sentence apart in a document. TDDiscourse was created by augmenting TimeBank-Dense. "
        "TimeBank-Dense focuses mainly on event pairs which are in the same or adjacent sentences (though they do "
        "include labels for some event pairs which are more than one sentence apart). TDDiscourse was created to "
        "address this gap and to turn the focus towards discourse-level temporal ordering, which turns out to be a "
        "harder task.",
        reader=TableDatasetReader,
        path=(
            "data/TDDiscourse/TDDMan/TDDManTrain.tsv",
            "data/TDDiscourse/TDDMan/TDDManDev.tsv",
            "data/TDDiscourse/TDDMan/TDDManTest.tsv",
            "data/TDDiscourse/TDDAuto/TDDAutoTrain.tsv",
            "data/TDDiscourse/TDDAuto/TDDAutoDev.tsv",
            "data/TDDiscourse/TDDAuto/TDDAutoTest.tsv",

        )
    ),

    "timebank-dense": DatasetMetadata(
        name="timebank-dense",
        base={"timebank-1.2"},
        columns=("doc", "src", "tgt", "relation"),
        event_index="eid",
        reader=TableDatasetReader,
        path=(
            "data/TimeBank-Dense/TimebankDense.full.txt",
        )
    ),
}


def load_dataset(name: str) -> Dataset:
    """

    Load temporally annotated dataset.

    The supported datasets are:
        TimeBank
        TimeBank.1-2
        Platinum
        Tempeval-3
        TimeBank-PT
        Aquaint
        MATRES
        TDDiscourse
        TimeBank-Dense

    Parameters
    ----------
    name

    Returns
    -------

    """

    name = name.lower().strip()
    metadata = DATASETS_METADATA[name]

    # guardians
    if name not in DATASETS_METADATA:
        raise ValueError(f"{name} not found on datasets")

    # table dataset
    if metadata.base:

        # read base dataset
        base_datasets = []
        for dataset_name in metadata.base:
            base_datasets += load_dataset(dataset_name)
        base_dataset = sum(base_datasets)

        # define reader
        reader = metadata.reader(metadata, base_dataset)

    # tml dataset
    else:
        reader = metadata.reader()

    return [reader.read(path) for path in metadata.path]


def load_timebank():
    return load_dataset("timebank")
