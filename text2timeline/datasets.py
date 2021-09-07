
from typing import List, Union

from text2timeline.base import Dataset
from text2timeline.readers import TMLDatasetReader, TableDatasetReader


DATASET_READER = {
    "tml": TMLDatasetReader,
    "table": TableDatasetReader
}


# datasets that only have a table with the temporal links
DATASETS_METADATA = {
    "timebank": {
        "url": "",
        "description": "",
        "type": "tml",
        "path": (
            "data/TempEval-3/Train/TBAQ-cleaned/TimeBank",
        )
    },

    "timebank-1.2": {
        "url": "",

        "description":
            """""",

        "type": "tml",
        "path": (
            "data/TimeBank-1.2/data/timeml",
        )
    },

    "aquaint": {
        "url": "",
        "description": "",
        "type": "tml",
        "path": (
            'data/TempEval-3/Train/TBAQ-cleaned/AQUAINT',
        )
    },

    "timebank-pt": {
        "url": "nlx-server.di.fc.ul.pt/~fcosta/TimeBankPT/TimeBankPTv1.0.zip",

        "description":
            """TimeBankPT was obtained by translating the English data used in the first TempEval 
        competition (http://timeml.org/tempeval).\nTimeBankPT can be found at 
        http://nlx.di.fc.ul.pt/~fcosta/TimeBankPT.\nThat page contains some information about the corpus and a link 
        to the release.""",

        "type": "tml",
        "path": (
            'data/TimeBankPT/train',
            'data/TimeBankPT/tests'
        )
    },

    "tempeval-3": {
        "url": "",
        "description": "",
        "type": "tml",
        "path": (
            'data/TempEval-3/Train/TBAQ-cleaned/TimeBank',
            'data/TempEval-3/Train/TBAQ-cleaned/AQUAINT',
            # 'data/TempEval-3/Train/trainT3',
            'data/TempEval-3/Test/TempEval-3-Platinum'
        )
    },

    "platinum": {
        "url": "",
        "description": "",
        "type": "tml",
        "path": (
            'data/TempEval-3/Test/TempEval-3-Platinum',
        )
    },

    "matres": {
        "base": ("tempeval-3", ),
        "extension": "*.txt",
        "columns": ("doc", "src_token", "tgt_token", "src", "tgt", "relation"),
        "index": "eiid",
        "url": "https://github.com/qiangning/MATRES",
        "description": "",
        "type": "table",
        "path": (
            'data/MATRES',
        )
    },

    "tddiscourse": {
        "base": {"timebank-1.2"},
        "extension": "*.tsv",
        "columns": ("doc", "src", "tgt", "relation"),
        "index": "eid",
        "url": "https://github.com/aakanksha19/TDDiscourse",
        "description":
            "TDDiscourse is a dataset for temporal ordering of events, which specifically focuses on event pairs that "
            "are more than one sentence apart in a document. TDDiscourse was created by augmenting TimeBank-Dense. "
            "TimeBank-Dense focuses mainly on event pairs which are in the same or adjacent sentences (though they do "
            "include labels for some event pairs which are more than one sentence apart). TDDiscourse was created to "
            "address this gap and to turn the focus towards discourse-level temporal ordering, which turns out to be a "
            "harder task.",
        "type": "table",
        "path": (
            'data/TDDiscourse/TDDMan',
            'data/TDDiscourse/TDDAuto'
        )
    },

    "timebank-dense": {
        "base": {"timebank-1.2"},
        "extension": "*.txt",
        "columns": ("doc", "src", "tgt", "relation"),
        "index": "eid",
        "url": "",
        "description": "",
        "type": "table",
        "path": (
            "data/TimeBank-Dense",
        )
    },
}


def load_tml_dataset(name: str) -> Dataset:
    """

    Load temporally annotated dataset.

    The supported datasets are:
        TimeBank
        AQUAINT
        TimeBank-PT
        TempEval-3
        Platinum

    Parameters
    ----------
    name

    Returns
    -------

    """

    name = name.lower().strip()
    metadata = DATASETS_METADATA.get(name)

    # guardians
    if name not in DATASETS_METADATA:
        raise ValueError(f"{name} not found on datasets")

    if metadata["type"] != "tml":
        raise TypeError(f"{name} is not an TML dataset.")

    reader = TMLDatasetReader(metadata)
    return [reader.read(path) for path in metadata['path']]


def load_table_dataset(name: str) -> Dataset:
    """

    Load temporally annotated dataset.

    The supported datasets are:
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

    if metadata["type"] != "table":
        raise TypeError(f"{name} is not an table dataset.")

    # read base dataset
    base_datasets = []
    for dataset_name in metadata["base"]:
        base_datasets += load_tml_dataset(dataset_name)
    base_dataset = sum(base_datasets)

    reader = TableDatasetReader(metadata, base_dataset)

    return [reader.read(path) for path in metadata['path']]
