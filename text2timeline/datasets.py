
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
            'data/TimeBank-1.2/data/timeml',
            'data/TimeBank-1.2/data/extra'
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
        "url": "",
        "description": "",
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
            'data/TempEval-3/Train/trainT3',
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
        "base": ("aquaint", "timebank", "platinum"),
        "extension": "*.txt",
        "columns": ("doc", "src_token", "tgt_token", "src", "tgt", "relation"),
        "url": "",
        "description": "",
        "type": "table",
        "path": (
            'data/MATRES',
        )
    },

    "tddiscourse": {
        "base": {"timebank"},
        "extension": "*.tsv",
        "columns": ("doc", "src", "tgt", "relation"),
        "url": "",
        "description": "",
        "type": "table",
        "path": (
            'data/TDDiscourse/TDDMan',
            'data/TDDiscourse/TDDAuto'
        )
    },

    "timebank-dense": {
        "base": {"timebank"},
        "extension": "*.txt",
        "columns": ("doc", "src", "tgt", "relation"),
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
    metadata = DATASETS_METADATA[name]

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

    reader = TableDatasetReader(metadata)
    return [reader.read(path) for path in metadata['path']]
