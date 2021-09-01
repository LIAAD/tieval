
from typing import List, Union

from text2timeline.base import Dataset
from text2timeline.readers import TMLDatasetReader


DATASETS = [
    'timebank',  #
    'timebank-1.2',
    'aquaint',  #
    'matres',
    'timebank-pt',  #
    'tddiscourse',
    'timebank-dense',
    'tempeval-3',
    'platinum'
]

# datasets that have the text annotated in a .tml file
DATASETS_INDEPENDENT = [
    'timebank',
    'aquaint',
    'timebank-pt',
    'tempeval-3',
    'timebank-1.2',
    'platinum'
]

# datasets that only have a table with the temporal links
DATASETS_DEPENDENT = {
    'matres': ['aquaint', 'timebank', 'platinum'],
    'tddiscourse': ['timebank'],
    'timebank-dense': ['timebank']
}

# paths to each dataset.
PATHS = {
    'timebank': [
        'data/TempEval-3/Train/TBAQ-cleaned/TimeBank'
    ],

    'timebank-1.2': [
        'data/TimeBank-1.2/data/extra',
        'data/TimeBank-1.2/data/timeml',
    ],

    'aquaint': [
        'data/TempEval-3/Train/TBAQ-cleaned/AQUAINT'
    ],

    'platinum': [
        'data/TempEval-3/Test/TempEval-3-Platinum'
    ],

    'timebank-pt': [
        'data/TimeBankPT/train',
        'data/TimeBankPT/tests'
    ],

    'tempeval-3': [
        'data/TempEval-3/Train/TBAQ-cleaned/AQUAINT',
        'data/TempEval-3/Train/TBAQ-cleaned/TimeBank',
        'data/TempEval-3/Train/trainT3',
        'data/TempEval-3/Test/TempEval-3-Platinum'
    ],

    # 'tempeval-2': 'data/Tempeval-2',

    'matres': [
        'data/MATRES'
    ],

    'tddiscourse': [
        'data/TDDiscourse/TDDAuto',
        'data/TDDiscourse/TDDMan'
    ],

    'timebank-dense': [
        'data/TimeBank-Dense'
    ]
}


def load_dataset(name: str) -> Dataset:
    name = name.lower().strip()
    reader = TMLDatasetReader()
    return [reader.read(path) for path in PATHS[name]]
