from text2story import narrative

import glob

import os

from pprint import pprint

# TODO: add support to tempeval-2


DATASETS = [
    'timebank',
    'aquaint',
    'matres',
    'timebank-pt',
    'tddiscourse',
    'timebank-dense',
    'tempeval-3'
]

# datasets that have the text annotated in a .tml file
DATASETS_INDEPENDENT = [
    'timebank',
    'aquaint',
    'timebank-pt',
    'tempeval-3'
]

# datasets that only have a table with the temporal links
DATASETS_DEPENDENT = {
    'matres': {
        'dependencies': ['aquaint', 'timebank', 'platinum']
    },
    'tddiscourse': {
        'dependencies': ['timebank']
    },
    'timebank-dense': {
        'dependencies': ['timebank']
    }
}

# path to each dataset.
PATHS = {
    'timebank': 'data/TempEval-3/Train/TBAQ-cleaned/TimeBank',
    'aquaint': 'data/TempEval-3/Train/TBAQ-cleaned/AQUAINT',
    'platinum': 'data/TempEval-3/Test/TempEval-3-Platinum',
    'timebankpt': 'data/TimeBankPT',
    'tempeval-3': 'data/TempEval-3',
    'tempeval-2': 'data/Tempeval-2',
    'matres': 'data/MATRES',
    'tddiscourse': 'data/TDDiscourse'
}


datasets = ['matres', 'timebank-pt']
dataset = 'timebank'


def read(datasets: list):

    # guardian
    cond = [dataset for dataset in datasets if dataset.lower() not in DATASETS]
    if any(cond):
        datasets_not_supported = "\", \" ".join(cond)
        msg = f'Dataset \"{datasets_not_supported}\" is not supported. The supported datasets are {", ".join(DATASETS)}.'
        raise NameError(msg)

    for dataset in datasets:
        if dataset in DATASETS_INDEPENDENT:
            read_independent_dataset(dataset)
        else:
            read_dependent_dataset(dataset)


def read_independent_dataset(dataset):
    glob_path = os.path.join(PATHS[dataset], '*.tml')
    file_paths = glob.glob(glob_path)

    pprint(file_paths)
"""
    list(os.walk(PATHS[dataset]))
    test = list(os.walk(PATHS['tempeval-3']))
    pprint(test)
    print(f'Reading dataset {dataset} from folder {folder}...')

    data[folder][dataset] = [narrative.Document(file_path) for file_path in file_paths]

"""
path = PATHS['tempeval-3']

glob_path = os.path.join(path, '*.tml')
file_paths = glob.glob(glob_path)


def get_folders_with_tml(path):
    result = dict()
    for walk_path, folders, files in os.walk(path):
        for folder in folders:
            folder_path = os.path.join(walk_path, folder)
            result[folder] = get_folders_with_tml(folder_path)

        tml_files = [file for file in files if file.endswith('.tml')]
        if any(tml_files):
            return tml_files

test = get_folders_with_tml(path)


def read_dependent_dataset(dataset):
    path = PATHS[dataset]


read(datasets)