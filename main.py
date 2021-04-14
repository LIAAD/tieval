from text2story import narrative

from glob import glob

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
    'tempeval-3',
    'timebank-1.2'
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
    'timebank-1.2': 'data/TimeBank-1.2',
    'aquaint': 'data/TempEval-3/Train/TBAQ-cleaned/AQUAINT',
    'platinum': 'data/TempEval-3/Test/TempEval-3-Platinum',
    'timebankpt': 'data/TimeBankPT',
    'tempeval-3': 'data/TempEval-3',
    'tempeval-2': 'data/Tempeval-2',
    'matres': 'data/MATRES',
    'tddiscourse': 'data/TDDiscourse'
}


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


def get_folders_with_tml(path):
    return [dp for dp, _, _ in os.walk(path) if any(glob(dp + '/*.tml'))]


def get_tml_path(path: str) -> dict:
    """ Given a path it will return a dictionary in which the keys are the folder path and the values are the paths to
    every .tml path in that directory.

    :param path:
    :return:
    """
    result = dict()
    for dp, _, _ in os.walk(path):

        tml_files = glob(dp + '/*.tml')
        if any(tml_files):
            name = dp[len(path) + 1:]
            result[name] = tml_files
    return result


doc = narrative.TimeBank12Document('data/TimeBank-1.2/data/extra/SJMN91-06338157.tml')

path = PATHS['timebank-1.2']
tml_paths = get_tml_path(path)

for folder_name, paths in tml_paths.items():
    for path in paths:
        try:
            doc = narrative.TimeBank12Document(path)
        except:
            print(path)



docs = {folder_name: [narrative.Document(path) for path in paths] for folder_name, paths in tml_paths.items()}

pprint(docs)

def read_dependent_dataset(dataset):
    pass
