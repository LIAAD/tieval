import os

import glob

from text2timeline import base

import collections


def load_data(path) -> dict:
    """ Read TempEval-3 datasets.

    :param path:
    :return: A dictionary with keys 'train' and 'tests'.
    """
    paths = {
        'train': {
            'aquaint': os.path.join(path, r'Train/TBAQ-cleaned/AQUAINT'),
            'timebank': os.path.join(path, r'Train/TBAQ-cleaned/TimeBank'),
            'train_t3': os.path.join(path, r'Train/trainT3')
        },
        'tests': {
            'platinum': os.path.join(path, r'Test/TempEval-3-Platinum')
        }
    }

    data = collections.defaultdict(dict)
    for folder in paths:

        for dataset in paths[folder]:

            glob_path = os.path.join(paths[folder][dataset], '*.tml')
            file_paths = glob.glob(glob_path)

            print(f'Reading dataset {dataset} from folder {folder}...')

            data[folder][dataset] = [base.Document(file_path) for file_path in file_paths]

            print('Done.\n')
    return data
