import os

import glob

from text2story import narrative

import collections


def load_data(path) -> dict:
    """ Read TempEval-3 data.

    :param path:
    :return: A dictionary with keys 'train' and 'test'.
    """
    paths = {
        'train': {
            'aquaint': os.path.join(path, r'Train/TBAQ-cleaned/AQUAINT'),
            'timebank': os.path.join(path, r'Train/TBAQ-cleaned/TimeBank'),
            'train_t3': os.path.join(path, r'Train/trainT3')
        },
        'test': {
            'platinum': os.path.join(path, r'Test/TempEval-3-Platinum')
        }
    }

    data = collections.defaultdict(dict)
    for folder in paths:

        for dataset in paths[folder]:

            glob_path = os.path.join(paths[folder][dataset], '*.tml')
            file_paths = glob.glob(glob_path)

            print(f'Reading dataset {dataset} from folder {folder}...')

            data[folder][dataset] = [narrative.Document(file_path) for file_path in file_paths]

            print('Done.\n')
    return data
