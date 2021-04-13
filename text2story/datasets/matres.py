import os

import glob

from text2story import narrative

import collections


def load_data(path) -> dict:
    """ Read MATRES dataset.

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



    return data
