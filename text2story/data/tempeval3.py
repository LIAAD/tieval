import text2story.read_xml as rxml

import os

import collections


def load_data(path, tokenizer):
    paths = {
        'train': {
            'aquaint': os.path.join(path, r'Train/TBAQ-cleaned/AQUAINT'),
            'timebank': os.path.join(path, r'Train/TBAQ-cleaned/TimeBank'),
            # 'train_t3': os.path.join(path, r'Train/trainT3')  # TODO: include train_t3 in the dataset.
        },
        'test': {
            'platinum': os.path.join(path, r'Test/TempEval-3-Platinum')
        }
    }

    data = collections.defaultdict(dict)
    for folder in paths:
        for dataset in paths[folder]:
            print(f'Reading dataset {dataset} from folder {folder}...')
            base, tlinks = rxml.read_dir(paths[folder][dataset], tokenizer)
            data[folder][dataset] = {
                'base': base,
                'tlinks': tlinks
            }
            print('Done.\n')

    return data
