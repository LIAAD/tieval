from text2story import narrative

from glob import glob

import os

import numpy as np

import collections

from pprint import pprint

# TODO: add support to tempeval-2


DATASETS = [
    'timebank',  #
    'timebank-1.2',
    'aquaint',  #
    'matres',
    'timebank-pt',  #
    'tddiscourse',
    'timebank-dense',
    'tempeval-3' #
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
    'timebank-pt': 'data/TimeBankPT',
    'tempeval-3': 'data/TempEval-3',
    'tempeval-2': 'data/Tempeval-2',
    'matres': 'data/MATRES',
    'tddiscourse': 'data/TDDiscourse'
}

# map between dataset and document object build to read it
DATASET_DOCUMENT_OBJ = {
    'timebank': narrative.TimeBank12Document,
    'timebank-1.2': narrative.TimeBank12Document,
    'aquaint': narrative.AquaintDocument,
    'matres': narrative.Document,
    'timebank-pt': narrative.TimeBankPTDocument,
    'tddiscourse': narrative.Document,
    'timebank-dense': narrative.Document,
    'tempeval-3': narrative.TempEval3Document,
}


class Dataset:
    """

    Read temporal relation dataset.

    The set of possible datasets are:
        - timebank
        - timebank-1.2
        - aquaint
        - matres
        - timebank-pt
        - tddiscourse
        - timebank-dense
        - tempeval-3

    """

    def __init__(self, dataset: str):

        # guardian
        if dataset.lower() not in DATASETS:
            msg = f'Dataset \"{dataset}\" is not supported. ' \
                  f'The supported datasets are {", ".join(DATASETS)}.'
            raise NameError(msg)

        self.name = dataset
        self.path = PATHS[self.name]

        self.docs = None

    def tml_paths(self) -> dict:
        """

        Given a path, it will return a dictionary in which the keys are the folder path and the values are the paths
        to every .tml path in that directory.

        ...

        :param path:
        :return:
        """

        result = dict()

        # walk inside path to look for folders that contain .tml files
        for dp, _, _ in os.walk(self.path):

            # search for the .tml files
            tml_files = glob(dp + '/*.tml')

            if any(tml_files):
                name = dp[len(self.path) + 1:]  # get folder name
                if name == '':
                    name = 'root'

                result[name] = tml_files

        return result

    def read(self):
        """ Read a dataset.

        :return:
        """

        # read dataset
        if self.name in DATASETS_INDEPENDENT:
            self.docs = self.__read_independent_dataset()

        else:
            self.docs = self.__read_dependent_dataset()

    def __read_independent_dataset(self):
        """

        Read dataset that has the .tml files.

        ...

        :param dataset:
        :return:
        """

        # get the .tml files
        tml_paths = self.tml_paths()

        # get Document object to read dataset
        Document = DATASET_DOCUMENT_OBJ[self.name]

        # read documents
        docs = {folder_name: [Document(path) for path in paths] for folder_name, paths in tml_paths.items()}

        return docs

    def __read_dependent_dataset(self):
        """

        Read datasets that do not have the original text, only the annotations.

        ...

        :param dataset:
        :return:
        """

        return None
