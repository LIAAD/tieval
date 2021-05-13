from typing import List
from text2timeline import base
from glob import glob
import os
import collections

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
    'timebank': base.TimeBank12Document,
    'timebank-1.2': base.TimeBank12Document,
    'aquaint': base.AquaintDocument,
    'matres': base.Document,
    'timebank-pt': base.TimeBankPTDocument,
    'tddiscourse': base.Document,
    'timebank-dense': base.Document,
    'tempeval-3': base.TempEval3Document,
}


class ReadDataset:
    """

    Read temporal relation datasets.

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
            return self.__read_independent_dataset()

        else:
            return self.__read_dependent_dataset()

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
        datasets = []
        for folder_name, paths in tml_paths.items():

            # read all documents in folder
            docs = []
            for path in paths:
                doc = Document(path)
                docs.append(doc)

            # build dataset
            dataset = Dataset(
                name=self.name,
                folder=folder_name,
                docs=docs
            )

            datasets += [dataset]

        return datasets

    def __read_dependent_dataset(self):
        """

        Read datasets that do not have the original text, only the annotations.

        ...

        :param dataset:
        :return:
        """

        return None


class DatasetReader:
    """

    Read temporal relation datasets.

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

    def __init__(self):

        self.datasets = []

    def read(self, datasets: list):

        for dataset_name in datasets:

            dataset = ReadDataset(dataset_name)

            # update datasets dictionary with the new datasets
            self.datasets += dataset.read()


class Dataset:

    def __init__(self, name, folder, docs):
        self.name = name
        self.folder = folder
        self.docs = docs

    def __repr__(self):
        return f"Dataset(name={self.name}, folder={self.folder})"

    def __add__(self, other):

        result = Dataset(
            name=f'{self.name} {other.name}',
            folder=None,
            docs=self.docs + other.docs
        )

        return result

    def tlinks_count(self):
        """

        Get labels names and count of dataset.

        :return:
        """

        # get relation of all tlinks
        relations = [tlink.interval_relation for doc in self.docs for tlink in doc.tlinks]

        return collections.Counter(relations)

    def split(self, percentage, names=['train', 'valid']):

        # TODO: add shuffle?

        num_docs = len(self.docs)
        num_train_docs = int(num_docs * percentage)

        train_docs = self.docs[:num_train_docs]
        valid_docs = self.docs[num_train_docs:]

        train_set = Dataset(name=names[0], folder=None, docs=train_docs)
        valid_set = Dataset(name=names[1], folder=None, docs=valid_docs)

        return train_set, valid_set

    def reduce_tlinks(self, map):

        for doc in self.docs:
            for tlink in doc.tlinks:
                tlink.interval_relation = map[tlink.interval_relation]

    def augment_tlinks(self, relations: List[str] = None):
        """ Adds the inverse relations of the specified relations.
        If None relations are passed

        :param relations:
        :return:
        """

        for doc in self.docs:
            doc.augment_tlinks(relations)
