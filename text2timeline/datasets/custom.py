import warnings
from typing import List
from text2timeline import base
from glob import glob
import os
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
        'data/TimeBank-1.2/data'
    ],

    'aquaint': [
        'data/TempEval-3/Train/TBAQ-cleaned/AQUAINT'
    ],

    'platinum': [
        'data/TempEval-3/Test/TempEval-3-Platinum'
    ],

    'timebank-pt': [
        'data/TimeBankPT/train',
        'data/TimeBankPT/test'
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

# map between dataset and document object build to read it
DATASET_DOCUMENT_OBJ = {
    'timebank': base.TimeBank12Document,
    'timebank-1.2': base.TimeBank12Document,
    'aquaint': base.AquaintDocument,
    'timebank-pt': base.TimeBankPTDocument,
    'tempeval-3': base.TempEval3Document,
    'platinum': base.PlatinumDocument
}

TABLE_READER = {
    'matres': base.Document,
    'tddiscourse': base.Document,
    'timebank-dense': base.Document,
}


class ReadTlinksTable:
    def __init__(self, dataset):

        self.dataset = dataset
        self._tlink_id = 1

        if dataset.name == 'matres':
            self._tlink_from_line = self._matres

        elif dataset.name == 'tddiscourse':
            self._tlink_from_line = self._tddiscourse

        elif dataset.name == 'timebank-dense':
            self._tlink_from_line = self._tbdense

    def read(self, table):

        with open(table, 'r') as f:
            content = [line.split() for line in f.read().split('\n')]

            tlinks_by_doc = collections.defaultdict(list)
            for line in content:
                if line:
                    tlinks_by_doc[line[0]] += [line[1:]]  # doc: tlink

            tlinks = {}
            for doc, lines in tlinks_by_doc.items():

                if doc not in self.dataset.doc_names:
                    warnings.warn(f"Document {doc} not found on the source dataset")
                    continue

                tlinks[doc] = [self._tlink_from_line(line, doc) for line in lines]

        return tlinks

    def _find_expressions(self, exp_id: str, doc: str):

        if exp_id in self.dataset[doc].expressions_uid:
            return self.dataset[doc].expressions_uid[exp_id]

        elif exp_id in self.dataset[doc].expressions_id:
            return self.dataset[doc].expressions_id[exp_id]

        else:
            warnings.warn(f"Expression with id {exp_id} was not found in document {doc}")
            return None

    def _matres(self, line: list, doc: str):

        src, tgt = f'ei{line[2]}', f'ei{line[3]}'

        source = self._find_expressions(src, doc)
        target = self._find_expressions(tgt, doc)

        if source is None or target is None:
            return None, None

        tlink = base.TLink(
            id=f'l{self._tlink_id}',
            source=source,
            target=target,
            relation=line[4]
        )

        self._tlink_id += 1
        return tlink

    def _tddiscourse(self, line: list, doc: str):

        src, tgt = line[0], line[1]

        source = self._find_expressions(src, doc)
        target = self._find_expressions(tgt, doc)

        if source is None or target is None:
            return None, None

        tlink = base.TLink(
            id=f'l{self._tlink_id}',
            source=source,
            target=target,
            relation=line[2]
        )

        self._tlink_id += 1
        return tlink

    def _tbdense(self, line: list, doc: str):
        return self._tddiscourse(line, doc)


class Dataset:

    def __init__(self, name, docs, path=None):
        self.name = name
        self.docs = docs
        self.path = path

    def __repr__(self):
        return f"Dataset(name={self.name}, path={self.path})"

    def __add__(self, other):

        result = Dataset(
            name=f'{self.name} {other.name}',
            docs=self.docs + other.docs
        )

        return result

    def __getitem__(self, item):
        for doc in self.docs:
            if doc.name == item:
                return doc

    @property
    def doc_names(self):
        return [doc.name for doc in self.docs]

    def tlinks_count(self):
        """

        Get labels names and count of dataset.

        :return:
        """

        # get relation of all tlinks
        relations = [tlink.relation for doc in self.docs for tlink in doc.tlinks]

        return collections.Counter(relations)

    def split(self, percentage, names=['train', 'valid']):

        # TODO: add shuffle?

        num_docs = len(self.docs)
        num_train_docs = int(num_docs * percentage)

        train_docs = self.docs[:num_train_docs]
        valid_docs = self.docs[num_train_docs:]

        train_set = Dataset(name=names[0], docs=train_docs)
        valid_set = Dataset(name=names[1], docs=valid_docs)

        return train_set, valid_set

    def reduce_tlinks(self, map):

        for doc in self.docs:
            for tlink in doc.tlinks:
                tlink.relation = map[tlink.relation]

    def augment_tlinks(self, relations: List[str] = None):
        """ Adds the inverse relations of the specified relations.
        If None relations are passed

        :param relations:
        :return:
        """

        for doc in self.docs:
            doc.augment_tlinks(relations)


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
        self.paths = PATHS[self.name]

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

        # get Document object to read dataset
        Document = DATASET_DOCUMENT_OBJ[self.name]

        # read documents
        datasets = []
        for path in self.paths:

            tml_paths = [os.path.join(path, p)
                         for p in os.listdir(path)
                         if p.endswith('.tml')]

            # read all documents in path
            docs = [Document(p) for p in tml_paths]

            # build dataset
            dataset = Dataset(
                name=self.name,
                path=path,
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

        # read datasets in which the current depends on
        dataset_reader = DatasetReader()
        dataset_reader.read(DATASETS_DEPENDENT[self.name])
        dataset = sum(dataset_reader.datasets, Dataset('', []))

        dataset.name = self.name

        # read tlinks of the dataset
        table_reader = ReadTlinksTable(dataset)

        datasets = []
        for path in PATHS[self.name]:
            table_paths = [os.path.join(path, p)
                           for p in os.listdir(path)
                           if p.endswith(('txt', 'tsv'))]

            for table_path in table_paths:

                tlinks_by_doc = table_reader.read(table_path)

                docs = []
                for doc_name, tlinks in tlinks_by_doc.items():
                    doc = dataset[doc_name]
                    doc.tlinks = tlinks
                    docs += [doc]

                datasets += [Dataset(name=self.name, docs=docs, path=table_path)]

        return datasets


class DatasetReader:
    """

    Read temporal relation datasets.

    """

    def __init__(self):
        self.datasets = []

    def read(self, datasets: list):
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

        for dataset_name in datasets:

            dataset = ReadDataset(dataset_name)

            # update datasets dictionary with the new datasets
            self.datasets += dataset.read()
