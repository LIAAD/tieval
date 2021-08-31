import collections
import warnings

from text2timeline import base


class ReadDatasetFromTable:
    """
    Read datasets that are a table where each row is a temporal link. As is the case of: MATRES, TDDiscourse and
    TimeBank-Dense.
    """

    def __init__(self, dataset_name: str):

        self.dataset = dataset_name
        self.tlink_reader = TLINK_READER[dataset_name]

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

                tlinks[doc] = [self.tlink_reader(line, doc) for line in lines]

        return tlinks

    def _find_expressions(self, exp_id: str, doc: str):

        if exp_id in self.dataset[doc].expressions_uid:
            return self.dataset[doc].expressions_uid[exp_id]

        elif exp_id in self.dataset[doc].expressions_id:
            return self.dataset[doc].expressions_id[exp_id]

        else:
            warnings.warn(f"Expression with id {exp_id} was not found in document {doc}")
            return None

