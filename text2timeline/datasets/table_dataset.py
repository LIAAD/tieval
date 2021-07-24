import collections
import warnings

from text2timeline import base


def _read_matres_tlink(self, line: list, doc: str) -> base.TLink:
    """Read a temporal link from the MATRES dataset."""

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


def _read_tddiscourse_tlink(self, line: list, doc: str):
    """Read a temporal link from the TDDiscourse dataset."""

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


def _read_tbdense_tlink(self, line: list, doc: str):
    """Read a temporal link from the Timebank Dense dataset."""

    return self._tddiscourse(line, doc)


# mapping between the name of the dataset and the respective tlink reader
TLINK_READER = {
    'matres': _read_matres_tlink,
    'tddiscourse': _read_tddiscourse_tlink,
    'tbdense': _read_tbdense_tlink,

}


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

