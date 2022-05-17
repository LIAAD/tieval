import logging
import os

from tieval import datasets
from tieval.entities import Timex

logging.basicConfig(level=logging.DEBUG)


def test_endpoints(tmp_path):

    os.chdir(tmp_path)

    for dataset in datasets.SUPPORTED_DATASETS:

        data = datasets.read(dataset)
        for doc in data.documents:

            for ent in doc.entities:

                if isinstance(ent, Timex) and ent.is_dct:
                    continue

                msg = f"Dataset: {dataset} Document: {doc.name} Entity: {ent.id} Missing endpoints"
                assert ent.endpoints is not None, msg

                s, e = ent.endpoints
                msg = f"Dataset: {dataset}\nDocument: {doc.name}\nExpected:_{ent.text}_\nObserved:_{doc.text[s:e]}_"
                assert doc.text[s:e] == ent.text, msg
