import collections
import os

from tieval import datasets
from tieval.entities import Timex


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


# def test_closure(tmp_path):
#
#     os.chdir(tmp_path)
#
#     n_datasets = len(datasets.SUPPORTED_DATASETS)
#     for dataset_id, dataset in enumerate(datasets.SUPPORTED_DATASETS):
#         print(f"Working in dataset {dataset} {dataset_id}/{n_datasets}")
#
#         data = datasets.read(dataset)
#         n_docs = len(data)
#         for doc_id, doc in enumerate(data.documents):
#             print(f"Working in document {doc.name} {doc_id}/{n_docs}")
#
#             tls = doc.tlinks
#             tls_inf = doc.temporal_closure
#
#             msg = f"Number o original tlinks is smaller than the one inferred for document {doc.name} in dataset {dataset}"
#             assert len(tls) <= len(tls_inf), msg
#
#             entity_pairs = collections.defaultdict(list)
#             for tl in tls_inf:
#                 # sorted is to ensure that the assertion is independent of the source and target order.
#                 key = tuple(sorted([tl.source.id, tl.target.id]))
#                 entity_pairs[key] += [tl]
#
#             for key, value in entity_pairs.items():
#                 msg = f"Inferred tlinks contains more than one relation for the same entity pair in {doc.name} in dataset {dataset}"
