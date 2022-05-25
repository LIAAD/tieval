import collections

from tieval import datasets


def run_doc(doc_name, dataset):
    data = datasets.read(dataset)
    doc = data[doc_name]

    tls = doc.tlinks
    tls_inf = doc.temporal_closure

    msg = f"Number o original tlinks is smaller than the one inferred for document {doc.name} in dataset {dataset}"
    assert len(tls) <= len(tls_inf), msg

    entity_pairs = collections.defaultdict(list)
    for tl in tls_inf:
        # sorted is to ensure that the assertion is independent of the source and target order.
        key = tuple(sorted([tl.source.id, tl.target.id]))
        entity_pairs[key] += [tl]


def run_all():
    n_datasets = len(datasets.SUPPORTED_DATASETS)
    for dataset_id, dataset in enumerate(datasets.SUPPORTED_DATASETS):
        print(f"Working in dataset {dataset} {dataset_id}/{n_datasets}")

        data = datasets.read(dataset)
        n_docs = len(data)
        for doc_id, doc in enumerate(data.documents):
            print(f"Working in document {doc.name} {doc_id}/{n_docs}")

            tls = doc.tlinks
            tls_inf = doc.temporal_closure

            msg = f"Number o original tlinks is smaller than the one inferred for document {doc.name} in dataset {dataset}"
            assert len(tls) <= len(tls_inf), msg

            entity_pairs = collections.defaultdict(list)
            for tl in tls_inf:
                # sorted is to ensure that the assertion is independent of the source and target order.
                key = tuple(sorted([tl.source.id, tl.target.id]))
                entity_pairs[key] += [tl]

            for key, value in entity_pairs.items():
                msg = f"Inferred tlinks contains more than one relation for the same entity pair in {doc.name} in dataset {dataset}"


def run_all():
    n_datasets = len(datasets.SUPPORTED_DATASETS)
    for dataset_id, dataset in enumerate(datasets.SUPPORTED_DATASETS):
        print(f"Working in dataset {dataset} {dataset_id}/{n_datasets}")

        data = datasets.read(dataset)
        n_docs = len(data)
        for doc_id, doc in enumerate(data.documents):
            print(f"Working in document {doc.name} {doc_id}/{n_docs}")

            tls = doc.tlinks
            tls_inf = doc.temporal_closure

            msg = f"Number o original tlinks is smaller than the one inferred for document {doc.name} in dataset {dataset}"
            assert len(tls) <= len(tls_inf), msg

            entity_pairs = collections.defaultdict(list)
            for tl in tls_inf:
                # sorted is to ensure that the assertion is independent of the source and target order.
                key = tuple(sorted([tl.source.id, tl.target.id]))
                entity_pairs[key] += [tl]

            for key, value in entity_pairs.items():
                msg = f"Inferred tlinks contains more than one relation for the same entity pair in {doc.name} in dataset {dataset}"


if __name__ == "__main__":
    run_doc("APW19980818.0515", "aquaint")