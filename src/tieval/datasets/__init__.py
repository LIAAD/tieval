from pathlib import Path

from tieval.base import Dataset
from tieval.datasets.metadata import DATASETS_METADATA
from tieval.utils import download_url

SUPPORTED_DATASETS = list(DATASETS_METADATA)


def read(dataset: str, path: Path = None) -> Dataset:
    """Load temporally annotated dataset."""

    dataset = dataset.lower().strip()

    if dataset not in DATASETS_METADATA:
        raise ValueError(f"{dataset} not found on datasets")
    metadata = DATASETS_METADATA[dataset]

    if path is None:
        path = Path("data/")

    if not (path / dataset).is_dir():
        download(dataset, path)

    if metadata.base:  # table dataset
        base_datasets = []
        for dataset_name in metadata.base:
            base_datasets += [read(dataset_name, path)]
        base_dataset = sum(base_datasets)
        reader = metadata.reader(base_dataset)
    else:  # tml dataset
        reader = metadata.reader(metadata.doc_reader)

    return reader.read((path / dataset))


def download(dataset: str, path: Path) -> None:
    """ Download corpus."""

    dataset = dataset.lower().strip()
    metadata = DATASETS_METADATA.get(dataset)

    if metadata is None:
        raise Exception(f"{dataset} not recognized.")

    path.mkdir(exist_ok=True, parents=True)

    download_url(metadata.url, path)
