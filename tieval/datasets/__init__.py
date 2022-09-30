import os
import pathlib

from tieval.base import Dataset
from tieval.datasets.metadata import DATASETS_METADATA
from tieval.utils import download_url

SUPPORTED_DATASETS = list(DATASETS_METADATA)


def read(dataset: str) -> Dataset:
    """Load temporally annotated dataset.
    The supported datasets are:
        TimeBank
        TimeBank.1-2
        Platinum
        Tempeval-3
        TimeBank-PT
        Aquaint
        MATRES
        TDDiscourse
        TimeBank-Dense

    
    :param str dataset: The name of the dataset to read.
    :return List[Dataset]: A list with the dataset.
    """

    dataset = dataset.lower().strip()
    metadata = DATASETS_METADATA[dataset]

    # guardians
    if dataset not in DATASETS_METADATA:
        raise ValueError(f"{dataset} not found on datasets")

    # download the corpus in case the dataset was not downloaded yet
    if not metadata.path.is_dir():
        download(dataset)

    # table dataset
    if metadata.base:

        # read base dataset
        base_datasets = []
        for dataset_name in metadata.base:
            base_datasets += [read(dataset_name)]
        base_dataset = sum(base_datasets)

        # define reader
        reader = metadata.reader(base_dataset)

    # tml dataset
    else:
        reader = metadata.reader(metadata.doc_reader)

    return reader.read(metadata.path)


def download(dataset: str, path: str = None) -> None:
    """ Download corpus.

    This function facilitates the download of temporal annotated corpora.

    For corpora that is available for download the downloader will take care of the download
    and store the data into the \"data\" folder of the current directory. If the data requires
    human action to download a message will be printed to explain the steps necessary to
    get access to the dataset.

    The available datasets are:
        - TimeBank-1.2
        - AQUAINT
        - TimeBankPT
        - TempEval-3
        - MATRES
        - TDDiscourse
        - TimeBank-Dense
        - TCR

    :param str dataset: The name of the dataset to download.
    """

    dataset = dataset.lower().strip()
    metadata = DATASETS_METADATA.get(dataset)

    if metadata is None:
        raise Exception(f"{dataset} not recognized.")

    if path:
        path = pathlib.Path(path)
    else:
        path = pathlib.Path("./data")

    path.mkdir(exist_ok=True, parents=True)

    download_url(metadata.url, path)
