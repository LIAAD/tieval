"""
Description:
------------
    This module facilitates the download of temporal annotated corpora.

    For corpora that is available for download the downloader will take care of the download
    and store the data into the \"data\" folder of the current directory. If the data requires
    human action to download a message will be printed to explain the steps necessary to
    get access to the dataset.

"""

import os
import pathlib

from tieval.datasets import DATASETS_METADATA
from tieval import utils


def download(dataset: str, path: str = "./data") -> None:
    """ Download corpus.

    The available datasets are:
        - TimeBank-1.2
        - AQUAINT
        - TimeBankPT
        - TempEval-3
        - MATRES
        - TDDiscourse
        - TimeBank-Dense
        - TCR

    Parameters
    ----------
    dataset : str
        The name of the dataset to download.
    path : str = "data"
        Path to store the dataset.
    """

    dataset = dataset.lower().strip()
    metadata = DATASETS_METADATA.get(dataset)

    # check that the dataset is supported
    if metadata is None:
        raise Exception(f"{dataset} not recognized.")

    # check if path folder exists
    path = pathlib.Path(path)
    if not path.is_dir():
        os.mkdir(path)

    # check if it has already been downloaded
    metadata.data_path = path
    if metadata.path.is_dir():
        print(f"Dataset {dataset} was already on {path}.")
        return

    utils._download_url(metadata.url, path)
