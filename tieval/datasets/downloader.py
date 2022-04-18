"""
Description:
------------
    This module facilitates the download of temporal annotated corpora.

    For corpora that is available for download the downloader will take care of the download
    and store the data into the \"data\" folder of the current directory. If the data requires
    human action to download a message will be printed to explain the steps necessary to
    get access to the dataset.

Datasets:
---------
    - TimeBank-1.2
    - AQUAINT
    - TimeBankPT
    - TempEval-3
    - MATRES
    - TDDiscourse
    - TimeBank-Dense"
    - TCR

"""

import os
import requests
import zipfile
import io

from tieval.datasets import DATASETS_METADATA
from tieval import DATA_PATH


def _download_url(url: str) -> None:
    """Download dataset from url.

    Parameters
    ----------
    url: str
        The name of the dataset to download.
    """

    print(f"Downloading from {url}")

    response = requests.get(url, stream=True)

    if response.ok:

        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(DATA_PATH)
        print("Done.")

    else:
        raise Exception(f"Request code: {response.status_code}")


def download(dataset: str) -> None:
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
    dataset: str
        The name of the dataset to download.
    """

    dataset = dataset.lower().strip()
    metadata = DATASETS_METADATA.get(dataset)

    # check that the dataset is supported
    if metadata is None:
        raise Exception(f"{dataset} not recognized.")

    # check if DATA_PATH folder exists
    if not DATA_PATH.is_dir():
        os.mkdir(DATA_PATH)

    # check if it has already been downloaded
    if metadata.path.is_dir():
        print(f"Dataset {dataset} was already on {DATA_PATH}.")
        return

    _download_url(metadata.url)
