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
import pathlib
import requests
import zipfile
import tarfile
import io

from text2timeline.datasets import DATASETS_METADATA

CWD = pathlib.Path.cwd()
DATA_PATH = pathlib.Path("./data")


def _download_repo(metadata):
    repo_name = metadata.repo.split("/")[-1]

    print(f"Downloading {repo_name} from {metadata.repo}")
    os.system(f"git clone {metadata.repo} {DATA_PATH.joinpath(repo_name)}")
    print("Done.")


def _download_url(metadata):

    print(f"Downloading {metadata.name} from {metadata.urls}")

    for url in metadata.urls:

        response = requests.get(url, stream=True)

        if response.ok:

            if ".zip" in url:
                z = zipfile.ZipFile(io.BytesIO(response.content))
                z.extractall(DATA_PATH)

            elif ".tar.gz" in url:
                z = tarfile.open(fileobj=response.raw, mode="r|gz")
                z.extractall(DATA_PATH)

            elif ".txt" in url:

                file = pathlib.Path(metadata.path[0])
                if not file.exists():
                    file.parent.mkdir()

                with open(file, "wb") as f:
                    f.write(response.content)

        else:
            raise Exception(f"Request code: {response.status_code}")

    print("Done.")


def download(dataset: str) -> None:

    dataset = dataset.lower().strip()
    metadata = DATASETS_METADATA.get(dataset)

    # check that the dataset is supported
    if metadata is None:
        raise Exception(f"{dataset} not recognized.")

    # check id data folder exists
    if not DATA_PATH.is_dir():
        os.mkdir(DATA_PATH)

    # download
    if metadata.repo:
        _download_repo(metadata)

    elif metadata.urls:
        _download_url(metadata)

    else:
        print(metadata.download_description)
