import os
import pytest

from tieval.datasets import download, read


def test_download_and_read_te3(tmp_path):

    os.chdir(tmp_path)

    download("tempeval_3")
    data_path = tmp_path / "data/tempeval_3"
    assert data_path.is_dir()

    te3 = read("tempeval_3")
    assert len(te3.documents) == 275  # number of documents


def test_download_and_read_matres(tmp_path):

    os.chdir(tmp_path)

    download("matres")
    data_path = tmp_path / "data/matres"
    assert data_path.is_dir()

    matres = read("matres")
    assert len(matres.documents) == 274  # number of documents


def test_download_and_read_grapheve(tmp_path):

    os.chdir(tmp_path)

    download("grapheve")
    data_path = tmp_path / "data/grapheve"
    assert data_path.is_dir()
