import os

from tieval.datasets import download, read


def test_download_and_read_te3(tmp_path):

    os.chdir(tmp_path)

    download("tempeval_3")
    data_path = tmp_path / "data/tempeval_3"
    assert data_path.is_dir()

    te3 = read("tempeval_3")
    assert len(te3.documents) == 275  # number of documents


def test_download_and_read_te2_french(tmp_path):

    os.chdir(tmp_path)

    download("tempeval_2_french")
    data_path = tmp_path / "data/tempeval_2_french"
    assert data_path.is_dir()

    te2 = read("tempeval_2_french")
    assert len(te2.documents) == 83  # number of documents

    for doc in te2.documents:
        print(doc.dct.value, doc.name)


def test_download_and_read_te2_spanish(tmp_path):

    os.chdir(tmp_path)

    download("tempeval_2_spanish")
    data_path = tmp_path / "data/tempeval_2_spanish"
    assert data_path.is_dir()

    te2 = read("tempeval_2_spanish")
    assert len(te2.documents) == 193


def test_download_and_read_meantime_italian(tmp_path):

    os.chdir(tmp_path)

    download("meantime_italian")
    data_path = tmp_path / "data/meantime_italian"
    assert data_path.is_dir()

    meantime = read("meantime_italian")
    assert len(meantime.documents) == 120

    for doc in meantime.documents:
        for timex in doc.timexs:
            if not timex.is_dct:
                s, e = timex.endpoints
                print(timex.type)
                print(timex.text)
                print()


def test_download_and_read_meantime_spanish(tmp_path):

    os.chdir(tmp_path)

    download("meantime_spanish")
    data_path = tmp_path / "data/meantime_spanish"
    assert data_path.is_dir()

    meantime = read("meantime_spanish")
    assert len(meantime.documents) == 120

    for doc in meantime.documents:
        for timex in doc.timexs:
            if timex.endpoints:
                s, e = timex.endpoints
                assert timex.text == doc.text[s: e]


def test_download_and_read_krauts(tmp_path):

    os.chdir(tmp_path)

    download("krauts")
    data_path = tmp_path / "data/krauts"
    assert data_path.is_dir()

    krauts = read("krauts")
    assert len(krauts.documents) == 192

    for doc in krauts.documents:
        for timex in doc.timexs:
            if timex.endpoints:
                s, e = timex.endpoints
                assert timex.text == doc.text[s: e]


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
