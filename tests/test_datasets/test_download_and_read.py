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
    assert len(te2.documents) == 83


def test_download_and_read_te2_spanish(tmp_path):

    os.chdir(tmp_path)

    download("tempeval_2_spanish")
    data_path = tmp_path / "data/tempeval_2_spanish"
    assert data_path.is_dir()

    te2 = read("tempeval_2_spanish")
    assert len(te2.documents) == 210


def test_download_and_read_te2_italian(tmp_path):

    os.chdir(tmp_path)

    download("tempeval_2_italian")
    data_path = tmp_path / "data/tempeval_2_italian"
    assert data_path.is_dir()

    te2 = read("tempeval_2_italian")
    assert len(te2.documents) == 64

    test_docs = set(doc.name for doc in te2.test)
    train_docs = set(doc.name for doc in te2.train)
    assert len(test_docs & train_docs) == 0
    assert len(train_docs & test_docs) == 0


def test_download_and_read_meantime_italian(tmp_path):

    os.chdir(tmp_path)

    download("meantime_italian")
    data_path = tmp_path / "data/meantime_italian"
    assert data_path.is_dir()

    meantime = read("meantime_italian")
    assert len(meantime.documents) == 120

    n_timexs = sum(len(doc.timexs) for doc in meantime.documents)
    assert n_timexs == 338

    for doc in meantime.documents:
        for timex in doc.timexs:
            if not timex.is_dct:
                s, e = timex.endpoints
                assert timex.text == doc.text[s: e]


def test_download_and_read_meantime_dutch(tmp_path):

    os.chdir(tmp_path)

    download("meantime_dutch")
    data_path = tmp_path / "data/meantime_dutch"
    assert data_path.is_dir()

    meantime = read("meantime_dutch")
    assert len(meantime.documents) == 120

    n_timexs = sum(len(doc.timexs) for doc in meantime.documents)
    assert n_timexs == 346

    for doc in meantime.documents:
        for timex in doc.timexs:
            if not timex.is_dct:
                s, e = timex.endpoints
                assert timex.text == doc.text[s: e]


def test_download_and_read_meantime_english(tmp_path):

    os.chdir(tmp_path)

    download("meantime_english")
    data_path = tmp_path / "data/meantime_english"
    assert data_path.is_dir()

    meantime = read("meantime_english")
    assert len(meantime.documents) == 120

    n_timexs = sum(len(doc.timexs) for doc in meantime.documents)
    assert n_timexs == 350

    for doc in meantime.documents:
        for timex in doc.timexs:
            if not timex.is_dct:
                s, e = timex.endpoints
                assert timex.text == doc.text[s: e]


def test_download_and_read_meantime_spanish(tmp_path):

    os.chdir(tmp_path)

    download("meantime_spanish")
    data_path = tmp_path / "data/meantime_spanish"
    assert data_path.is_dir()

    meantime = read("meantime_spanish")
    assert len(meantime.documents) == 120

    n_timexs = sum(len(doc.timexs) for doc in meantime.documents)
    assert n_timexs == 344

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


def test_download_and_read_krauts_diezeit(tmp_path):

    os.chdir(tmp_path)

    download("krauts_diezeit")
    data_path = tmp_path / "data/krauts_diezeit"
    assert data_path.is_dir()

    krauts = read("krauts_diezeit")
    assert len(krauts.documents) == 50


def test_download_and_read_krauts_dolomiten_42(tmp_path):

    os.chdir(tmp_path)

    download("krauts_dolomiten_42")
    data_path = tmp_path / "data/krauts_dolomiten_42"
    assert data_path.is_dir()

    krauts = read("krauts_dolomiten_42")
    assert len(krauts.documents) == 42


def test_download_and_read_krauts_dolomiten_100(tmp_path):

    os.chdir(tmp_path)

    download("krauts_dolomiten_100")
    data_path = tmp_path / "data/krauts_dolomiten_100"
    assert data_path.is_dir()

    krauts = read("krauts_dolomiten_100")
    assert len(krauts.documents) == 100


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


def test_download_and_read_spanish_timebank(tmp_path):

    os.chdir(tmp_path)

    download("spanish_timebank")
    data_path = tmp_path / "data/spanish_timebank"
    assert data_path.is_dir()

    stb = read("spanish_timebank")
    assert len(stb.documents) == 210

    test_docs = set(doc.name for doc in stb.test)
    train_docs = set(doc.name for doc in stb.train)
    assert len(test_docs & train_docs) == 0
    assert len(train_docs & test_docs) == 0


def test_download_and_read_narrative_container(tmp_path):

    os.chdir(tmp_path)
    corpus_name = "narrative_container"

    download(corpus_name)
    data_path = tmp_path / f"data/{corpus_name}"
    assert data_path.is_dir()

    data = read(corpus_name)
    assert len(data.documents) == 63

    test_docs = set(doc.name for doc in data.test)
    train_docs = set(doc.name for doc in data.train)
    assert len(test_docs & train_docs) == 0
    assert len(train_docs & test_docs) == 0


def test_download_and_read_wikiwars(tmp_path):

    os.chdir(tmp_path)
    corpus_name = "wikiwars"

    download(corpus_name)
    data_path = tmp_path / f"data/{corpus_name}"
    assert data_path.is_dir()

    data = read(corpus_name)
    assert len(data.documents) == 22


def test_download_and_read_wikiwars_de(tmp_path):

    os.chdir(tmp_path)
    corpus_name = "wikiwars_de"

    download(corpus_name)
    data_path = tmp_path / f"data/{corpus_name}"
    assert data_path.is_dir()

    data = read(corpus_name)
    assert len(data.documents) == 22
