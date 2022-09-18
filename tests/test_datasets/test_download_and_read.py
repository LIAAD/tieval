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
                assert timex.text != ""
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
                assert timex.text != ""
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
    assert n_timexs == 349

    for doc in meantime.documents:
        for timex in doc.timexs:
            if not timex.is_dct:
                assert timex.text != ""
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
                assert timex.text != ""
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
    dataset_name = "grapheve"
    download(dataset_name)
    data_path = tmp_path / f"data/{dataset_name}"
    assert data_path.is_dir()

    corpus = read(dataset_name)
    assert len(corpus.documents) == 103
    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 0
    n_events = sum(len(doc.events) for doc in corpus.documents)
    assert n_events == 4_298
    n_tlinks = sum(len(doc.tlinks) for doc in corpus.documents)
    assert n_tlinks == 18_204


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


def test_download_and_read_fr_timebank(tmp_path):
    os.chdir(tmp_path)

    download("fr_timebank")
    data_path = tmp_path / "data/fr_timebank"
    assert data_path.is_dir()

    corpus = read("fr_timebank")
    assert len(corpus.documents) == 108

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 533
    n_events = sum(len(doc.events) for doc in corpus.documents)
    assert n_events == 2115
    n_tlinks = sum(len(doc.tlinks) for doc in corpus.documents)
    assert n_tlinks == 2303


def test_download_and_read_tcr(tmp_path):
    os.chdir(tmp_path)

    download("tcr")
    data_path = tmp_path / "data/tcr"
    assert data_path.is_dir()

    corpus = read("tcr")
    assert len(corpus.documents) == 25

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 242
    n_events = sum(len(doc.events) for doc in corpus.documents)
    assert n_events == 1134
    n_tlinks = sum(len(doc.tlinks) for doc in corpus.documents)
    assert n_tlinks == 3515


def test_download_and_read_ancient_time_arabic(tmp_path):
    os.chdir(tmp_path)

    download("ancient_time_arabic")
    data_path = tmp_path / "data/ancient_time_arabic"
    assert data_path.is_dir()

    corpus = read("ancient_time_arabic")
    assert len(corpus.documents) == 5

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 106


def test_download_and_read_ancient_time_dutch(tmp_path):
    os.chdir(tmp_path)

    download("ancient_time_dutch")
    data_path = tmp_path / "data/ancient_time_dutch"
    assert data_path.is_dir()

    corpus = read("ancient_time_dutch")
    assert len(corpus.documents) == 5

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 130


def test_download_and_read_ancient_time_english(tmp_path):
    os.chdir(tmp_path)

    download("ancient_time_english")
    data_path = tmp_path / "data/ancient_time_english"
    assert data_path.is_dir()

    corpus = read("ancient_time_english")
    assert len(corpus.documents) == 5

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 311


def test_download_and_read_ancient_time_french(tmp_path):
    os.chdir(tmp_path)

    download("ancient_time_french")
    data_path = tmp_path / "data/ancient_time_french"
    assert data_path.is_dir()

    corpus = read("ancient_time_french")
    assert len(corpus.documents) == 5

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 290


def test_download_and_read_ancient_time_german(tmp_path):
    os.chdir(tmp_path)

    download("ancient_time_german")
    data_path = tmp_path / "data/ancient_time_german"
    assert data_path.is_dir()

    corpus = read("ancient_time_german")
    assert len(corpus.documents) == 5

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 196


def test_download_and_read_ancient_time_italian(tmp_path):
    os.chdir(tmp_path)

    download("ancient_time_italian")
    data_path = tmp_path / "data/ancient_time_italian"
    assert data_path.is_dir()

    corpus = read("ancient_time_italian")
    assert len(corpus.documents) == 5

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 234


def test_download_and_read_ancient_time_spanish(tmp_path):
    os.chdir(tmp_path)

    download("ancient_time_spanish")
    data_path = tmp_path / "data/ancient_time_spanish"
    assert data_path.is_dir()

    corpus = read("ancient_time_spanish")
    assert len(corpus.documents) == 5

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 217


def test_download_and_read_ancient_time_vietnamese(tmp_path):
    os.chdir(tmp_path)

    download("ancient_time_vietnamese")
    data_path = tmp_path / "data/ancient_time_vietnamese"
    assert data_path.is_dir()

    corpus = read("ancient_time_vietnamese")
    assert len(corpus.documents) == 4

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 120


def test_download_and_read_ph_english(tmp_path):
    os.chdir(tmp_path)

    dataset_name = "ph_english"
    download(dataset_name)
    data_path = tmp_path / f"data/{dataset_name}"
    assert data_path.is_dir()

    corpus = read(dataset_name)
    assert len(corpus.documents) == 24_642

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 254_803


def test_download_and_read_ph_french(tmp_path):
    os.chdir(tmp_path)

    dataset_name = "ph_french"
    download(dataset_name)
    data_path = tmp_path / f"data/{dataset_name}"
    assert data_path.is_dir()

    corpus = read(dataset_name)
    assert len(corpus.documents) == 27_154

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 83_431


def test_download_and_read_ph_german(tmp_path):
    os.chdir(tmp_path)

    dataset_name = "ph_german"
    download(dataset_name)
    data_path = tmp_path / f"data/{dataset_name}"
    assert data_path.is_dir()

    corpus = read(dataset_name)
    assert len(corpus.documents) == 19_095

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 194_043


def test_download_and_read_ph_italian(tmp_path):
    os.chdir(tmp_path)

    dataset_name = "ph_italian"
    download(dataset_name)
    data_path = tmp_path / f"data/{dataset_name}"
    assert data_path.is_dir()

    corpus = read(dataset_name)
    assert len(corpus.documents) == 9_619

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 58_823


def test_download_and_read_ph_portuguese(tmp_path):
    os.chdir(tmp_path)

    dataset_name = "ph_portuguese"
    download(dataset_name)
    data_path = tmp_path / f"data/{dataset_name}"
    assert data_path.is_dir()

    corpus = read(dataset_name)
    assert len(corpus.documents) == 24_293

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 111_810


def test_download_and_read_ph_spanish(tmp_path):
    os.chdir(tmp_path)

    dataset_name = "ph_spanish"
    download(dataset_name)
    data_path = tmp_path / f"data/{dataset_name}"
    assert data_path.is_dir()

    corpus = read(dataset_name)
    assert len(corpus.documents) == 33_266

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 348_011


def test_download_and_read_ph_eventtime(tmp_path):
    os.chdir(tmp_path)

    dataset_name = "eventtime"
    download(dataset_name)
    data_path = tmp_path / f"data/{dataset_name}"
    assert data_path.is_dir()

    corpus = read(dataset_name)
    assert len(corpus.documents) == 36

    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    assert n_timexs == 0
    n_events = sum(len(doc.events) for doc in corpus.documents)
    assert n_events == 1_498
