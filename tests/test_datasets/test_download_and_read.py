from pathlib import Path

from tieval.datasets import download, read


def _test_download_and_read(corpus: str, path: Path):
    download(corpus, path)
    corpus_path = path / corpus
    assert corpus_path.is_dir()

    corpus = read(corpus, corpus_path)

    train_docs = set(doc.text for doc in corpus.train)
    test_docs = set(doc.text for doc in corpus.test)
    assert len(test_docs & train_docs) == 0
    assert len(train_docs & test_docs) == 0

    n_docs = len(corpus.documents)
    n_timexs = sum(len(doc.timexs) for doc in corpus.documents)
    n_events = sum(len(doc.events) for doc in corpus.documents)
    n_tlinks = sum(len(doc.tlinks) for doc in corpus.documents)
    return n_docs, n_events, n_timexs, n_tlinks


def test_download_and_read_te3(tmp_path):
    corpus = "tempeval_3"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 275
    assert n_events == 11_780
    assert n_timexs == 2_223
    assert n_tlinks == 11_881


def test_download_and_read_te2_french(tmp_path):
    corpus = "tempeval_2_french"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 83
    assert n_events == 1301
    assert n_timexs == 367
    assert n_tlinks == 372


def test_download_and_read_te2_spanish(tmp_path):
    corpus = "tempeval_2_spanish"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 210
    assert n_events == 12_384
    assert n_timexs == 1502
    assert n_tlinks == 13_304


def test_download_and_read_te2_italian(tmp_path):
    corpus = "tempeval_2_italian"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 64
    assert n_events == 5_377
    assert n_timexs == 653
    assert n_tlinks == 6_884


def test_download_and_read_meantime_italian(tmp_path):
    corpus = "meantime_italian"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 120
    assert n_events == 1_980
    assert n_timexs == 338
    assert n_tlinks == 1_675


def test_download_and_read_meantime_dutch(tmp_path):
    corpus = "meantime_dutch"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 120
    assert n_events == 1_346
    assert n_timexs == 346
    assert n_tlinks == 1_487


def test_download_and_read_meantime_english(tmp_path):
    corpus = "meantime_english"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 120
    assert n_events == 1882
    assert n_timexs == 349
    assert n_tlinks == 1753


def test_download_and_read_meantime_spanish(tmp_path):
    corpus = "meantime_spanish"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 120
    assert n_events == 2000
    assert n_timexs == 344
    assert n_tlinks == 1975


def test_download_and_read_krauts(tmp_path):
    corpus = "krauts"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 192
    assert n_events == 0
    assert n_timexs == 1282
    assert n_tlinks == 0


def test_download_and_read_krauts_diezeit(tmp_path):
    corpus = "krauts_diezeit"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 50
    assert n_events == 0
    assert n_timexs == 553
    assert n_tlinks == 0


def test_download_and_read_krauts_dolomiten_42(tmp_path):
    corpus = "krauts_dolomiten_42"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 42
    assert n_events == 0
    assert n_timexs == 228
    assert n_tlinks == 0


def test_download_and_read_krauts_dolomiten_100(tmp_path):
    corpus = "krauts_dolomiten_100"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 100
    assert n_events == 0
    assert n_timexs == 501
    assert n_tlinks == 0


def test_download_and_read_matres(tmp_path):
    corpus = "matres"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 274
    assert n_events == 6065
    assert n_timexs == 0
    assert n_tlinks == 13504


def test_download_and_read_grapheve(tmp_path):
    corpus = "grapheve"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 103
    assert n_events == 4298
    assert n_timexs == 0
    assert n_tlinks == 18204


def test_download_and_read_spanish_timebank(tmp_path):
    corpus = "spanish_timebank"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 210
    assert n_events == 12384
    assert n_timexs == 1532
    assert n_tlinks == 21107


def test_download_and_read_narrative_container(tmp_path):
    corpus = "narrative_container"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 63
    assert n_events == 3559
    assert n_timexs == 439
    assert n_tlinks == 737


def test_download_and_read_wikiwars(tmp_path):
    corpus = "wikiwars"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 22
    assert n_events == 0
    assert n_timexs == 2_662
    assert n_tlinks == 0


def test_download_and_read_wikiwars_de(tmp_path):
    corpus = "wikiwars_de"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 22
    assert n_events == 0
    assert n_timexs == 2_239
    assert n_tlinks == 0


def test_download_and_read_fr_timebank(tmp_path):
    corpus = "fr_timebank"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 108
    assert n_timexs == 533
    assert n_events == 2115
    assert n_tlinks == 2303


def test_download_and_read_tcr(tmp_path):
    corpus = "tcr"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 25
    assert n_timexs == 242
    assert n_events == 1134
    assert n_tlinks == 3515


def test_download_and_read_ancient_time_arabic(tmp_path):
    corpus = "ancient_time_arabic"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 5
    assert n_timexs == 106
    assert n_events == 0
    assert n_tlinks == 0


def test_download_and_read_ancient_time_dutch(tmp_path):
    corpus = "ancient_time_dutch"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 5
    assert n_timexs == 130
    assert n_events == 0
    assert n_tlinks == 0


def test_download_and_read_ancient_time_english(tmp_path):
    corpus = "ancient_time_english"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 5
    assert n_timexs == 311


def test_download_and_read_ancient_time_french(tmp_path):
    corpus = "ancient_time_french"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 5
    assert n_timexs == 290
    assert n_events == 0
    assert n_tlinks == 0


def test_download_and_read_ancient_time_german(tmp_path):
    corpus = "ancient_time_german"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 5
    assert n_timexs == 196
    assert n_events == 0
    assert n_tlinks == 0


def test_download_and_read_ancient_time_italian(tmp_path):
    corpus = "ancient_time_italian"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 5
    assert n_timexs == 234
    assert n_events == 0
    assert n_tlinks == 0


def test_download_and_read_ancient_time_spanish(tmp_path):
    corpus = "ancient_time_spanish"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 5
    assert n_timexs == 217
    assert n_events == 0
    assert n_tlinks == 0


def test_download_and_read_ancient_time_vietnamese(tmp_path):
    corpus = "ancient_time_vietnamese"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 4
    assert n_timexs == 120
    assert n_events == 0
    assert n_tlinks == 0


def test_download_and_read_ph_english(tmp_path):
    corpus = "ph_english"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 24_642
    assert n_timexs == 254_803
    assert n_events == 0
    assert n_tlinks == 0


def test_download_and_read_ph_french(tmp_path):
    corpus = "ph_french"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 27_154
    assert n_timexs == 83_431
    assert n_events == 0
    assert n_tlinks == 0


def test_download_and_read_ph_german(tmp_path):
    corpus = "ph_german"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 19_095
    assert n_timexs == 194_043
    assert n_events == 0
    assert n_tlinks == 0


def test_download_and_read_ph_italian(tmp_path):
    corpus = "ph_italian"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 9_619
    assert n_timexs == 58_823
    assert n_events == 0
    assert n_tlinks == 0


def test_download_and_read_ph_portuguese(tmp_path):
    corpus = "ph_portuguese"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 24_293
    assert n_timexs == 111_810
    assert n_events == 0
    assert n_tlinks == 0


def test_download_and_read_ph_spanish(tmp_path):
    corpus = "ph_spanish"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 33_266
    assert n_timexs == 348_011
    assert n_events == 0
    assert n_tlinks == 0


def test_download_and_read_eventtime(tmp_path):
    corpus = "eventtime"
    n_docs, n_events, n_timexs, n_tlinks = _test_download_and_read(corpus, tmp_path)
    assert n_docs == 36
    assert n_timexs == 0
    assert n_events == 1_498
    assert n_tlinks == 0
