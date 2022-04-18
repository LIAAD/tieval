import os
import pathlib

from tieval import datasets
from tieval.datasets.datasets import read


# dowload the supported datasets
# for ds_name in datasets.SUPPORTED_DATASETS:
#     datasets.download()


def test_read_aquaint(tmp_path):
    # datasets.download("aquaint")
    assert read("aquaint")


def test_read_eventtime():
    assert read("eventtime")


def test_read_grapheve():
    assert read("grapheve")


def test_read_matres():
    assert read("matres")


def test_read_meantime_english():
    assert read("meantime_english")


def test_read_meantime_spanish():
    assert read("meantime_spanish")


def test_read_meantime_dutch():
    assert read("meantime_dutch")


def test_read_meantime_italian():
    assert read("meantime_italian")


def test_read_platinum():
    assert read("platinum")


def test_read_tcr():
    assert read("tcr")


def test_read_tddiscourse():
    assert read("tddiscourse")


def test_read_tempeval_2_chinese():
    assert read("tempeval_2_chinese")


def test_read_tempeval_2_english():
    assert read("tempeval_2_english")


def test_read_tempeval_2_french():
    assert read("tempeval_2_french")


def test_read_tempeval_2_italian():
    assert read("tempeval_2_italian")


def test_read_tempeval_2_korean():
    assert read("tempeval_2_korean")


def test_read_tempeval_2_spanish():
    assert read("tempeval_2_spanish")


def test_read_tempeval_3():
    assert read("tempeval_3")


def test_read_timebank_12():
    assert read("timebank_1.2")


def test_read_timebank_dense():
    assert read("timebank_dense")


def test_read_timebankpt():
    assert read("timebankpt")


def test_read_timebank():
    assert read("timebank")


def test_read_traint3():
    assert read("traint3")


def test_read_uds_t():
    assert read("uds_t")
