import os

import pytest

from tieval.datasets.datasets import read


os.chdir("../../")


def test_read_aquaint():
    aquaint = read("aquaint")


def test_read_eventtime():
    eventtime = read("eventtime")


def test_read_grapheve():
    grapheve = read("grapheve")


def test_read_matres():
    matres = read("matres")


def test_read_mctaco():
    mctaco = read("mctaco")


def test_read_meantime_english():
    meantime_english = read("meantime_english")


def test_read_meantime_spanish():
    meantime_spanish = read("meantime_spanish")


def test_read_meantime_dutch():
    meantime_dutch = read("meantime_dutch")


def test_read_meantime_italian():
    meantime_italian = read("meantime_italian")


def test_read_platinum():
    platinum = read("platinum")


def test_read_tcr():
    tcr = read("tcr")


def test_read_tddiscourse():
    tddiscourse = read("tddiscourse")


def test_read_tempeval_2_chinese():
    tempeval_2_chinese = read("tempeval_2_chinese")


def test_read_tempeval_2_english():
    tempeval_2_english = read("tempeval_2_english")


def test_read_tempeval_2_french():
    tempeval_2_french = read("tempeval_2_french")


def test_read_tempeval_2_italian():
    tempeval_2_italian = read("tempeval_2_italian")


def test_read_tempeval_2_korean():
    tempeval_2_korean = read("tempeval_2_korean")


def test_read_tempeval_2_spanish():
    tempeval_2_spanish = read("tempeval_2_spanish")


def test_read_tempeval_3():
    tempeval_3 = read("tempeval_3")


def test_read_tempqa():
    tempqa = read("tempqa")


def test_read_tempquestions():
    tempquestions = read("tempquestions")


def test_read_timebank_12():
    timebank_12 = read("timebank_1.2")


def test_read_timebank_dense():
    timebank_dense = read("timebank_dense")


def test_read_timebankpt():
    timebankpt = read("timebankpt")


def test_read_timebank():
    timebank = read("timebank")


def test_read_torque():
    torque = read("torque")


def test_read_traint3():
    traint3 = read("traint3")


def test_read_uds_t():
    uds_t = read("uds_t")
