import os

import tieval.datasets
from tieval import datasets
from tieval.models.identification import (
    HeidelTime,
    TimexIdentificationBaseline,
    EventIdentificationBaseline
)


class TestHeidelTime:

    def test_predict(self, tmp_path):

        os.chdir(tmp_path)

        te3 = tieval.datasets.read("tempeval_3")

        model = HeidelTime()
        pred = model.predict(te3.test)

        assert len(pred) == len(te3.test)


class TestTimexIdentificationBaseline:

    def test_predict(self, tmp_path):

        os.chdir(tmp_path)

        te3 = tieval.datasets.read("tempeval_3")

        model = TimexIdentificationBaseline()
        pred = model.predict(te3.test)

        assert len(pred) == len(te3.test)

    def test_download(self, tmp_path):

        os.chdir(tmp_path)

        model = TimexIdentificationBaseline()
        assert model.path.is_dir()


class TestEventIdentificationBaseline:

    def test_predict(self, tmp_path):

        os.chdir(tmp_path)

        te3 = tieval.datasets.read("tempeval_3")

        model = EventIdentificationBaseline()
        pred = model.predict(te3.test)

        assert len(pred) == len(te3.test)

    def test_download(self, tmp_path):

        model = EventIdentificationBaseline()
        assert model.path.is_dir()
