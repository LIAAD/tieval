from tieval import datasets
from tieval.models.identification import (
    HeidelTime,
    TimexIdentificationBaseline,
    EventIdentificationBaseline
)


class TestHeidelTime:

    def test_predict(self, tmp_path):

        datasets.download("tempeval_3", tmp_path)
        te3 = datasets.read("tempeval_3")

        model = HeidelTime()
        pred = model.predict(te3.test)

        assert len(pred) == len(te3.test)


class TestTimexIdentificationBaseline:

    def test_predict(self, tmp_path):

        datasets.download("tempeval_3", tmp_path)
        te3 = datasets.read("tempeval_3")

        model = TimexIdentificationBaseline()
        pred = model.predict(te3.test)

        assert len(pred) == len(te3.test)

    def test_download(self, tmp_path):
        model = TimexIdentificationBaseline(path=tmp_path / "models")
        assert model.path.is_dir()


class TestEventIdentificationBaseline:

    def test_predict(self, tmp_path):

        datasets.download("tempeval_3", tmp_path)
        te3 = datasets.read("tempeval_3")

        model = EventIdentificationBaseline(path=tmp_path / "models")
        pred = model.predict(te3.test)

        assert len(pred) == len(te3.test)

    def test_download(self, tmp_path):

        model = EventIdentificationBaseline(path=tmp_path / "models")
        assert model.path.is_dir()
