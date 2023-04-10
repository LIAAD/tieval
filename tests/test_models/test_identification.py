import os

import tieval.datasets
from tieval.models.identification import (
    HeidelTime,
    TimexIdentificationBaseline,
    EventIdentificationBaseline
)


class TestHeidelTime:

    def test_predict(self, tmp_path):
        os.chdir(tmp_path)

        text = "Thurs August 31st - News today that they are beginning to evacuate the London children tomorrow. Percy is a billeting officer. I can't see that they will be much safer here."
        expected_pred = [[(6, 17), (25, 30), (87, 95)]]

        model = HeidelTime()
        pred = model.predict([text])
        assert pred == expected_pred


class TestTimexIdentificationBaseline:

    def test_predict(self, tmp_path):
        os.chdir(tmp_path)

        te3 = tieval.datasets.read("tempeval_3")

        model = TimexIdentificationBaseline()
        pred = model.predict([te3.test[0]])

        assert len(pred) == 1

    def test_download(self, tmp_path):
        os.chdir(tmp_path)

        model = TimexIdentificationBaseline()
        assert model.path.is_dir()


class TestEventIdentificationBaseline:

    def test_predict(self, tmp_path):
        os.chdir(tmp_path)

        te3 = tieval.datasets.read("tempeval_3")

        model = EventIdentificationBaseline()
        pred = model.predict([te3.test[0]])

        assert len(pred) == 1

    def test_download(self, tmp_path):
        model = EventIdentificationBaseline()
        assert model.path.is_dir()
