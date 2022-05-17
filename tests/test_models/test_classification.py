import os

from tieval.models import CogCompTime2


class TestCogCompTime2:

    def test_download(self, tmp_path):

        os.chdir(tmp_path)

        model = CogCompTime2()  # model is downloaded on call
        assert tmp_path / "resources"
