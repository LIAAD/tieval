
from pathlib import Path

from tieval.base import Document
from tieval.datasets.readers import TempEval3DocumentReader


DATA_PATH = Path(__file__).parent / "files"


class TestDocument:

    doc = TempEval3DocumentReader(DATA_PATH / "tempeval_3.tml").read()

    def test_sentences(self):
        sents = self.doc.sentences
        assert len(sents) == 2
