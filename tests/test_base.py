
from pathlib import Path

from tieval.base import Document, Sentence, Text
from tieval.datasets.readers import TempEval3DocumentReader


DATA_PATH = Path(__file__).parent / "files"


class TestDocument:

    doc = TempEval3DocumentReader(DATA_PATH / "tempeval_3.tml").read()


class TestSentence:

    sent = Sentence(
        content="Hi, my name is.",
        span=[100, 115]
    )

    def test_tokens(self):
        tkns = self.sent.tokens
        assert len(tkns) == 6
