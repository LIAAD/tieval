
from pathlib import Path

from tieval.base import Sentence
from tieval.links import TLink
from tieval.entities import Entity
from tieval.datasets.readers import TempEval3DocumentReader


DATA_PATH = Path(__file__).parent / "files"


class TestDocument:

    doc = TempEval3DocumentReader(DATA_PATH / "tempeval_3.tml").read()

    def test_print(self):
        print(self.doc)
        
    def test_temporal_closure(self):
        tlinks = self.doc.temporal_closure
        tlink = next(iter(tlinks))
        assert isinstance(tlink, TLink)
        assert isinstance(tlink.source, Entity)
        assert isinstance(tlink.target, Entity)
        
    def test_temporal_closure_length(self):
        closure_tlinks = self.doc.temporal_closure
        assert len(closure_tlinks) == 291


class TestSentence:

    sent = Sentence(
        content="Hi, my name is.",
        offsets=(100, 115)
    )

    def test_tokens(self):
        tkns = self.sent.tokens
        assert len(tkns) == 6
