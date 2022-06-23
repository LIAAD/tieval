from pathlib import Path

from tieval.datasets.readers import (
    TempEval2DocumentReader,
    TempEval2FrenchDocumentReader,
    TimeBank12DocumentReader,
    TCRDocumentReader,
    TimeBankPTDocumentReader,
    TempEval3DocumentReader,
    MeanTimeDocumentReader
)

DATA_PATH = Path(__file__).parent.parent / "files"


class TestTempEval2DocumentReader:
    reader = TempEval2DocumentReader(DATA_PATH / "tempeval_2_italian.json")

    def test_read(self):
        self.reader.read()


class TestTempEval3DocumentReader:
    reader = TempEval3DocumentReader(DATA_PATH / "tempeval_3.tml")

    def test_read(self):
        self.reader.read()


class TestTimeBank12DocumentReader:
    reader = TimeBank12DocumentReader(DATA_PATH / "timebank12.tml")

    def test_read(self):
        self.reader.read()


class TestTCRDocumentReader:
    reader = TCRDocumentReader(DATA_PATH / "tcr.tml")

    def test_read(self):
        self.reader.read()


class TestTimeBankPTDocumentReader:
    reader = TimeBankPTDocumentReader(DATA_PATH / "timebankpt.tml")

    def test_read(self):
        self.reader.read()


class TestTempEval2FrenchDocumentReader:
    reader = TempEval2FrenchDocumentReader(DATA_PATH / "tempeval_2_french.xml")

    def test_read(self):
        self.reader.read()


class TestMeanTimeDocumentReader:
    reader = MeanTimeDocumentReader(DATA_PATH / "meantime.xml")

    def test_read(self):
        doc = self.reader.read()
        tlink = next(iter(doc.tlinks))
        tlink.source
        tlink.target
