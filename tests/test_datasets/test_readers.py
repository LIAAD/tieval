from pathlib import Path

from tieval.datasets.readers import (
    MeanTimeDocumentReader,
    TempEval3DocumentReader,
    GraphEveDocumentReader,
    TempEval2DocumentReader,
    TempEval2FrenchDocumentReader,
    TimeBank12DocumentReader,
    TCRDocumentReader,
    TimeBankPTDocumentReader
)

from tieval.datasets.readers import JSONDatasetReader

DATA_PATH = Path(__file__).parent.parent / "files"


class TestTempEval2DocumentReader:
    reader = TempEval2DocumentReader(DATA_PATH / "tempeval_2_italian.json")

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
