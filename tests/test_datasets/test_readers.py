from pathlib import Path

from tieval.datasets.readers import \
    MeanTimeDocumentReader, \
    TempEval3DocumentReader, \
    GraphEveDocumentReader, \
    TempEval2DocumentReader, \
    TempEval2FrenchDocumentReader, \
    TimeBank12DocumentReader

from tieval.datasets.readers import JSONDatasetReader

DATA_PATH = Path("../../data")


class TestTempEval2DocumentReader:
    reader = TempEval2DocumentReader(DATA_PATH / "tempeval_2_italian/train/cs.morph015.json")

    def test_read(self):
        self.reader.read()


class TestTimeBank12DocumentReader:
    reader = TimeBank12DocumentReader(DATA_PATH / "timebank_1.2/train/wsj_0685.tml")

    def test_read(self):
        self.reader.read()


class TestTempEval2FrenchDocumentReader:
    reader = TempEval2FrenchDocumentReader(DATA_PATH / "tempeval_2_french/train/baldwin_frratrain_28.xml")

    def test_read(self):
        self.reader.read()


class TestJSONDatasetReader:

    def test_read(self):

        reader = JSONDatasetReader(TempEval2DocumentReader)
        te2 = reader.read(DATA_PATH / "tempeval_2_chinese")

        reader = JSONDatasetReader(TempEval2DocumentReader)
        te2 = reader.read(DATA_PATH / "tempeval_2_english")

        reader = JSONDatasetReader(TempEval2DocumentReader)
        te2 = reader.read(DATA_PATH / "tempeval_2_italian")

        reader = JSONDatasetReader(TempEval2DocumentReader)
        te2 = reader.read(DATA_PATH / "tempeval_2_korean")

        reader = JSONDatasetReader(TempEval2DocumentReader)
        te2 = reader.read(DATA_PATH / "tempeval_2_spanish")
