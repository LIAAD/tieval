from pathlib import Path

from tieval.datasets.readers import (
    AncientTimeDocumentReader,
    TempEval2DocumentReader,
    TempEval2FrenchDocumentReader,
    TimeBank12DocumentReader,
    TCRDocumentReader,
    TimeBankPTDocumentReader,
    TempEval3DocumentReader,
    MeanTimeDocumentReader,
    KRAUTSDocumentReader,
    WikiWarsDocumentReader,
    ProfessorHeidelTimeDocumentReader
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

        for timex in doc.timexs:
            if not timex.is_dct:
                s, e = timex.offsets
                assert doc.text[s:e] == timex.text


class TestKRAUTSDocumentReader:
    reader = KRAUTSDocumentReader(DATA_PATH / "krauts.tml")

    def test_read(self):
        doc = self.reader.read()

        for timex in doc.timexs:
            if not timex.is_dct:
                s, e = timex.offsets
                assert doc.text[s:e] == timex.text


class TestWikiWarsDocumentReader:
    reader = WikiWarsDocumentReader(DATA_PATH / "wikiwars_de.xml")

    def test_read(self):
        doc = self.reader.read()

        for timex in doc.timexs:
            if not timex.is_dct:
                s, e = timex.offsets
                assert doc.text[s:e] == timex.text


class TestAncientTimeDocumentReader:
    reader = AncientTimeDocumentReader(DATA_PATH / "ancient_time.tml")

    def test_read(self):
        doc = self.reader.read()

        for timex in doc.timexs:
            if not timex.is_dct:
                s, e = timex.offsets
                assert doc.text[s:e] == timex.text


class TestProfessorHeidelTimeDocumentReader:
    reader = ProfessorHeidelTimeDocumentReader(DATA_PATH / "professor_heideltime.json")

    def test_read(self):
        doc = self.reader.read()

        for timex in doc.timexs:
            if not timex.is_dct:
                s, e = timex.offsets
                assert doc.text[s:e] == timex.text
