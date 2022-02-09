import pytest

from pathlib import Path

from text2timeline.datasets.utils import TMLHandler


@pytest.fixture
def sample_text():
    text = "WSJ891102-0188 \n = 891102 \n 891102-0188. \n Pacific First Financial Corp. \n 11/02/89 \n WALL STREET " \
           "JOURNAL (J) \n PFFS T.RYL \n TENDER OFFERS, MERGERS, ACQUISITIONS (TNM)\nSAVINGS AND LOANS, THRIFTS, " \
           "CREDIT UNIONS (SAL) \n SEATTLE  \n\n   Pacific First Financial Corp. said shareholders approved its " \
           "acquisition by Royal Trustco Ltd. of Toronto for $27 a share, or $212 million. \n\n   The thrift holding " \
           "company said it expects  to obtain regulatory approval and complete the transaction by year-end."
    return text


@pytest.fixture
def sample_tags():

    timexs = ["t9", "t10"]
    events = ["e1", "e2", "e8", "e4", "e5", "e6", "e19", "e7", "e20"]
    tlinks = ["l1", "l2", "l3", "l4", "l5", "l6"]
    minstances = ["ei79", "ei76", "ei77", "ei73", "ei81", "ei74", "ei80", "ei78", "ei75"]
    slinks = ["l13", "l7", "l8", "l9", "l10", "l11"]
    alinks = ["l12"]

    return events, timexs, tlinks, minstances, slinks, alinks


class TestXMLHandler:

    path = Path(r"/home/hugosousa/Projects/text2timeline/data/sample.tml")
    xml = TMLHandler(path)

    def test_text(self, sample_text):
        assert self.xml.text.strip() == sample_text

    def test_get_tag(self, sample_tags):

        events = [element.attrib['eid'] for element in self.xml.get_tag("EVENT")]
        timexs = [element.attrib['tid'] for element in self.xml.get_tag("TIMEX3")]
        tlinks = [element.attrib['lid'] for element in self.xml.get_tag("TLINK")]
        minstances = [element.attrib['eiid'] for element in self.xml.get_tag("MAKEINSTANCE")]
        slinks = [element.attrib['lid'] for element in self.xml.get_tag("SLINK")]
        alinks = [element.attrib['lid'] for element in self.xml.get_tag("ALINK")]

        tags = events, timexs, tlinks, minstances, slinks, alinks

        assert tags == sample_tags

    def test_endpoints(self):
        """Check if the text limited by inferred endpoints is the same as the element text."""

        tags = self.xml.root.iterfind(".//*")
        for tag in tags:

            if "endpoints" in tag.attrib:
                start, end = tag.attrib["endpoints"]

                text = tag.attrib["text"]
                text_inferred = self.xml.text[start: end]
                assert text_inferred == text
