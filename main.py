from text2timeline.utils import XMLHandler
from text2timeline.readers import TMLDocumentReader
import os


fpath = "data/sample.tml"

xml = XMLHandler(fpath)

print(xml.text.strip())
reader = TMLDocumentReader()

reader.read(fpath)
