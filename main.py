from text2timeline.utils import XMLHandler
from text2timeline.readers import TMLDocumentReader
import os


fpath = "data/sample.tml"
reader = TMLDocumentReader()
reader.read(fpath)
