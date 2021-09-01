from text2timeline.utils import XMLHandler
from text2timeline.readers import TMLDocumentReader
from text2timeline.datasets import load_dataset
import os


# reader = TMLDocumentReader()
# reader.read("data/TempEval-3/Train/TBAQ-cleaned/TimeBank/ABC19980304.1830.1636.tml")

tb = load_dataset("timebank")
