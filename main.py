from text2story.datasets.custom import Dataset
from text2story.datasets.custom import DatasetReader

from pprint import pprint

# load dataset
reader = DatasetReader()
reader.read(['timebank-pt'])

