from text2timeline.datasets.custom import DatasetReader

from pprint import pprint

# load dataset
reader = DatasetReader()
reader.read(['timebank-1.2'])

train_docs = reader.datasets[1]
test_docs = reader.datasets[0]
