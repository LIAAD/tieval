from text2timeline.datasets.custom import DatasetReader

from pprint import pprint

# load dataset
reader = DatasetReader()
reader.read('timebank')

pprint(reader.datasets)

reader.datasets[0]