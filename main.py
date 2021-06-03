from text2timeline.datasets.custom import DatasetReader

from pprint import pprint

# load dataset
reader = DatasetReader()
reader.read(['timebank', 'timebank-dense'])


pprint(reader.datasets)

tb = reader.datasets[0]
tbdense = reader.datasets[1]

len([tlink for doc in tb.docs for tlink in doc.tlinks])
len([tlink for doc in tbdense.docs for tlink in doc.tlinks])