from text2timeline.datasets.custom import DatasetReader
from text2timeline import toolbox

from pprint import pprint

# load dataset
reader = DatasetReader()
reader.read(['timebank-pt'])

train_valid_set = reader.datasets[0]
train_set, valid_set = train_valid_set.split(0.8)
test_set = reader.datasets[1]

pprint(train_set.tlinks_count())
pprint(valid_set.tlinks_count())
pprint(test_set.tlinks_count())

# reduce dataset tlinks set
doc = train_set.docs[0]

tlink = doc.tlinks[0]

dir(tlink.source)
tlink.source.endpoints
tlink.target.endpoints

pprint(dir(doc))
doc.tokens

test = [tlink for doc in train_set.docs for tlink in doc.tlinks if tlink.interval_relation == 'OVERLAP']

tlink = test[0]

print(~tlink)
print(tlink)

print('hi')
