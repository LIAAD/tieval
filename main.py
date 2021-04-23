from text2story.datasets.custom import Dataset

dataset = Dataset('timebank-pt')

dataset.read()

train = dataset.docs['train']

doc = train[0]

print(doc)

doc.events
doc.timexs
doc.tlinks

tlink = doc.tlinks[0]

tlink.source
