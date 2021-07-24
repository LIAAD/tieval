from text2timeline.base import Event, Timex, TLink

source = Event({
    'eid': 'e1',
    'class': 'OCCURRENCE',
    'text': 'expansion',
    'endpoints': (82, 91),
    'eventID': 'e1',
    'eiid': 'ei50001',
    'tense': 'NONE',
    'aspect': 'NONE',
    'polarity': 'POS',
    'pos': 'NOUN'
})

target = Timex({
    'tid': 't0',
    'type': 'DATE',
    'value': '1998-03-03',
    'temporalFunction': 'false',
    'functionInDocument': 'CREATION_TIME'
})

tlink = TLink(
    id='l70',
    source=source,
    target=target,
    relation=[(1, '<', 0)]
)

print(tlink.relation)
print(tlink.point_relation)

tlink.__dict__

dir(tlink)
print(tlink.relation)

tlink.point_relation