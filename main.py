from text2story.narrative import Document
from text2story.narrative import TLink

from pprint import pprint

reference = {
    ("A", "B", "BEFORE"),
    ("B", "C", "IS_INCLUDED"),
    ("D", "C", "INCLUDES"),
    ("E", "D", "CONTAINS"),
    ("F", "E", "AFTER"),
    ("G", "H", "BEGINS-ON"),
    ("I", "G", "BEFORE"),
    ("J", "K", "IBEFORE"),
    ("K", "L", "BEGUN_BY"),
    ("L", "K", "BEGINS"),  # duplicate
}

infered = {
    ('A', 'B', 'before'),
    ('A', 'F', 'before'),
    ('B', 'F', 'before'),
    ('C', 'B', 'includes'),
    ('C', 'F', 'before'),
    ('D', 'B', 'includes'),
    ('D', 'C', 'includes'),
    ('D', 'F', 'before'),
    ('E', 'B', 'includes'),
    ('E', 'C', 'includes'),
    ('E', 'D', 'includes'),
    ('E', 'F', 'before'),
    ('G', 'H', 'simultaneous-start'),
    ('I', 'G', 'before'),
    ('I', 'H', 'before'),
}

tlinks = {f'l{idx}': TLink(scr, tgt, interval_relation=rel) for idx, (scr, tgt, rel) in enumerate(reference)}

tlinks_tc = Document('empty.txt').temporal_closure(tlinks)

pprint(tlinks_tc)
pprint([(tlink.source, tlink.target, tlink.interval_relation) for lid,  tlink in tlinks_tc.items()])
pprint([(tlink.source, tlink.target, tlink.interval_relation) for lid,  tlink in tlinks.items()])
