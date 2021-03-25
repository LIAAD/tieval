from text2story.narrative import Document
from text2story.narrative import TLink

from pprint import pprint
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(annotation, title):
    from_ = [scr for scr, _, _ in annotation]
    to = [tgt for _, tgt, _ in annotation]

    df = pd.DataFrame({'from': from_, 'to': to})

    G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())

    # Make the graph
    nx.draw(G, with_labels=True, node_size=1000, alpha=0.3, arrows=True)
    plt.title(title)


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


"""
plt.figure(figsize=(10, 5))
plt.subplot(121)
plot_graph(reference, 'Reference')
plt.subplot(122)
plot_graph(infered, 'Closure')
plt.show()
"""
reference = {
    ("A", "B", "BEFORE"),
    ("B", "C", "IS_INCLUDED"),
    ("D", "C", "INCLUDES"),
    ("E", "D", "CONTAINS"),
    ("F", "E", "AFTER"),
}
tlinks = {f'l{idx}': TLink(scr, tgt, interval_relation=rel) for idx, (scr, tgt, rel) in enumerate(reference)}
tlinks_tc = Document('resources/empty.txt').temporal_closure(tlinks)

print('Reference tlinks.')
pprint([(tlink.source, tlink.target, tlink.interval_relation) for lid,  tlink in tlinks.items()])
print('Temporal closure tlinks.')
pprint([(tlink.source, tlink.target, tlink.interval_relation) for lid,  tlink in tlinks_tc.items()])
