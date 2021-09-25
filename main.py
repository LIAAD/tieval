from text2timeline.readers import TMLDocumentReader
from text2timeline.temporal_relation import PointRelation

p = PointRelation(end_start="=")
p.timeline()

path = "data/sample.tml"

reader = TMLDocumentReader()
doc = reader.read(path)

doc.temporal_closure