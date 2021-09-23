from text2timeline.readers import TMLDocumentReader

path = "data/sample.tml"

reader = TMLDocumentReader()
doc = reader.read(path)

doc.temporal_closure