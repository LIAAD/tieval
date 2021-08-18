from text2timeline.utils import XMLHandler

import xml.etree.ElementTree as ET

from pprint import pprint

handler = XMLHandler("data/TimeBank-1.2/data/timeml/ABC19980114.1830.0611.tml")

handler.get_tag('TIMEX3')
handler.get_tag('EVENT')
handler.get_tag('MAKEINSTANCE')

handler.root

handler.root.tag
handler.root.attrib

handler.root[1]

data='''<?xml version="1.0" encoding="UTF-8"?>
<metadata>
<TIMEX>
<EVENT>
<NUMEX>example</NUMEX>
</EVENT>
</TIMEX>
</metadata>
'''

root = ET.fromstring(data)

root.tag

for e in root[0]:
    print(e.tag)

list(root[0].itertext())

root.findall('EVENT')
pprint(dir(root))

for i in root.iter():
    print(i.tag, "".join(i.itertext()))