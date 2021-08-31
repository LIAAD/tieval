
from typing import List, Tuple, Iterable, Union

from xml.etree import ElementTree as ET

import re

from pprint import pprint


class XMLHandler:
    """Object responsible for handling all the operations regarding XML files."""

    def __init__(self, path: str):
        self.path = path

        tree = ET.parse(self.path)
        self.root = tree.getroot()
        self. _expression_indexes()

    @property
    def text(self):
        return ''.join(list(self.root.itertext()))

    def get_tag(self, tag: str) -> List:
        return [element for element in self.root.iter()
                if element.tag == tag]

    def _expression_indexes(self) -> dict:
        """
        Finds start and end indexes of each expression (EVENT or TIMEX).

        :return:
        """

        # Find indexes of the expressions_uid.
        text_blocks = []
        start = 0
        for txt in self.root.itertext():
            end = start + len(txt)
            text_blocks += [(start, end, txt)]
            start = end

        # Get the tags of each expression.
        text_tags = []
        for element in self.root.iterfind('.//*'):

            # there are cases where there is a nested tag <EVENT><NUMEX>example</NUMEX></EVENT>
            text = "".join(element.itertext())

            if element.attrib and element.tag == 'EVENT':
                text_tags.append((text, element.attrib['eid']))

            elif element.attrib and element.tag == 'TIMEX3':
                text_tags.append((text, element.attrib['tid']))

        # Join the indexes with the tags.
        expression_idxs = {self.dct.id: (-1, -1)}  # Initialize with the position of DCT.
        while text_tags:
            txt, tag_id = text_tags.pop(0)
            for idx, (start, end, txt_sch) in enumerate(text_blocks):
                if txt == txt_sch:
                    expression_idxs[tag_id] = (start, end)
                    # remove the items that were found.
                    text_blocks = text_blocks[idx + 1:]
                    break

        return expression_idxs

