
from typing import List

from xml.etree import ElementTree as ET

import re


class XMLHandler:
    """Object responsible for handling all the operations regarding XML files."""

    def __init__(self, path: str):
        self.path = path

        tree = ET.parse(self.path)
        self.root = tree.getroot()

    @property
    def text(self):
        return ''.join(list(self.root.itertext()))

    def get_tag(self, tag_name: str) -> List:
        return [element for element in self.root.iter()
                if element.tag == tag_name]

