
from typing import List, Union
from pathlib import Path

from xml.etree import ElementTree as ET


class TMLHandler:
    """Object responsible for handling all the operations regarding XML files."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = path

        # parse the xml file
        tree = ET.parse(self.path)
        self.root = tree.getroot()

        # by default we add the text and endpoints to the xml element attributes
        self._add_text_to_attributes()
        self._add_endpoints_to_attributes()

    @property
    def text(self) -> str:
        return ''.join(list(self.root.itertext()))

    def get_tag(self, tag: str) -> List:
        return [element for element in self.root.iter()
                if element.tag == tag]

    def _add_text_to_attributes(self) -> None:
        """Add the text to attributes of each xml element."""

        for element in self.root.iterfind('.//*'):
            text = "".join(element.itertext())
            if text:
                element.attrib["text"] = text

    def _add_endpoints_to_attributes(self) -> None:
        """Add the text endpoints to the attributes of endpoint od each xml element."""

        # Find indexes of each text block.
        text_blocks = []
        start = 0
        for txt in self.root.itertext():
            end = start + len(txt)
            text_blocks += [(start, end, txt)]
            start = end

        # Add the endpoints to each xml element attribute.
        elements = [
            *self.root.iterfind('.//TIMEX3'),
            *self.root.iterfind('.//EVENT'),
        ]

        for element in elements:
            text = "".join(element.itertext())
            if text:
                for idx, (start, end, block) in enumerate(text_blocks):
                    if text == block:
                        element.attrib["endpoints"] = (start, end)
                        # remove the items that were found.
                        text_blocks = text_blocks[idx + 1:]
                        break


class XMLHandler:
    """Object responsible for handling all the operations regarding XML files."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = path

        # parse the xml file
        tree = ET.parse(self.path)
        self.root = tree.getroot()

        # by default we add the text and endpoints to the xml element attributes
        self._add_text_to_attributes()
        self._add_endpoints_to_attributes()

    @property
    def text(self) -> str:
        text = ""
        sent_num = 0
        for token in self.get_tag("token"):

            if int(token.attrib["sentence"]) != sent_num:
                text += "\n"
                sent_num = int(token.attrib["sentence"])

            text += " " + token.attrib["text"]
        return text

    def get_tag(self, tag: str) -> List:
        return [element for element in self.root.iter()
                if element.tag == tag]

    def _add_text_to_attributes(self) -> None:
        """Add the text to attributes of each xml element."""

        for element in self.root.iterfind('.//*'):
            text = "".join(element.itertext())
            if text:
                element.attrib["text"] = text

    def _add_endpoints_to_attributes(self) -> None:
        """Add the text endpoints to the attributes of endpoint od each xml element."""

        # Find indexes of each text block.
        text_blocks = []
        start = 0
        for txt in self.root.itertext():
            end = start + len(txt)
            text_blocks += [(start, end, txt)]
            start = end

        # Add the endpoints to each xml element attribute.
        for element in self.root.iterfind('.//*'):

            text = "".join(element.itertext())
            if text:
                for idx, (start, end, block) in enumerate(text_blocks):
                    if text == block:
                        element.attrib["endpoints"] = (start, end)
                        # remove the items that were found.
                        text_blocks = text_blocks[idx + 1:]
                        break
