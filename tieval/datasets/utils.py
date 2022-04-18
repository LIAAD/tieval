import string
from pathlib import Path
from typing import Union, List, Dict
from xml.etree import ElementTree as ET

import xmltodict


def xml2dict(path: Union[str, Path]) -> Dict:
    # parse the xml file
    with open(path, 'r', encoding='utf-8') as f:
        xml = f.read()

    result = xmltodict.parse(
        xml,
        attr_prefix="",
        cdata_key="text",
        dict_constructor=dict
    )

    return result


def detokenize(tokens: List[str]) -> str:
    text = [
        " " + tkn
        if not tkn.startswith("'") and tkn not in string.punctuation
        else tkn
        for tkn in tokens
    ]

    return "".join(text).strip()
