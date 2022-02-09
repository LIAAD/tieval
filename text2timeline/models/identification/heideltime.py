from typing import Iterable

from tqdm import tqdm

from py_heideltime import py_heideltime

from text2timeline.base import Document
from text2timeline.entities import Timex


class Heideltime:

    def __init__(self, language='English',  document_type='news'):
        self.language = language
        self.document_type = document_type

    def predict(self, documents: Iterable[Document]):

        pred_timexs = {}
        for doc in tqdm(documents):

            dct = doc.dct.value[:10]

            results = py_heideltime(
                doc.text.strip(),
                language=self.language,
                document_type=self.document_type,
                document_creation_time=dct
            )

            # format heideltime outputs to Timex instances
            idx, timexs = 0, []
            for value, text in results[0]:

                s = doc.text[idx:].find(text)
                e = s + len(text)
                endpoints = (s + idx, e + idx)
                idx += e

                attrib = {
                    "value": value,
                    "text": text,
                    "endpoints": endpoints
                }

                timexs += [Timex(attrib)]

            pred_timexs[doc.name] = timexs

        return pred_timexs




