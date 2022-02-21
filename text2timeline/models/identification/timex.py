from typing import Iterable

from tqdm import tqdm

import random

import spacy
from spacy.util import minibatch
from spacy.util import compounding
from spacy.training import Example

from py_heideltime import py_heideltime

from text2timeline import MODELS_PATH
from text2timeline.models.base import BaseModel
from text2timeline.models.base import BaseTrainableModel
from text2timeline.base import Document
from text2timeline.entities import Timex


class TimexIdentificationBaseline(BaseTrainableModel):

    def __init__(self):
        self.path = MODELS_PATH / "timex_identification"

        self.nlp = None

        if self.path.is_dir():
            self.load()

    def predict(self, documents: Iterable[Document]):

        result = {}
        for doc in documents:
            prediction = self.nlp(doc.text)

            timexs = []
            for entity in prediction.ents:

                timexs += [Timex(
                    text=entity.text,
                    endpoints=(entity.start_char, entity.end_char)
                )]

            result[doc.name] = timexs

        return result

    def fit(
            self,
            documents: Iterable[Document],
            n_epochs: int = 30,
            from_scratch: bool = False
    ):

        train_set = self.data_pipeline(documents)

        # creat model
        if from_scratch:
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("ner")

            ner = self.nlp.get_pipe("ner")
            ner.add_label("TIMEX")

            self.nlp.begin_training()

        # train for 30 iterations
        for epoch in range(n_epochs):

            # shuffle
            random.shuffle(train_set)

            losses = {}
            batches = minibatch(train_set, size=compounding(4., 32., 1.001))
            print(f"Epoch {epoch}")
            for batch in batches:
                texts, annotations = zip(*batch)

                example = []
                for i in range(len(texts)):
                    doc = self.nlp.make_doc(texts[i])
                    example += [Example.from_dict(doc, annotations[i])]

                self.nlp.update(
                    example,
                    drop=0.5,
                    losses=losses
                )

            print("\tLosses", losses)

    def save(self):
        """Store model in disk."""
        self.nlp.to_disk(self.path)

    def load(self):
        self.nlp = spacy.load(self.path)

    @staticmethod
    def data_pipeline(documents: Iterable[Document]):

        annotations = []
        for doc in documents:
            annot = {
                "entities": list(set([
                    (event.endpoints[0], event.endpoints[1], "EVENT")
                    for event in doc.events
                ]))
            }

            annotations += [(doc.text, annot)]

        return annotations


class HeidelTime(BaseModel):

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

                timexs += [Timex(
                    value=value,
                    text=text,
                    endpoints=endpoints
                )]

            pred_timexs[doc.name] = timexs

        return pred_timexs
