
import pathlib
from typing import Iterable
import random

import spacy
from spacy.util import compounding
from spacy.util import minibatch
from spacy.training import Example

from tieval.base import Document
from tieval.entities import Event
from tieval.models.base import BaseTrainableModel
from tieval.models import metadata
from tieval import utils


class EventIdentificationBaseline(BaseTrainableModel):

    def __init__(self, path: str = "./models") -> None:

        path = pathlib.Path(path)
        self.path = path / "event_identification"

        self.nlp = None

        if self.path.is_dir():
            self.load()

        else:
            # TODO: download the model
            self.download()

    def predict(self, documents: Iterable[Document]):

        result = {}
        for doc in documents:
            prediction = self.nlp(doc.text)

            events = []
            for entity in prediction.ents:

                events += [Event(
                    text=entity.text,
                    endpoints=(entity.start_char, entity.end_char)
                )]

            result[doc.name] = events

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
            ner.add_label("EVENT")

            self.nlp.begin_training()

        # train
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

        if not self.path.is_dir():
            self.path.mkdir()

        self.nlp.to_disk(self.path)

    def load(self):
        self.nlp = spacy.load(self.path)

    def download(self):
        url = metadata.MODELS_URL["event_identification"]
        utils._download_url(url, self.path.parent)

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
