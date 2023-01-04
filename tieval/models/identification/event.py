import pathlib
import random
from typing import Iterable

import spacy
from spacy.training import Example
from spacy.util import compounding
from spacy.util import minibatch

from tieval import utils
from tieval.base import Document
from tieval.entities import Event
from tieval.models import metadata
from tieval.models.base import BaseTrainableModel


class EventIdentificationBaseline(BaseTrainableModel):

    def __init__(self, path: str = "./models") -> None:

        path = pathlib.Path(path)
        self.path = path / "event_identification"

        self.nlp = None

        if not self.path.is_dir():
            self.download()

        self.load()

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
            dev_documents: Iterable[Document] = None,
            dropout: float = 0,
            from_scratch: bool = False
    ) -> None:
        """Tran the model.

        Parameters
        ----------
        documents : Iterable[Document]
            The set of documents to train on.
        from_scratch : bool
            If False (the default value) it will fine-tune the model. If set to True it will train from scratch.
        dropout : float
        dev_documents : Iterable[Document]
        """

        # preprocess data
        train_set = self.data_pipeline(documents)
        n_train_entities = sum(len(doc[1]["entities"]) for doc in train_set)

        # creat model
        if from_scratch:
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("ner")

            ner = self.nlp.get_pipe("ner")
            ner.add_label("EVENT")

            self.nlp.begin_training()

        # train
        random.shuffle(train_set)

        losses = {}
        batches = minibatch(train_set, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)

            example = []
            for i in range(len(texts)):
                doc = self.nlp.make_doc(texts[i])
                example += [Example.from_dict(doc, annotations[i])]

            self.nlp.update(example, drop=dropout, losses=losses)

        if dev_documents:  # validate on the development set

            dev_set = self.data_pipeline(dev_documents)
            n_dev_entities = sum(len(doc[1]["entities"]) for doc in dev_set)

            dev_losses = {}
            dev_batches = minibatch(dev_set, size=32)
            for dev_batch in dev_batches:

                texts, annotations = zip(*dev_batch)

                example = []
                for i in range(len(texts)):
                    doc = self.nlp.make_doc(texts[i])
                    example += [Example.from_dict(doc, annotations[i])]

                self.nlp.update(example, losses=dev_losses)

            print(f"Losses:\t"
                  f"Train {losses['ner'] / n_train_entities:.5f}\t"
                  f"Dev {dev_losses['ner'] / n_dev_entities:.5f}")

        else:  # in case no dev set is provided just print the train losses
            print(f"\tLosses Train {losses['ner'] / n_train_entities:.5f}")

    def save(self):
        """Store model in disk."""

        if not self.path.is_dir():
            self.path.mkdir()

        self.nlp.to_disk(self.path)

    def load(self):
        self.nlp = spacy.load(self.path)

    def download(self):
        url = metadata.MODELS_URL["event_identification"]
        utils.download_url(url, self.path.parent)

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
