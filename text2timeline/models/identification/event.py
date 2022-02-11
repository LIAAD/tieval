from typing import Iterable

import random
from pathlib import Path

import spacy
from spacy.util import compounding
from spacy.util import minibatch
from spacy.training import Example

from text2timeline.base import Document
from text2timeline.entities import Event

MODELS_PATH = Path("models")


class EventIdentificationBaseline:

    def __init__(self):
        self.path = MODELS_PATH / "event_identification"
        self.nlp = spacy.load(self.path)

    def predict(self, documents: Iterable[Document]):

        result = {}
        for doc in documents:
            prediction = self.nlp(doc.text.strip())

            events = []
            for entity in prediction.ents:
                attrib = {
                    "text": entity.text,
                    "endpoints": (entity.start_char, entity.end_char)
                }

                events += [Event(attrib)]

            result[doc.name] = events

        return result

    def train(self, documents: Iterable[Document]):

        train_set = self.preprocess(documents)

        # creat model
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("ner")

        ner = self.nlp.get_pipe("ner")
        ner.add_label("EVENT")

        self.nlp.begin_training()

        # train for 30 iterations
        for epoch in range(30):

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

    @staticmethod
    def preprocess(documents: Iterable[Document]):

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
