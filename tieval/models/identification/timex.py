import pathlib
import random
from typing import Iterable

import spacy
from tqdm import tqdm
from spacy.util import minibatch
from spacy.util import compounding
from spacy.training import Example
from py_heideltime import py_heideltime

from tieval.base import Document
from tieval.entities import Timex
from tieval.models.base import (
    BaseModel,
    BaseTrainableModel
)
from tieval.models import metadata
from tieval import utils


class TimexIdentificationBaseline(BaseTrainableModel):

    def __init__(self, path: str = "./models"):

        path = pathlib.Path(path)
        self.path = path / "timex_identification"

        self.nlp = None

        if not self.path.is_dir():
            self.download()

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
            dev_documents: Iterable[Document] = None,
            dropout: float = 0,
            from_scratch: bool = False
    ) -> None:
        """Tran the model."""

        # preprocess data
        train_set = self.data_pipeline(documents)
        n_train_entities = sum(len(doc[1]["entities"]) for doc in train_set)

        # creat model
        if from_scratch:
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("ner")

            ner = self.nlp.get_pipe("ner")
            ner.add_label("TIMEX")

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
        url = metadata.MODELS_URL["timex_identification"]
        utils.download_url(url, self.path.parent)

    @staticmethod
    def data_pipeline(documents: Iterable[Document]):

        annotations = []
        for doc in documents:
            annot = {
                "entities": list(set([
                    (timex.endpoints[0], timex.endpoints[1], "TIMEX")
                    for timex in doc.timexs
                    if not timex.is_dct
                ]))
            }

            annotations += [(doc.text, annot)]

        return annotations


class HeidelTime(BaseModel):
    """ The HeidelTime model. This is a wrapper class of the
    `py_heideltime <https://github.com/JMendes1995/py_heideltime>`_ implementation. Follow the installation steps
    provided in the py_heideltime repository in order for it ot work properly.


    :param str language: {"English", "Portuguese", "Spanish", "Germany", "Dutch", "Italian", "French"}
        Language of the text that will be processed.
    :param str document_type: {"News", "Narrative", "Colloquial", "Scientific"}
        The type of document that will be processed.


    .. seealso::
        `Strötgen, Gertz: HeidelTime: High Qualitiy Rule-based
        Extraction and Normalization of Temporal Expressions. SemEval'10. <https://aclanthology.org/S10-1071/>`_
    """

    def __init__(self, language="English",  document_type="news"):
        self.language = language
        self.document_type = document_type

    def predict(self, documents: Iterable[Document]):
        """ Make predictions.

        Parameters
        ----------
        documents : Iterable[Document]
            An iterable containing the documents tto extract the temporal expressions.

        Returns
        -------
        pred_timexs : dict[str, list[Timex]]
            A dictionary that maps the name of each document to a list with the identified temporal expressions.
        """

        pred_timexs = {}
        for doc in tqdm(documents):

            dct = doc.dct.value[:10]
            results = py_heideltime(
                doc.text.strip(),
                language=self.language,
                document_type=self.document_type,
                document_creation_time=dct
            )

            if results is None:
                pred_timexs[doc.name] = []
                continue

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
