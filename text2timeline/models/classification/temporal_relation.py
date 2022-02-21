from typing import Iterable

from text2timeline.models.base import BaseModel
from text2timeline.base import Document
from text2timeline.links import TLink


class TLinkClassificationBaseline(BaseModel):

    def predict(self, documents: Iterable[Document]):

        predictions = {}
        for doc in documents:
            doc_predictions = [
                TLink(
                    tlink.source,
                    tlink.target,
                    relation="BEFORE"
                )
                for tlink in doc.tlinks
            ]

            predictions[doc.name] = doc_predictions

        return predictions
