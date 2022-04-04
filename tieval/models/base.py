
from abc import ABC
from abc import abstractmethod
from typing import Iterable

from tieval.base import Document


class BaseModel(ABC):

    @abstractmethod
    def predict(self, documents: Iterable[Document]):
        pass


class BaseTrainableModel(ABC):

    @abstractmethod
    def predict(self, documents: Iterable[Document]):
        pass

    @abstractmethod
    def fit(self, documents: Iterable[Document]):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @staticmethod
    @abstractmethod
    def data_pipeline(documents: Iterable[Document]):
        pass
