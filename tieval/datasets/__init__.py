""" Module to download and read datasets.

Contains functions to download and read temporally annotated datasets.
The corpus available are:
    - TimeBank
    - TimeBank.1-2
    - Platinum
    - Tempeval-3
    - TimeBank-PT
    - Aquaint
    - MATRES
    - TDDiscourse
    - TimeBank-Dense


"""

from tieval.datasets.readers import (
    EventTimeDatasetReader,
    TempEval3DocumentReader,
    GraphEveDocumentReader,
    XMLDatasetReader,
    JSONDatasetReader,
    TimeBankDenseDatasetReader,
    TimeBank12DocumentReader,
    MeanTimeDocumentReader,
)
from tieval.datasets.metadata import DATASETS_METADATA
from tieval.datasets.datasets import read
from tieval.datasets.downloader import download

SUPPORTED_DATASETS = list(DATASETS_METADATA)
