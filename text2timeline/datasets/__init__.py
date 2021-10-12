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

from text2timeline.datasets.readers import TMLDatasetReader
from text2timeline.datasets.readers import TableDatasetReader

from text2timeline.datasets.metadata import DATASETS_METADATA
from text2timeline.datasets.datasets import load
from text2timeline.datasets.downloader import download
