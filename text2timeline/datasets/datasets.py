
from typing import List

from text2timeline.base import Dataset
from text2timeline.datasets import DATASETS_METADATA


def load(dataset: str) -> List[Dataset]:
    """Load temporally annotated dataset.

    The supported datasets are:
        TimeBank
        TimeBank.1-2
        Platinum
        Tempeval-3
        TimeBank-PT
        Aquaint
        MATRES
        TDDiscourse
        TimeBank-Dense

    Parameters
    ----------
    dataset: str
        The name of the dataset to read.

    Returns
    -------
    List[Dataset]
        A list with all the datasets  # TODO: return only one dataset.

    """

    dataset = dataset.lower().strip()
    metadata = DATASETS_METADATA[dataset]

    # guardians
    if dataset not in DATASETS_METADATA:
        raise ValueError(f"{dataset} not found on datasets")

    # table dataset
    if metadata.base:

        # read base dataset
        base_datasets = []
        for dataset_name in metadata.base:
            base_datasets += load(dataset_name)
        base_dataset = sum(base_datasets)

        # define reader
        reader = metadata.reader(metadata, base_dataset)

    # tml dataset
    else:
        reader = metadata.reader()

    return [reader.read(path) for path in metadata.path]
