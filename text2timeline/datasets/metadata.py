""" Store metadata about datasets.

Objects
-------
    - DatasetMetadata

Constants
---------
    - DATASETS_METADATA

"""

from dataclasses import dataclass
from typing import Union, Tuple, Iterable

from text2timeline.datasets.readers import TMLDatasetReader, TableDatasetReader


@dataclass
class DatasetMetadata:
    """ Store dataset metadata."""

    name: str
    reader: Union[TMLDatasetReader, TableDatasetReader] = None
    path: Iterable[str] = None
    url: str = None
    repo: str = None
    base: Iterable[str] = None
    extension: str = None
    columns: Tuple = None
    event_index: str = None
    description: str = None
    download_description: str = None


# datasets that only have a table with the temporal links
DATASETS_METADATA = {
    "aquaint": DatasetMetadata(
        name="AQUAINT",
        reader=TMLDatasetReader,
        path=(
            "data/aquaint",
        ),
        url="https://drive.inesctec.pt/s/sW8Acr9wxGzmn74/download",
        download_description=
        """
        AQUAINT is part of the Linguistic Data Consortium catalog https://catalog.ldc.upenn.edu/LDC2002T31.
        In order to download one needs to creat and account and accept the terms of usage.
        
        ** TempEval-3 dataset contains a version of AQUAINT that is downloadable**
        """
    ),

    "eventtime": DatasetMetadata(
        name="EventTime",
        url="https://drive.inesctec.pt/s/TT6M7TXTbo93kKb/download",
    ),

    "grapheve": DatasetMetadata(
        name="GraphEve",
        url="https://drive.inesctec.pt/s/2oYDkRTwTSpyY4N/download"
    ),

    "matres": DatasetMetadata(
        name="MATRES",
        url="https://drive.inesctec.pt/s/bNkyKYLog9TLTfp/download",
        base=("tempeval-3",),
        columns=("doc", "src_token", "tgt_token", "src", "tgt", "relation"),
        event_index="eiid",
        repo="https://github.com/qiangning/MATRES",
        description="",
        reader=TableDatasetReader,
        path=(
            'data/matres/aquaint.txt',
            'data/matres/timebank.txt',
            'data/matres/platinum.txt',
        )
    ),

    "mctaco": DatasetMetadata(
        name="MCTaco",
        url="https://drive.inesctec.pt/s/dNkSnFKsyjjYRCQ/download"
    ),

    "meantime": DatasetMetadata(
        name="MeanTime",
        url="https://drive.inesctec.pt/s/YskTKn5H7Bj7Dtt/download"
    ),

    "platinum": DatasetMetadata(
        name="Platinum",
        url="https://drive.inesctec.pt/s/X7PdbFF466ffMsC/download",
        reader=TMLDatasetReader,
        path=(
            'data/TempEval-3/Test/TempEval-3-Platinum',
        ),
    ),

    "tcr": DatasetMetadata(
        name="TCR",
        url="https://drive.inesctec.pt/s/H9xi7mRpEkYL6ws/download",
        description="""
        TCR stands for Temporal and Causal Reasoning, which is a new dataset proposed in "Joint 
        Reasoning for Temporal and Causal Relations" (Q. Ning et al., 2018). TCR is jointly annotated with both 
        temporal and causal relations. Specifically, the temporal relations were annotated based on the scheme 
        proposed in "A Multi-Axis Annotation Scheme for Event Temporal Relations" using CrowdFlower; the causal 
        relations were mapped from the "EventCausalityData 
        """,
        reader=TMLDatasetReader,
        repo="https://github.com/CogComp/TCR",
        path=(
            "data/TCR/TemporalPart",
        )
    ),

    "tddiscourse": DatasetMetadata(
        name="TDDiscourse",
        url="https://drive.inesctec.pt/s/reaoG7LJD4sjdaN/download",
        base={"timebank-1.2"},
        columns=("doc", "src", "tgt", "relation"),
        event_index="eid",
        repo="https://github.com/aakanksha19/TDDiscourse",
        description=
        """TDDiscourse is a dataset for temporal ordering of events, which specifically focuses on event pairs that 
        are more than one sentence apart in a document. TDDiscourse was created by augmenting TimeBank-Dense. 
        TimeBank-Dense focuses mainly on event pairs which are in the same or adjacent sentences (though they do 
        include labels for some event pairs which are more than one sentence apart). TDDiscourse was created to 
        address this gap and to turn the focus towards discourse-level temporal ordering, which turns out to be a 
        harder task.""",
        reader=TableDatasetReader,
        path=(
            "data/TDDiscourse/TDDMan/TDDManTrain.tsv",
            "data/TDDiscourse/TDDMan/TDDManDev.tsv",
            "data/TDDiscourse/TDDMan/TDDManTest.tsv",
            "data/TDDiscourse/TDDAuto/TDDAutoTrain.tsv",
            "data/TDDiscourse/TDDAuto/TDDAutoDev.tsv",
            "data/TDDiscourse/TDDAuto/TDDAutoTest.tsv",

        )
    ),

    "tempeval-2": DatasetMetadata(
        name="TempEval-2",
        url="https://drive.inesctec.pt/s/ZD6wzmrgysr8mne/download"
    ),

    "tempeval-3": DatasetMetadata(
        name="TempEval-3",
        url="https://drive.inesctec.pt/s/K457eTAo656gqMw/download",
        reader=TMLDatasetReader,
    ),

    "tempqa": DatasetMetadata(
        name="TempQA",
        url="https://drive.inesctec.pt/s/q6jntQeoa2EZntD/download"
    ),

    "tempquestions": DatasetMetadata(
        name="TempQuestions",
        url="https://drive.inesctec.pt/s/KgdXNKz35DW2opN/download"
    ),

    "timebank-1.2": DatasetMetadata(
        name="TimeBank-1.2",
        url="https://drive.inesctec.pt/s/2yTsSCRsmJ9nGKH/download",
        reader=TMLDatasetReader,
        path=(
            "data/TimeBank-1.2/data/timeml",
        ),
        description=
        """The TimeBank Corpus contains 183 news articles that have been annotated with temporal information, 
        adding events, times and temporal links between events and times.""",
        download_description=
        """
        TimeBank-1.2 is part of the Linguistic Data Consortium catalog https://catalog.ldc.upenn.edu/LDC2006T08.
        In order to download one needs to creat and account and accept the terms of usage.
        
        ** TempEval-3 dataset contains a version of TimeBank that is downloadable**
        """
    ),

    "timebank-dense": DatasetMetadata(
        name="TimeBank-Dense",
        url="https://drive.inesctec.pt/s/8AK3LYJpayn39px/download",
        base={"timebank-1.2"},
        columns=("doc", "src", "tgt", "relation"),
        event_index="eid",
        reader=TableDatasetReader,
        path=(
            "data/timebank-dense/TimebankDense.full.txt",
        ),
    ),

    "timebankpt": DatasetMetadata(
        name="TimeBank-Dense",
        url="https://drive.inesctec.pt/s/kxk9yMHAiK4XdKS/download",
        description=
        """TimeBankPT was obtained by translating the English data used in the first TempEval competition (
        http://timeml.org/tempeval).\nTimeBankPT can be found at http://nlx.di.fc.ul.pt/~fcosta/TimeBankPT.\nThat 
        page contains some information about the corpus and a link to the release.""",
        reader=TMLDatasetReader,
        path=(
            'data/TimeBankPT/train',
            'data/TimeBankPT/tests'
        )
    ),

    "timebank": DatasetMetadata(
        name="TimeBank",
        url="https://drive.inesctec.pt/s/i57gZQjZs5XiDpZ/download",
        reader=TMLDatasetReader,
    ),

    "torque": DatasetMetadata(
        name="TORQUE",
        url="https://drive.inesctec.pt/s/yLLiJtTW99XLpZE/download"
    ),

    "traint3": DatasetMetadata(
        name="Traint3",
        url="https://drive.inesctec.pt/s/4mF6NmZa47zyX2R/download",
        reader=TMLDatasetReader,
    ),

    "uds-t": DatasetMetadata(
        name="UDS-T",
        url="https://drive.inesctec.pt/s/db5dNCCyG3szWej/download"
    )
}
