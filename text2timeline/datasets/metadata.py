
from dataclasses import dataclass
from typing import Union, Tuple, Iterable

from text2timeline.readers import TMLDatasetReader, TableDatasetReader


@dataclass
class DatasetMetadata:
    name: str
    reader: Union[TMLDatasetReader, TableDatasetReader]
    path: Iterable[str]
    urls: Iterable[str] = None
    repo: str = None
    base: Iterable[str] = None
    extension: str = None
    columns: Tuple = None
    event_index: str = None
    description: str = None
    download_description: str = None


# datasets that only have a table with the temporal links
DATASETS_METADATA = {

    "timebank": DatasetMetadata(
        name="TimeBank",
        reader=TMLDatasetReader,
        path=(
            "data/TBAQ-cleaned/TimeBank",
        )
    ),

    "timebank-1.2": DatasetMetadata(
        name="TimeBank-1.2",
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

    "aquaint": DatasetMetadata(
        name="AQUAINT",
        reader=TMLDatasetReader,
        path=(
            'data/TBAQ-cleaned/AQUAINT',
        ),
        download_description=
        """
        AQUAINT is part of the Linguistic Data Consortium catalog https://catalog.ldc.upenn.edu/LDC2002T31.
        In order to download one needs to creat and account and accept the terms of usage.
        
        ** TempEval-3 dataset contains a version of AQUAINT that is downloadable**
        """
    ),

    "timebankpt": DatasetMetadata(
        name="TimeBankPT",
        urls=[
            "http://nlx-server.di.fc.ul.pt/~fcosta/TimeBankPT/TimeBankPTv1.0.zip"
        ],
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

    "tempeval-3": DatasetMetadata(
        name="TempEval-3",
        reader=TMLDatasetReader,
        path=(
            'data/TBAQ-cleaned/TimeBank',
            'data/TBAQ-cleaned/AQUAINT',
            # 'data/TempEval-3/Train/trainT3',
            'data/te3-platinum'
        ),
        urls=[
            "https://www.cs.york.ac.uk/semeval-2013/task1/data/uploads/datasets/tbaq-2013-03.zip",
            "https://www.cs.york.ac.uk/semeval-2013/task1/data/uploads/datasets/te3-platinumstandard.tar.gz"
        ]
    ),

    "platinum": DatasetMetadata(
        name="Platinum",
        reader=TMLDatasetReader,
        path=(
            'data/TempEval-3/Test/TempEval-3-Platinum',
        ),
        urls=[
            "https://www.cs.york.ac.uk/semeval-2013/task1/data/uploads/datasets/te3-platinumstandard.tar.gz"
        ]
    ),

    "matres": DatasetMetadata(
        name="MATRES",
        base=("tempeval-3",),
        columns=("doc", "src_token", "tgt_token", "src", "tgt", "relation"),
        event_index="eiid",
        repo="https://github.com/qiangning/MATRES",
        description="",
        reader=TableDatasetReader,
        path=(
            'data/MATRES/aquaint.txt',
            'data/MATRES/timebank.txt',
            'data/MATRES/platinum.txt',
        )
    ),

    "tddiscourse": DatasetMetadata(
        name="TDDiscourse",
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

    "timebank-dense": DatasetMetadata(
        name="TimeBank-Dense",
        base={"timebank-1.2"},
        columns=("doc", "src", "tgt", "relation"),
        event_index="eid",
        reader=TableDatasetReader,
        path=(
            "data/TimeBank-Dense/TimebankDense.full.txt",
        ),
        urls=[
            "https://www.usna.edu/Users/cs/nchamber/caevo/TimebankDense.full.txt"
        ]
    ),

    "tcr": DatasetMetadata(
        name="TCR",
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
    )
}
