
import os
import pytest
from collections import defaultdict

from text2timeline. datasets import load_dataset, DATASETS_METADATA

os.chdir("/home/hugosousa/Projects/text2timeline")


@pytest.fixture
def dataset_statistics():

    statistics = {
        "timebank-1.2": {
            "n_docs": 183,
            "n_events": 7935,
            "n_timexs": 1414,
            "n_tlinks": 6416
        },

    }

    return statistics


def test_load_dataset(dataset_statistics):

    def get_dataset_statistics(datasets):

        statistics = defaultdict(int)

        for dataset in datasets:
            statistics["n_docs"] += len(dataset)
            for doc in dataset.documents:
                statistics["n_events"] += len(set(event.id for event in doc.events))
                statistics["n_timexs"] += len(doc.timexs)
                statistics["n_tlinks"] += len(doc.tlinks)

        return statistics

    for dataset_name in dataset_statistics:
        dataset = load_dataset(dataset_name)

        statistics = dataset_statistics[dataset_name]
        statistics_found = get_dataset_statistics(dataset)

        for key in statistics:

            if statistics[key] is None:
                continue

            assert statistics_found[key] == statistics[key], f"The {key} is not correct for dataset {dataset_name}."


