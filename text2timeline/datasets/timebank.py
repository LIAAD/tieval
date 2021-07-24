from text2timeline.datasets.custom import ReadDataset


def load_data():
    return ReadDataset("timebank").read()
