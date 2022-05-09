import io
import requests
from typing import List
import zipfile


def _download_url(url: str, path: str) -> None:
    """Download from url.

    Parameters
    ----------
    url: str
        The url to download.
    path: str
        The path to store the object.
    """

    print(f"Downloading from {url}")

    response = requests.get(url, stream=True)
    if response.ok:

        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(path)
        print("Done.")

    else:
        raise Exception(f"Request code: {response.status_code}")


def get_spans(
        text: str,
        elements: List[str],
        start_idx: int = 0
) -> List[List[int]]:

    running_idx = start_idx
    spans = []
    for element in elements:

        offset = text.find(element)
        start = running_idx + offset

        element_len = len(element)
        end = start + element_len

        spans += [[start, end]]

        text = text[offset + element_len:]
        running_idx = end

    return spans


def resolve_sentence_idxs(sent_idx1: int, sent_idx2: int) -> List[int]:

    if sent_idx1 is None:
        return [sent_idx2]

    elif sent_idx2 is None:
        return [sent_idx1]

    elif sent_idx1 == sent_idx2:
        return [sent_idx1]

    else:
        return sorted([sent_idx1, sent_idx2])
