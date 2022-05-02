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


def get_spans(text: str, elements: List[str]) -> List[List[int]]:

    running_idx = 0
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