import io
from pathlib import Path
import requests
from tqdm import tqdm
from typing import List, Union
import zipfile


def download_url(url: str, path: Union[str, Path]) -> None:
    """Download from url.

    :param str url: The url to download.
    :param str path: The path to store the object.
    """

    print(f"Downloading from {url}")

    response = requests.get(url, stream=True)
    if response.ok:

        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(path)
        print("Done.")

    else:
        raise Exception(f"Request code: {response.status_code}")


def download_torch_weights(url: str, path: Union[str, Path]) -> None:

    print(f"Downloading from {url}")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    if response.ok:

        if not path.parent.is_dir():
            path.parent.mkdir()

        with open(path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

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
