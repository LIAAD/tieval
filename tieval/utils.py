import io
import requests
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
