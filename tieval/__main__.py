"""
Usage:
------

    $ tieval <command> [OPTIONS]

Commands:
    download        Download temporally annotated corpora.

Options:
    -h, --help      Show this help.
    -v, --version   Show the version.


Contacts:
--------
- hugo.o.sousa@inesctec.pt
"""

import sys

from tieval.datasets import downloader, download


def main():

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    opts = [o for o in sys.argv[1:] if o.startswith("-")]

    help_msg = ("-h" in opts) or ("--help" in opts)

    download_cmd = "download" in args
    if download_cmd:

        if help_msg:
            print(downloader.__doc__)
            return None

        args.remove("download")

        datasets = args
        for dataset in datasets:
            download(dataset)

        return None

    # return the documentation if no arguments or commands are passed
    print(__doc__)


if __name__ == "__main__":
    main()
