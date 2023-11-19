from setuptools import find_packages, setup
import subprocess
import os
from pathlib import Path

PWD = Path(__file__).parent.resolve()

README = (PWD / "README.md").read_text(encoding="utf-8")


version = (
    subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)

if "-" in version:
    # See: https://peps.python.org/pep-0440/#local-version-segments
    v, i, s = version.split("-")
    version = v + "+" + i + ".git." + s

# assert os.path.isfile("tieval/version.py")
version_path = PWD / "tieval" / "VERSION"
version_path.write_text(f"{version}\n", encoding="utf-8")


setup(
    name="tieval",
    version=version,
    url="https://github.com/LIAAD/tieval",
    license='MIT',
    author="Hugo Sousa",
    author_email="hugo.o.sousa@inesctec.pt",
    description="This framework facilitates the development and test of temporal-aware models.",
    long_description_content_type='text/markdown',
    long_description=README,
    packages=find_packages(exclude=('tests*',)),
    package_data={"tieval": ["VERSION"]},
    install_requires=[
        # "allennlp==2.9.3",
        "nltk",
        "xmltodict",
        "networkx>=2.8.1",
        "py_heideltime",
        "cached-path==1.1.2"
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires=">=3.8",
    extras_requires={
        "dev": [
            "py_test>=3.7",
            "check-manifest",
            "twine"
        ]
    },

)
