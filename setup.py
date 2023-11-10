from setuptools import find_packages
from setuptools import setup

with open("README.md", encoding="utf-8") as f:
    README = f.read()

setup(
    name="tieval",
    version='{{VERSION_PLACEHOLDER}}',
    url="https://github.com/LIAAD/tieval",
    license='MIT',
    author="Hugo Sousa",
    author_email="hugo.o.sousa@inesctec.pt",
    description=
    "This framework facilitates the development and test of temporal-aware models.",
    long_description_content_type='text/markdown',
    long_description=README,
    packages=find_packages(exclude=('tests*',)),
    install_requires=[
        "allennlp==2.9.3",
        "nltk",
        "tabulate",
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
