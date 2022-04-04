
from setuptools import find_packages
from setuptools import setup


README = open("README.md").read()


setup(
    name="tieval",
    version="0.0.2",
    url="https://github.com/LIAAD/tieval",
    license='MIT',

    author="Hugo Sousa",
    author_email="hugo.o.sousa@inesctec.pt",

    description=
    "This framework facilitates the development and test of temporal aware models",

    long_description_content_type='text/markdown',

    long_description=README,

    packages=find_packages(exclude=('tests*',)),

    install_requires=[],

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
