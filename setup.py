
from setuptools import find_packages
from setuptools import setup

setup(
    name="text2timeline",
    version="0.1.0",
    url="https://github.com/hmosousa/text2timeline",
    license='MIT',

    author="Hugo Sousa",
    author_email="hugo.o.sousa@inesctec.pt",

    description=" ",
    long_description=open("README.md").read(),

    packages=find_packages(exclude=('tests*',)),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
