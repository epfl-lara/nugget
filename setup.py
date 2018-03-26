import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "nugget",
    version = "0.0.1",
    author = "Romain Edelmann",
    author_email = "romain.edelmann@epfl.ch",
    description = "Neural-Network Guided Expression Transformation",
    license = "GPLv3",
    keywords = "expression transformation embedding recursive neural network tree lstm tree-lstm",
    url = "https://github.com/epfl-lara/nugget",
    download_url = "https://github.com/epfl-lara/nugget/archive/0.0.1.tar.gz",
    packages=['nugget'],
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    project_urls={
        "Source": 'https://github.com/epfl-lara/nugget',
        "Tracker": 'https://github.com/epfl-lara/nugget/issues',
    },
    install_requires=[
        "torch>=0.3",
        "treenet=0.0.1",
    ]
)