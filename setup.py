# setup.py
from setuptools import setup, find_packages

setup(
    name="molconvert",
    version="0.2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "biopython",
        "rdkit",
    ],
    entry_points={
        "console_scripts": [
            "convert=molconvert.cli.main:run_convert",
            "rmsd=molconvert.cli.main:run_rmsd",
        ]
    },
)
