import os
from setuptools import setup, find_packages

setup(
    name='DeltaTopic',
    packages=find_packages(),
    author='Yichen Zhang',
    version='0.0.1',
    description="Packages to implement BALSAM and DeltaTopic as described in the paper: Unraveling dynamically-encoded latent transcriptomic patterns in pancreatic cancer cells by topic modelling",
    url="https://github.com/causalpathlab/deltaTopic",
    entry_points={'console_scripts':
        ['BALSAM = run.BALSAM:main',
        'DeltaTopic = run.DeltaTopic:main']
    },
    install_requires=['pandas==1.4.1',
                      'torch', 
                      'h5py==3.6.0', 
                      'numpy==1.21.5', 
                      'anndata==0.7.8',
                      'pytorch_lightning==1.9.0', 
                      'scanpy', 
                      'scipy ==1.8.0']
)