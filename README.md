[![PyPi][badge-pypi]][link-pypi]
[![Docs][badge-docs]][link-docs]

[badge-pypi]: https://badge.fury.io/py/DeltaTopic.svg
[link-pypi]: https://pypi.org/project/DeltaTopic/
[badge-docs]: https://readthedocs.org/projects/deltatopic/badge/?version=latest
[link-docs]: https://deltatopic.readthedocs.io

# DeltaTopic: Dynamically-Encoded Latent Transcriptomic pattern Analysis by Topic modeling

## Installation

```
python setup.py build
python setup.py install
```

## Dependencies

* python - pandas, anndata, pytorch, pytorch_lightning

To replicate the analysis in the paper, R packages (Optional):

* R - data.table, dplyr, msigdbr, goseq, fgsea, ggplot2, ComplexHeatmap, circlize

## Example code for training BALSAM and DeltaTopic models

```
# train BALASM model on the spliced count data
BALSAM --nLV 32 --EPOCHS 5 
```
```
# train deltaTopic model
DeltaTopic --nLV 32 --EPOCHS 5 
```

### Analysis Rmd code for reproduce figures in the paper

```bash
cd R_figures
make all
```
