# Instructions on how to reproduce the figures in paper XXX

## Step 1: get weights matrices for deltaTopic and Balsam
```
Rscript -e "rmarkdown::render('_0_get_weights.Rmd')"
```

## Step 2: knit any of `_FigX_XXX.Rmd` to produce the corresponding figures
```
Rscript -e "rmarkdown::render('_FigX_XXX.Rmd')"
```
