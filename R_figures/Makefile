# Define variables
RMD_FILES := $(wildcard *.Rmd)
HTML_FILES := $(RMD_FILES:%.Rmd=%.html)
MD_FILES := $(RMD_FILES:%.Rmd=%.md)
# Define rules
all: $(HTML_FILES)

%.html: %.Rmd
	Rscript -e "rmarkdown::render('$<')"

# remove HTML and MD files
clean:
	rm -f $(HTML_FILES)
	rm -r $(MD_FILES)
# remove output files
cleanoutput:
	rm -rf output/*