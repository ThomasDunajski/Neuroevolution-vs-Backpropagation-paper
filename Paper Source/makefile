MASTER=paper

all: clean
	pdflatex $(MASTER).tex
ifneq ("$(wildcard ./${MASTER}.bib)","")
	bibtex $(MASTER)
	pdflatex $(MASTER).tex
endif
	pdflatex $(MASTER).tex

clean:
	@rm -rf *.aux *.bbl *.blg *.brf *.log *.lof *.lot *.lol *.out *.tcp *.toc *.tps *.bak *.backup *.pdfsync *.synctex.gz *.*~
	@for i in aux bbl blg brf log lof lot lol out tcp toc tps bak backup pdfsync synctex.gz; do find -name *.$$i -exec rm {} + ; done
	@find -name *.*~ -exec rm {} +

cleanall: clean
	@rm *.pdf *.dvi

press: all
	@ps2pdf14 -dPDFSETTINGS=/prepress $(MASTER).pdf
	@mv $(MASTER).pdf.pdf $(MASTER).pdf
