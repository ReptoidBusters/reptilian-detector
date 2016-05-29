NAME = presentation
SRC = $(NAME).tex
TARGET = $(NAME).pdf
LATEX = pdflatex --shell-escape $(SRC)
TMP = *.aux *.log *.nav *.out *.snm *.toc *.vrb *-eps-converted-to.pdf

all: $(TARGET)

$(TARGET): *.tex
	$(LATEX)
	$(LATEX)
	rm -rf $(TMP)

.PHONY: clean
clean:
	rm -rf $(TARGET) $(TMP)
