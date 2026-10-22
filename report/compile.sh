#!/bin/bash
# Compile the LaTeX report to PDF (needs pdflatex and biber).
cd "$(dirname "$0")" || exit 1

# First pass
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

# Bibliography
biber main > /dev/null 2>&1

# Second pass (for references)
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

# Third pass (for TOC)
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

# Clean aux files
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.bcf *.run.xml

echo "PDF compiled: $(pwd)/main.pdf"
