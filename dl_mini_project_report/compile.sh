#!/bin/bash
# Compile LaTeX to PDF
cd /home/me/Desktop/esi/dl/mini-proj/latex_report

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

echo "PDF compiled: /home/me/Desktop/esi/dl/mini-proj/latex_report/main.pdf"
