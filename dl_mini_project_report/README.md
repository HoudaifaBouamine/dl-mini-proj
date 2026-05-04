# LaTeX Project - ECG Arrhythmia Classification

This folder contains the complete LaTeX source for the mini-project report.

## File Structure

```
latex_report/
├── main.tex                      # Main document
├── packages.tex                  # Package imports
├── compile.sh                    # Compilation script
├── Assets/
│   └── Logo_ESI_SBA.png       # ESI logo
├── FrontMatter/
│   ├── Titlepage.tex           # Title page
│   ├── Abstract.tex            # Abstract
│   └── Acronyms.tex           # Acronyms
├── Chapters/
│   ├── 01_Introduction.tex    # Introduction
│   ├── 02_Background.tex      # Background & Related Work
│   ├── 03_Implementation.tex   # Methodology & Implementation
│   ├── 04_Results.tex         # Results & Evaluation
│   └── 05_Conclusion.tex      # Conclusion & Future Work
└── Bibliography/
    └── bibliography.bib         # Bibliography entries
```

## How to Compile

### Option 1: Local LaTeX Installation
```bash
cd /home/me/Desktop/esi/dl/mini-proj/latex_report
./compile.sh
```

### Option 2: Online LaTeX (Overleaf, etc.)
1. Upload the entire `latex_report/` folder to Overleaf
2. Set `main.tex` as the main document
3. Click "Recompile"

### Option 3: Using Kaggle/Colab (if LaTeX installed)
```python
!cd /kaggle/working/latex_report && bash compile.sh
```

## Required LaTeX Packages
- texlive-latex-extra (includes extreport, babel, etc.)
- biber (bibliography backend)
- fonts-lmodern (Latin Modern fonts)

## Output
The compiled PDF will be: `main.pdf`

## Notes
- All paths are relative to `main.tex`
- The ESI logo is expected in `Assets/Logo_ESI_SBA.png`
- Bibliography uses biber backend (not bibtex)
