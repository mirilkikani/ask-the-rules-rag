# Data sources and licences

| File | Document | Source | Licence |
|---|---|---|---|
| eu_ai_act_2024_1689_en.pdf | Regulation (EU) 2024/1689 (AI Act), original OJ text of 12.7.2024, 144 pages | https://eur-lex.europa.eu/eli/reg/2024/1689/oj | CC BY 4.0, © European Union |
| gdpr_2016_679_en.pdf | Regulation (EU) 2016/679 (GDPR), 88 pages | https://eur-lex.europa.eu/eli/reg/2016/679/oj | CC BY 4.0, © European Union |
| dsk_oh_ki_datenschutz_2024_de.pdf | DSK Orientierungshilfe "Künstliche Intelligenz und Datenschutz" (6.5.2024), 15 pages | https://www.datenschutzkonferenz-online.de/media/oh/20240506_DSK_Orientierungshilfe_KI_und_Datenschutz.pdf | No licence stated on the website |
| dsk_oh_rag_2025_de.pdf | DSK Orientierungshilfe zu RAG-Systemen (Oct 2025), 18 pages | https://www.datenschutzkonferenz-online.de/media/oh/DSK_OH_RAG.pdf | No licence stated on the website |

Total: 265 pages, 4 files (checked with src/check_pdfs.py).

## Changes made
The PDFs were converted to text and split into chunks for search. The text itself was not edited.

## Rules for this repo
- Raw PDFs are NOT uploaded to GitHub (data/raw/ is in .gitignore).
- EU texts: credit "© European Union, EUR-Lex, CC BY 4.0".
- DSK texts: no licence found, so we link only to the official source and show only short quotes with page numbers.
- The AI Act is used in its original 2024 version; later amendments are not included.