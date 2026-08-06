# Rapport PFE LaTeX — Guide Overleaf

## Contenu

Rapport complet **Cycle Ingénieur ESPRIT** — Projet Talys Consulting  
**Auteur :** Chedi Bouali  
**Encadrants :** Oussema Midouni, Hamza Ksentini (entreprise) | Salma Hajjem (académique)

## Structure (12 chapitres + annexes)

1. Introduction générale  
2. Contexte du projet  
3. Analyse de l'existant et solution proposée  
4. Spécifications des besoins  
5. Conception et architecture  
6. Données et feature engineering  
7. Modélisation ML (classique, séquentiel, graphe)  
8. Agent LangGraph  
9. RAG et rapports  
10. Réalisation (API + React)  
11. Tests et validation  
12. Conclusion et perspectives  
+ Annexes (features, install, REF-08, code, PlantUML, glossaire, biblio)

## Utilisation sur Overleaf

### Option A — Upload ZIP

1. Compresse le dossier `reports/latex_pfe/` en ZIP  
2. Overleaf → **New Project** → **Upload Project**  
3. Set main document : `main.tex`  
4. Compiler : **pdfLaTeX**

### Option B — Copier fichier par fichier

1. Crée un projet Blank sur Overleaf  
2. Upload `main.tex`, `preamble.tex`, `frontmatter.tex`, `annexes.tex`  
3. Crée le dossier `chapters/` et upload les 12 fichiers `.tex`  
4. Crée `figures/talys_logo.png` (logo Talys)

## Compiler en local

```bash
cd reports/latex_pfe
pdflatex main.tex
pdflatex main.tex   # 2e passe pour TOC
```

## Personnalisation

| Élément | Fichier |
|---------|---------|
| Page de garde | `frontmatter.tex` |
| Nom / encadrants | `frontmatter.tex` |
| Figures réelles | Remplace les `\todofig{...}` dans les chapitres |
| Logo | `figures/talys_logo.png` |

## Figures à ajouter (recommandé pour soutenance)

- Capture interface Explain (KPI + tableau)  
- Capture interface Chat (orchestration)  
- Courbes ROC (`reports/figures/` après entraînement)  
- Diagramme architecture (draw.io export PDF)  
- Rapport PDF généré  

Remplace dans les chapitres :

```latex
\todofig{Description}
```

par :

```latex
\begin{figure}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{figures/mon_capture.png}
  \caption{Description}
\end{figure}
```

## Volume visé

Avec le contenu actuel + figures réelles : **70–90 pages** en pdfLaTeX 12pt.

Pour allonger : ajoute des captures, développe le chapitre 7 (résultats chiffrés), ou insère un chapitre EDA avec graphiques.

## Fichiers du projet

```
latex_pfe/
├── main.tex
├── preamble.tex
├── frontmatter.tex
├── annexes.tex
├── figures/
│   └── talys_logo.png
└── chapters/
    ├── 01_introduction.tex
    ├── ...
    └── 12_conclusion.tex
```
