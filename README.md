# DockExplain

**Explainable molecular docking powered by Gemma, ProLIF, PyMOL, and literature-grounded AI reasoning.**

DockExplain is an AI-assisted molecular docking interpretation platform that turns raw docking outputs into understandable, evidence-aware scientific explanations. It combines molecular docking, protein-ligand interaction fingerprinting, 3D visualization, literature enrichment, and Gemma-based natural language explanation into one end-to-end workflow.

Instead of stopping at a docking score, DockExplain answers the more important question:

> **Why does this molecule bind to this protein, and which interactions matter?**

---

## Demo Summary

DockExplain takes a protein structure from the RCSB Protein Data Bank and a drug/molecule input, then generates:

- Docking score
- Best ligand pose
- Protein-ligand interaction table
- Binding-site residue analysis
- PyMOL visualization
- Literature-grounded explanation
- Gemma-generated scientific interpretation
- Downloadable PDF report
- JSON, TXT, CSV, SDF, PDB, PML, PNG, and ZIP outputs

---

## Why DockExplain?

Most docking tools produce technical outputs that are difficult for non-experts to interpret. DockExplain bridges that gap by converting computational docking results into clear, structured explanations for:

- Biology students
- Medicinal chemistry learners
- Computational chemistry researchers
- Drug discovery teams
- Hackathon demos
- AI-for-science prototypes

DockExplain does not just say *“the score is -7.2 kcal/mol.”*

It explains:

- What that score means
- Which residues interact with the ligand
- Whether the binding is driven by hydrogen bonding, hydrophobic packing, pi-stacking, ionic contacts, or other interactions
- Which residues are most important
- What the PyMOL visualization shows
- Whether literature evidence supports the interpretation
- What limitations should be considered

---

## Key Features

### 1. Protein Input from RCSB PDB

Enter an RCSB PDB ID such as:

```text
1M17
4COX
6LU7
