<div align="center">

# DockExplain

**AI-powered molecular docking analysis with plain-language explanations**

*Built for the [Gemma 4 Good Hackathon]([https://ai.google.dev/competition](https://www.kaggle.com/competitions/gemma-4-good-hackathon) · Health & Sciences Track*

</div>

---

DockExplain is an end-to-end computational drug discovery tool that takes a protein structure and a drug molecule, performs molecular docking, validates the result across through thorough relevant web search and published literature, and generates a plain-language explanation grounded in live literature — all powered by Google's **Gemma 4** large language model running locally via Ollama.

It ships with both a **command-line interface** and a **Flask web application** with real-time pipeline streaming.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Pipeline Overview](#pipeline-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Interactive CLI](#interactive-cli)
  - [Non-interactive CLI](#non-interactive-cli)
  - [Web Interface](#web-interface)
- [Output Files](#output-files)
- [Module Reference](#module-reference)
- [Docking Modes](#docking-modes)
- [Configuration](#configuration)
- [Web Interface Details](#web-interface-details)

---

## How It Works

A typical DockExplain run for `4COX + Erlotinib` proceeds like this:

```
1.  User provides PDB ID (4COX) and drug name (Erlotinib)
2.  PDB downloaded from RCSB, co-crystal ligands detected (HEM, NAG, IMN)
3.  User selects binding site; drug SMILES resolved from RCSB CCD / PubChem
4.  PDBFixer cleans the protein; OpenBabel converts drug SMILES → PDBQT
5.  AutoDock Vina docks the drug (exhaustiveness=8, 10 poses scored)
6.  ProLIF fingerprints the best pose → residue-level interaction table
7.  5-tier validation checks structural integrity, chemistry, and pose quality
8.  PyMOL renders a colour-coded 2400×1800 PNG (colour = interaction type)
9.  PubChem, ChEMBL, UniProt, PubMed, DuckDuckGo enrichment gathered
10. Gemma 4 (gemma4:e2b) generates a 7-section explanation using all data
11. ReportLab generates a formatted PDF report with interaction table + bibliography
```

The total pipeline runtime for a typical run is **8–10 minutes** (dominated by PyMOL ray-tracing ~12 min and Gemma generation ~7 min).

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DockExplain Pipeline                      │
├────────┬────────────────────────────────────────────────────────┤
│ Step 1 │ Docking + ProLIF fingerprint                           │
│        │ AutoDock Vina → 10 poses → best SDF → interaction CSV  │
├────────┼────────────────────────────────────────────────────────┤
│ Step 2 │ 5-Tier Structural Validation                           │
│        │ Integrity · Chemistry · Binding site · Pose · Biology  │
├────────┼────────────────────────────────────────────────────────┤
│ Step 3 │ PyMOL Visualization                                    │
│        │ Colour-coded PNG + interactive .pse session            │
├────────┼────────────────────────────────────────────────────────┤
│ Step 4 │ Literature Enrichment                                  │
│        │ PubChem · ChEMBL · UniProt · PubMed · DuckDuckGo       │
├────────┼────────────────────────────────────────────────────────┤
│ Step 5 │ Gemma 4 Explanation                                    │
│        │ 7-section plain-language analysis grounded in sources  │
├────────┼────────────────────────────────────────────────────────┤
│ Step 6 │ PDF Report                                             │
│        │ ReportLab — interactions · validation · AI analysis    │
└────────┴────────────────────────────────────────────────────────┘
```

---

## Features

**Docking**
- Automatic binding site detection from co-crystal HETATM records
- Drug input accepts: drug name, PDB 3-letter code, or SMILES string
- Multi-source drug resolution: RCSB CCD → PubChem → direct SMILES
- AutoDock Vina with configurable exhaustiveness (4–32)
- Automatic engine selection: qvina2 → qvina-w → AutoDock Vina
- Three docking modes: redock co-crystal, new drug into crystal site, apo pocket prediction

**Visualization**
- PyMOL script generated automatically with colour-coded interaction types
- Amber = hydrophobic, Blue = H-bond donor, Indigo = H-bond acceptor, Purple = π-stacking
- Distance dashes for all polar contacts
- 2400×1800 PNG at 300 DPI, interactive `.pse` session file

**Literature**
- PubMed E-utilities search (no API key required)
- PubChem drug CID, description, synonyms
- ChEMBL mechanism of action
- UniProt protein function
- DuckDuckGo: mechanism, biology, drug–protein binding, clinical use
- All sources deduplicated and passed to Gemma as a structured context object

**AI Explanation (Gemma 4)**
- 7-section structured output:
  1. Opening — binding score interpretation
  2. Key Interactions — per-residue analysis with atom-level chemistry
  3. Multi-role Residues — residues contributing more than one interaction type
  4. Binding Pocket Character — pocket polarity, shape, and complementarity
  5. Literature Cross-check — consistency with published data
  6. Clinical Relevance — drug indication and off-target context
  7. Biological Plausibility Assessment — 5 criteria with PLAUSIBLE/UNLIKELY/UNCERTAIN verdicts

**Web Interface**
- Real-time pipeline streaming via Server-Sent Events
- 3-step wizard: protein fetch → drug resolve → configure & launch
- Live step tracker with running/done/error states
- Dark terminal console with colour-coded log lines
- One-click ZIP download of all result files
- PDF report, PyMOL PNG, and `.pml` script individually downloadable
- Polling fallback ensures redirect fires even if SSE stream drops

---

## Architecture

```
Gemma4Good/
├── python_scripts/
│   ├── run_dockexplain.py          # Main orchestrator + CLI
│   ├── input_handler.py            # PDB download, ligand detection, drug resolution
│   ├── dockexplain_pipeline.py     # Vina docking + ProLIF fingerprinting
│   ├── validation_suite.py         # 5-tier structural QC
│   ├── pymol_visualize.py          # PyMOL script generation + rendering
│   ├── pocket_predictor.py         # fpocket / P2Rank for apo structures
│   ├── web_search_enricher.py      # PubChem + ChEMBL + PubMed + DDG enrichment
│   ├── gemma_explainer.py          # Gemma 4 prompt building + explanation
│   ├── literature_validator.py     # PubMed RAG validation
│   ├── docking_engine.py           # Docking binary detection + unified interface
│   ├── explanation_pdf_generator.py # ReportLab PDF generation
│   ├── app.py                      # Flask web application
│   ├── static/
│   │   └── style.css               # Shared pastel theme (all pages)
│   └── templates/
│       ├── index.html              # 3-step input wizard
│       ├── progress.html           # Live pipeline progress page
│       ├── results.html            # Full results dashboard
│       └── error.html              # Error page
└── data/
    ├── 4COX.pdb                    # Downloaded PDB files
    └── results/
        └── erlotinib/
            ├── erlotinib_results.json          # Docking results + interactions
            ├── erlotinib_validation.json       # 5-tier QC report
            ├── erlotinib_web_enrichment.json   # Literature context
            ├── erlotinib_explanation.json      # Gemma 4 full output
            ├── erlotinib_explanation.txt       # Plain text version
            ├── erlotinib_explanation.pdf       # Formatted PDF report
            ├── erlotinib_visualization.png     # PyMOL render (2400×1800)
            ├── erlotinib_session.pse           # PyMOL session file
            ├── erlotinib_pymol.pml             # PyMOL script
            ├── erlotinib_interactions.csv      # Interaction fingerprint
            ├── erlotinib_best_pose.sdf         # Best docked pose (SDF)
            ├── erlotinib_best_pose.pdb         # Best docked pose (PDB)
            ├── 4COX_clean.pdb                  # PDBFixer-cleaned structure
            └── erlotinib_run_manifest.json     # Full run audit log
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥ 3.10 | Runtime |
| Conda / Mamba | Any | Environment management |
| AutoDock Vina | ≥ 1.2 | Molecular docking engine |
| OpenBabel | ≥ 3.1 | Ligand/receptor format conversion |
| Ollama | Any | Local LLM serving |
| Gemma 4 model | `gemma4:e2b` | AI explanation generation |
| PyMOL | ≥ 2.5 | 3D visualization (optional) |
| fpocket / P2Rank | Any | Pocket prediction for apo structures (optional) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DockExplain.git
cd DockExplain
```

### 2. Create the Conda environment

```bash
conda create -n dockexplain python=3.10
conda activate dockexplain
```

### 3. Install Python dependencies

```bash
# Core scientific stack
conda install -c conda-forge rdkit mdanalysis numpy pandas

# Docking
pip install vina meeko prolif

# Protein preparation
pip install pdbfixer openmm

# Literature + web search
pip install duckduckgo-search requests beautifulsoup4

# Visualization
pip install pymol-open-source    # or: conda install -c conda-forge pymol-open-source

# PDF generation
pip install reportlab

# Web interface
pip install flask

# Ollama Python client
pip install ollama
```

### 4. Install OpenBabel

```bash
# macOS
brew install open-babel

# Ubuntu / Debian
sudo apt install openbabel

# Conda (all platforms)
conda install -c conda-forge openbabel
```

### 5. Install AutoDock Vina

```bash
# Option A: pip (recommended)
pip install vina

# Option B: download binary
# https://github.com/ccsb-scripps/AutoDock-Vina/releases
# Add to PATH

# Option C: QuickVina 2 (faster, drop-in replacement)
# https://qvina.github.io/
# Download qvina2 binary, add to PATH
```

### 6. Install and configure Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the Gemma 4 model
ollama pull gemma4:e2b

# Start the Ollama server (keep this running)
ollama serve
```

> **Note**: `gemma4:e2b` requires approximately 5 GB of disk space and 8 GB of RAM. On Apple Silicon (M1/M2/M3) it runs efficiently via Metal acceleration.

---

## Usage

### Interactive CLI

The recommended way to run DockExplain. The wizard prompts for all inputs.

```bash
cd python_scripts/
python run_dockexplain.py
```

You will be guided through:
1. Enter a PDB ID (e.g. `4COX`) — the structure is downloaded automatically
2. View protein metadata and detected co-crystal ligands
3. Select a binding site
4. Enter the drug (name, PDB code, or SMILES string)
5. Confirm the job configuration

### Non-interactive CLI

For scripted or automated runs:

```bash
# Minimal — drug name resolved via PubChem
python run_dockexplain.py --pdb 4COX --drug Erlotinib

# With explicit SMILES (use for novel or research compounds)
python run_dockexplain.py --pdb 1M17 --drug Ibuprofen \
    --smiles "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

# Custom output directory and exhaustiveness
python run_dockexplain.py --pdb 6LU7 --drug Nirmatrelvir \
    --out ../data/results/nirmatrelvir \
    --exhaustiveness 16

# Offline mode (no web search, no PubMed)
python run_dockexplain.py --pdb 4COX --drug Celecoxib \
    --no-web-search

# Fast run (skip visualization and PDF)
python run_dockexplain.py --pdb 4COX --drug Aspirin \
    --skip-pymol --no-web-search --skip-explanation

# Headless PyMOL render (no GUI popup)
python run_dockexplain.py --pdb 4COX --drug Indomethacin --headless
```

**All CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--pdb ID` | — | RCSB PDB ID (downloaded automatically) |
| `--drug NAME` | — | Drug name, PDB code, or SMILES |
| `--smiles SMILES` | — | Override drug SMILES explicitly |
| `--out DIR` | `../data/results/<drug>/` | Output directory |
| `--exhaustiveness N` | `8` | Vina search exhaustiveness (4–32) |
| `--model MODEL` | `gemma4:e2b` | Ollama model string |
| `--papers N` | `10` | PubMed papers for RAG validation |
| `--skip-pymol` | off | Skip PyMOL visualization |
| `--no-web-search` | off | Skip literature enrichment |
| `--skip-explanation` | off | Skip Gemma 4 explanation |
| `--headless` | off | Render PyMOL PNG without opening GUI |

### Web Interface

```bash
cd python_scripts/
python app.py
```

Open **http://localhost:5001** in your browser.

The web interface provides a 3-step wizard:
- **Step 1** — Enter a PDB ID; the protein is fetched, metadata shown, and co-crystal ligands displayed as clickable cards.
- **Step 2** — Enter a drug; the SMILES is resolved live via RCSB / PubChem and shown for confirmation.
- **Step 3** — Set exhaustiveness, toggle PyMOL and web search, then launch. The progress page streams all pipeline output in real-time.

On completion, the results page shows: docking score card, interaction table with colour-coded type pills, 5-tier QC report with animated confidence bar, PyMOL visualization, the full Gemma 4 explanation rendered as Markdown, and a literature sources list. Every file is downloadable individually or as a single ZIP archive.

---

## Output Files

All files are written to `../data/results/<drug_name>/`:

| File | Description |
|------|-------------|
| `*_results.json` | Docking score, mode, all interactions, SMILES, protein name |
| `*_interactions.csv` | Residue-level ProLIF interaction fingerprint |
| `*_best_pose.sdf` | Best docked ligand pose (3D coordinates) |
| `*_best_pose.pdb` | Same pose in PDB format (for PyMOL) |
| `*_clean.pdb` | PDBFixer-prepared receptor (missing residues added, waters removed) |
| `*_validation.json` | 5-tier QC: per-check results, confidence score, overall status |
| `*_web_enrichment.json` | Drug summary, protein summary, PubMed papers, web sources |
| `*_explanation.json` | Full Gemma 4 output: 7 sections + bibliography |
| `*_explanation.txt` | Plain text version of the explanation |
| `*_explanation.pdf` | Formatted PDF report (ReportLab) |
| `*_visualization.png` | PyMOL render: 2400×1800 px at 300 DPI |
| `*_session.pse` | PyMOL session (open to explore interactively) |
| `*_pymol.pml` | PyMOL script (re-render or modify) |
| `*_run_manifest.json` | Audit log: all step statuses, elapsed times, timestamps |

---

## Module Reference

### `input_handler.py`

Handles all pre-pipeline user input and data resolution.

| Function | Description |
|----------|-------------|
| `download_pdb(pdb_id, out_dir)` | Downloads a PDB from RCSB; skips if already cached |
| `detect_ligands(pdb_path)` | Scans HETATM records; filters water, buffers, ions; returns list sorted by atom count |
| `fetch_ligand_info(code)` | Queries RCSB CCD for name, formula, SMILES of a 3-letter PDB code |
| `resolve_drug_input(drug_input)` | Resolves drug name / PDB code / SMILES → canonical SMILES via RCSB → PubChem → direct |
| `run_input_handler(pdb_data_dir)` | Full interactive CLI wizard; returns job dict with `pdb_path`, `drug_name`, `drug_smiles` |

### `dockexplain_pipeline.py`

Core docking and interaction analysis engine.

| Function | Description |
|----------|-------------|
| `run_pipeline(raw_protein_pdb, drug_name, ...)` | Full pipeline: detect ligand → fix protein → prepare ligand → Vina dock → ProLIF fingerprint |
| `find_ligand_code(pdb_path)` | Identifies the co-crystal drug from HETATM (largest non-solvent residue) |
| `get_binding_site_center(pdb_path, ligand_code)` | Returns (x, y, z) centroid of the co-crystal ligand for Vina box positioning |
| `prepare_protein_pdbfixer(pdb_path, out_dir)` | Adds missing residues/atoms, removes HETATM, writes cleaned PDB |
| `prepare_ligand_smiles(smiles, out_dir, drug_name)` | RDKit 3D embedding → Meeko PDBQT conversion |
| `run_vina_docking(receptor_pdbqt, ligand_pdbqt, center, out_dir)` | Runs AutoDock Vina; returns list of (score, pdbqt_path) tuples |
| `extract_best_pose_sdf(vina_out, out_dir, drug_name)` | Converts best Vina PDBQT pose → SDF and PDB |
| `run_prolif_fingerprint(clean_pdb, pose_sdf)` | ProLIF + MDAnalysis interaction fingerprint; returns list of interaction dicts |

**Docking mode detection is automatic:**

| Mode | Condition | Approach |
|------|-----------|----------|
| A | Co-crystal PDB + no new SMILES | Redock the co-crystal ligand into its own site (validation) |
| B | Co-crystal PDB + new drug SMILES | Dock new drug into the co-crystal binding site |
| C | Apo PDB (no co-crystal ligand) | Predict pockets with fpocket/P2Rank, dock into top pocket |


### `pymol_visualize.py`

Generates PyMOL scripts and renders visualizations.

**Interaction colour scheme:**

| Type | Colour |
|------|--------|
| Hydrophobic | Amber `#FF8F00` |
| H-Bond Donor | Dark blue `#1565C0` |
| H-Bond Acceptor | Blue `#1976D2` |
| π–π Stacking | Purple `#6A1B9A` |
| Edge-to-Face | Light purple `#8E24AA` |
| Cationic | Dark red `#C62828` |
| Anionic | Pink `#AD1457` |
| Cation–π | Cyan `#00838F` |

Distance dashes are drawn for all polar contact types (H-bond, ionic, cation–π).

### `web_search_enricher.py`

Multi-source literature enrichment producing a `LiteratureContext` dataclass.

| Source | Data Retrieved |
|--------|---------------|
| PubChem | Drug CID, canonical name, synonyms, description, molecular weight |
| ChEMBL | Mechanism of action, primary target, drug type |
| UniProt | Protein function, subcellular location, disease associations |
| PubMed (E-utilities) | Abstracts via esearch → efetch; 4 queries per run |
| DuckDuckGo | 4 targeted searches: mechanism, protein biology, drug–protein binding, clinical use |

The `LiteratureContext` object is passed directly into Gemma's prompt builder — data is fetched exactly once per run.

### `gemma_explainer.py`

Builds a richly structured prompt and calls Gemma 4 via Ollama.

The prompt includes:
- Full interaction table with residue, type, description, chain
- Docking score and all pose metrics
- ProLIF fingerprint details and validation warnings
- Full `LiteratureContext` formatted as labelled sections
- PyMOL visualization guide (what the researcher will see)
- Atom-level multi-role residue guidance (e.g. LYS: hydrophobic chain + ionic amine)

Output is a 7-section Markdown document saved as both JSON and `.txt`.

### `pocket_predictor.py`

Wraps fpocket and P2Rank for apo protein analysis (Mode C).

```python
from pocket_predictor import predict_pockets

pockets = predict_pockets(
    protein_pdb='4COX.pdb',
    method='auto',        # 'fpocket', 'p2rank', or 'auto'
    n_top_pockets=3,
    verbose=True,
)
# Returns list of Pocket(center, score, volume, residues, method, confidence)
```

### `docking_engine.py`

Auto-detects and wraps available docking binaries.

Priority order: `qvina2` → `qvina-w` → `vina` → `vina_1.2.5` → AutoDock Vina Python API. All engines use identical configuration file format and are fully interchangeable.

### `explanation_pdf_generator.py`

Converts `*_explanation.json` → formatted A4 PDF using ReportLab.

Includes: cover page with drug/protein/score summary, interaction table with colour-coded type column, Gemma 4 explanation rendered with section headers, validation QC summary with confidence bar, bibliography of all literature sources.

---

## Docking Modes

DockExplain auto-detects which mode applies based on the inputs:

**Mode A — Co-crystal Redock (Validation)**

Used when a co-crystal PDB is provided with no additional drug. The co-crystal ligand is re-docked into its own binding site. RMSD of the best pose vs. the crystal pose measures docking accuracy. A good redock RMSD < 2.0 Å validates the pipeline setup.

**Mode B — New Drug into Co-crystal Site**

Used when a co-crystal PDB is provided alongside a different drug SMILES. The binding site center is taken from the co-crystal ligand coordinates. This is the standard drug repurposing / screening workflow.

**Mode C — Apo Structure with Pocket Prediction**

Used when the PDB has no drug-like HETATM records. fpocket or P2Rank identifies the top druggable pockets, which are ranked by druggability score. The top pocket is used as the docking box center by default; a pocket selection menu is shown in interactive mode.

---

## Validation Suite

The 5-tier validation returns a confidence score and per-check breakdown:

```
══════════════════════════════════════════════════════
DockExplain — Validation Report
  Drug    : Erlotinib
  Protein : CYCLOOXYGENASE-2
══════════════════════════════════════════════════════

── Tier 1: Structural Integrity ──
⚠ [1.1] RMSD (structural distortion)
    Large RMSD (75.75 Å > 2.0 Å). PDBFixer may have introduced
    significant structural changes.
✓ [1.2] Backbone atom conservation — 8816 atoms preserved
✓ [1.3] Chain identifier preservation — [A, B, C, D]
✓ [1.4] Alternate location handling — none found
✓ [1.5] Residue numbering continuity — no gaps

── Tier 2: Chemistry Correctness ──
✗ [2.1] Protonation states at pH 7.4 — 9 issues found
✓ [2.2] Histidine tautomers — no HIS within 10Å
✓ [2.4] Disulfide bond preservation — 20 bonds intact
✓ [2.5] Chirality and planarity — 2208 residues OK

── Summary ──
PASS : 13   WARN : 3   FAIL : 3   SKIP : 1
Confidence score : 76.3%
Overall status   : FAIL
══════════════════════════════════════════════════════
```

---

## Configuration

Key parameters configurable via CLI flags or the web interface:

| Parameter | Default | Notes |
|-----------|---------|-------|
| Vina exhaustiveness | `8` | Higher = more thorough but slower. `32` for publication-quality runs |
| Vina box size | `20 × 20 × 20 Å` | Centred on co-crystal ligand or predicted pocket |
| ProLIF cutoff | Default | Distance cutoffs per interaction type (MDAnalysis defaults) |
| pH for protonation check | `7.4` | Physiological; not configurable without code edit |
| Pocket proximity (cofactor metals) | `5 Å` | Warn if HEM/ZN removed within this radius of binding site |
| PyMOL render resolution | `2400 × 1800 px` | `ray_trace_mode=1`, `antialias=2` |
| Gemma context window | `~8k tokens` | Prompt size grows with number of interactions and sources |
| PubMed papers | `10` | Papers fetched per query; 4 queries run = up to 40 unique PMIDs |

---

## Web Interface Details

The Flask app (`app.py`) runs on **port 5001** by default.

**Key routes:**

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Input wizard (index.html) |
| `/progress/<job_id>` | GET | Live progress page with SSE stream |
| `/results/<job_id>` | GET | Full results dashboard |
| `/api/fetch-protein` | POST | Download PDB, return metadata + ligands |
| `/api/resolve-drug` | POST | Resolve drug input → canonical SMILES |
| `/api/start-job` | POST | Create job, spawn pipeline subprocess |
| `/api/stream/<job_id>` | GET | Server-Sent Events stream (live log) |
| `/api/job/<job_id>` | GET | Job status JSON (for polling fallback) |
| `/api/image/<job_id>` | GET | Serve PyMOL PNG |
| `/api/pdf/<job_id>` | GET | Download PDF report |
| `/api/pml/<job_id>` | GET | Download PyMOL script |
| `/api/zip/<job_id>` | GET | Download all result files as ZIP |

**Real-time streaming architecture:**

The pipeline runs as a subprocess with `stdout` captured line by line. Each line is stripped of ANSI escape codes, pushed to a per-job `queue.Queue`, and forwarded to the browser via an SSE event stream. Step transitions are detected by matching keywords (`STEP 1`, `STEP 2`, etc.) in the output. A polling fallback hits `/api/job/<job_id>` every 4 seconds, ensuring the results redirect fires even if the SSE connection drops.

---

## Acknowledgements

- **Google DeepMind** for the Gemma 4 model and the Gemma 4 Good Hackathon
- **The Scripps Research Institute** for AutoDock Vina
- **RCSB Protein Data Bank** for open access to structural data and the CCD API
- **OpenMM / PDBFixer, OpenBabel** team for protein preparation tools
- **MDAnalysis + ProLIF** team for molecular interaction fingerprinting
- **Schrödinger** for PyMOL open-source
- **NCBI** for open access to PubMed E-utilities

---

<div align="center">
<sub>Built with ❤️ for the Gemma 4 Good Hackathon · Health & Sciences Track</sub>
</div>
