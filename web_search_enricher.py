# web_search_enricher.py (FIXED with better protein name extraction)
# DockExplain — unified literature enrichment
#
# KEY FIX:
# - resolve_protein_name() now uses the FULL COMPND description from PDB
#   instead of just the classification tag ("DNA" → full G-quadruplex description)
# - Falls back through: RCSB → result_json → FULL PDB COMPND → raw name
# - Searches use meaningful protein context instead of generic terms

import time
import textwrap
import argparse
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import warnings as _warnings

# ── DuckDuckGo (renamed package, graceful fallback) ──────────────────────────
_DDGS_AVAILABLE = False
_DDGS_CLASS     = None

try:
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        from ddgs import DDGS as _DDGS_NEW
    _DDGS_CLASS     = _DDGS_NEW
    _DDGS_AVAILABLE = True
except ImportError:
    pass

if not _DDGS_AVAILABLE:
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            from duckduckgo_search import DDGS as _DDGS_OLD
        _DDGS_CLASS     = _DDGS_OLD
        _DDGS_AVAILABLE = True
    except ImportError:
        pass

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Paper:
    """A single peer-reviewed paper with full citation metadata."""
    pmid    : str
    year    : str
    title   : str
    authors : str   # "Smith J, et al."
    journal : str
    abstract: str
    url     : str

    def citation(self) -> str:
        """Return a formatted citation string."""
        return (
            f"{self.authors} ({self.year}). {self.title}. "
            f"{self.journal}. PMID: {self.pmid}. {self.url}"
        )


@dataclass
class SearchResult:
    title  : str
    url    : str
    snippet: str

    def citation(self) -> str:
        return f"{self.title}. {self.url}"


@dataclass
class LiteratureContext:
    """
    All evidence gathered for a drug–protein pair.
    Passed directly into gemma_explainer.build_prompt().
    """
    # ── Structured content for the prompt ───────────────
    drug_summary        : str  = ""
    protein_summary     : str  = ""
    interaction_evidence: str  = ""
    clinical_context    : str  = ""

    # ── Identity fields ──────────────────────────────────
    gene_symbol         : str  = ""
    protein_display_name: str  = ""
    drug_display_name   : str  = ""

    # ── Full paper records (for bibliography) ────────────
    papers              : list = field(default_factory=list)   # list[Paper]
    web_sources         : list = field(default_factory=list)   # list[SearchResult]

    # ── Flat URL list (legacy, kept for compatibility) ───
    sources             : list = field(default_factory=list)

    warnings            : list = field(default_factory=list)

    def all_citations(self) -> list[str]:
        """Return all citations (papers + web) as formatted strings."""
        citations = []
        for i, p in enumerate(self.papers, 1):
            citations.append(f"[{i}] {p.citation()}")
        offset = len(self.papers)
        for i, w in enumerate(self.web_sources, offset + 1):
            citations.append(f"[{i}] {w.citation()}")
        return citations


# ════════════════════════════════════════════════════════════════════════════
# RCSB GENE SYMBOL RESOLVER
# ════════════════════════════════════════════════════════════════════════════

def fetch_rcsb_gene_name(pdb_id: str) -> tuple[str, str]:
    """
    Query RCSB REST API for the gene symbol and full protein title.
    Returns (gene_symbol, full_name), either may be "" on failure.
    """
    if not _REQUESTS_AVAILABLE or not pdb_id:
        return "", ""

    pdb_id    = pdb_id.strip().upper()
    gene_sym  = ""
    full_name = ""

    try:
        r = requests.get(
            f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}",
            timeout=8
        )
        if r.status_code == 200:
            full_name = r.json().get("struct", {}).get("title", "").strip()
    except Exception:
        pass

    for entity_idx in ("1", "2"):
        if gene_sym:
            break
        try:
            r = requests.get(
                f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_idx}",
                timeout=8
            )
            if r.status_code != 200:
                continue
            entity = r.json()
            for org in (entity.get("rcsb_entity_source_organism") or []):
                for gn in (org.get("rcsb_gene_name") or []):
                    val = gn.get("value", "").strip()
                    if val:
                        gene_sym = val; break
                if gene_sym:
                    break
            if not gene_sym:
                for s in (entity.get("entity_src_gen") or []):
                    val = s.get("pdbx_gene_src_gene", "").strip()
                    if val:
                        gene_sym = val.split(",")[0].strip(); break
        except Exception:
            continue

    return gene_sym, full_name


def resolve_protein_name(
    protein_raw: str,
    result_json: dict | None = None,
    pdb_id     : str | None  = None,
    pdb_path   : str | None  = None,
) -> tuple[str, str]:
    """
    Return (search_name, display_name) using best available source.
    
    Priority:
      1. RCSB REST API (gene symbol + title) — most reliable
      2. result_json metadata fields (gene_symbol, protein_uniprot_name, etc.)
      3. PDB file COMPND field (full description) — NEW: improved fallback
      4. protein_raw (classification tag like "DNA")
    
    The search_name is what DDG/PubMed will use — should be meaningful
    (e.g. "MYC promoter G-quadruplex" not just "DNA").
    
    The display_name is what Gemma will show in the explanation.
    """
    
    # ── 1. Try RCSB REST API (most authoritative) ──────────────────────
    if pdb_id:
        gene_sym, full_name = fetch_rcsb_gene_name(pdb_id)
        if gene_sym:
            return gene_sym, (full_name or gene_sym)
        # If RCSB has a full_name but no gene_sym (e.g. for non-proteins),
        # use the full_name as search term
        if full_name:
            return full_name.strip(), full_name.strip()

    # ── 2. Try metadata fields in result_json ──────────────────────────
    if result_json:
        # Check for explicitly stored protein names
        for field_name in (
            "protein_full_name",       # NEW: if pipeline stores this
            "protein_compnd",          # NEW: COMPND field from PDB
            "protein_uniprot_name",
            "protein_gene",
            "gene_symbol",
            "gene",
        ):
            val = result_json.get(field_name, "")
            if val and len(val) > 5 and isinstance(val, str):
                return val.strip(), val.strip()

    # ── 3. Extract full COMPND from PDB file (NEW) ─────────────────────
    # This is the fallback that fixes the "DNA" → full description issue.
    # Read the actual PDB file and get the complete COMPND description.
    if pdb_path:
        full_compnd = _extract_compnd_from_pdb(pdb_path)
        if full_compnd and len(full_compnd) > 5:
            # Use full COMPND for searching (much more specific than "DNA")
            return full_compnd, full_compnd

    # ── 4. Fallback to raw name (minimal but present) ──────────────────
    fallback = (protein_raw or "Unknown protein").strip()
    return fallback, fallback


def _extract_compnd_from_pdb(pdb_path: str) -> str:
    """
    Extract the full COMPND field from a PDB file.
    The COMPND field can span multiple lines and describes the actual
    structure (e.g. "Solution structure of the MYC promoter G-quadruplex").
    
    Returns the concatenated COMPND text, or "" if not found.
    """
    if not pdb_path or not __import__("os").path.exists(pdb_path):
        return ""
    
    compnd_lines = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("COMPND"):
                    # COMPND lines have format: "COMPND   text here"
                    # Extract the text part (after column 10)
                    text = line[10:].strip()
                    # Remove trailing semicolons/commas
                    text = text.rstrip(";,").strip()
                    if text and text not in ("MOL_ID", "MOLECULE", "CHAIN",
                                             "ENGINEERED", "SYNTHETIC"):
                        compnd_lines.append(text)
    except Exception:
        pass
    
    # Join all COMPND lines and clean up
    full_compnd = " ".join(compnd_lines).strip()
    
    # Remove field labels that ended up in the concatenation
    import re
    full_compnd = re.sub(r"\s*(MOL_ID|MOLECULE|CHAIN):\s*", " ", full_compnd)
    full_compnd = re.sub(r"\s+", " ", full_compnd).strip()
    
    return full_compnd


# ════════════════════════════════════════════════════════════════════════════
# PUBMED — peer-reviewed literature (replaces literature_validator.py)
# ════════════════════════════════════════════════════════════════════════════

def _pubmed_search(query: str, max_results: int = 5) -> list[str]:
    """Search PubMed and return PMIDs. Uses NCBI E-utilities (no key needed)."""
    base   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    params = urllib.parse.urlencode({
        "db": "pubmed", "term": query,
        "retmax": max_results, "sort": "relevance", "retmode": "json",
    })
    try:
        with urllib.request.urlopen(
            f"{base}esearch.fcgi?{params}", timeout=10
        ) as r:
            return json.loads(r.read()).get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []


def _pubmed_fetch(pmids: list[str]) -> list[Paper]:
    """Fetch full Paper records for a list of PMIDs."""
    if not pmids:
        return []

    base   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    params = urllib.parse.urlencode({
        "db": "pubmed", "id": ",".join(pmids),
        "rettype": "abstract", "retmode": "xml",
    })
    try:
        with urllib.request.urlopen(
            f"{base}efetch.fcgi?{params}", timeout=15
        ) as r:
            xml_data = r.read()
    except Exception:
        return []

    papers = []
    try:
        root = ET.fromstring(xml_data)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            pmid    = pmid_el.text if pmid_el is not None else "?"

            title_el = article.find(".//ArticleTitle")
            title    = (title_el.text or "No title") if title_el is not None else "No title"

            year_el  = article.find(".//PubDate/Year")
            year     = year_el.text if year_el is not None else "?"

            # Authors: "LastName Initials" for first 3, then "et al."
            author_els = article.findall(".//Author")[:3]
            author_parts = []
            for a in author_els:
                last = a.findtext("LastName", "")
                init = a.findtext("Initials", "")
                if last:
                    author_parts.append(f"{last} {init}".strip())
            authors = ", ".join(author_parts)
            if len(article.findall(".//Author")) > 3:
                authors += ", et al."

            journal_el = article.find(".//Journal/Title")
            journal    = journal_el.text if journal_el is not None else ""
            if not journal:
                journal_el = article.find(".//MedlineTA")
                journal    = journal_el.text if journal_el is not None else ""

            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (el.text or "") for el in abstract_parts if el.text
            )

            if abstract:
                papers.append(Paper(
                    pmid     = pmid,
                    year     = year,
                    title    = title,
                    authors  = authors or "Unknown authors",
                    journal  = journal or "Unknown journal",
                    abstract = abstract[:1800],
                    url      = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                ))
    except Exception:
        pass

    return papers


def gather_pubmed_literature(
    drug        : str,
    protein     : str,
    gene_symbol : str,
    interactions: list[dict],
    n_papers    : int = 8,
    verbose     : bool = True,
) -> list[Paper]:
    """
    Build targeted PubMed queries from drug + protein + key residues
    and return deduplicated Paper records.

    Uses 3 query strategies:
      1. drug + gene_symbol + "binding interactions"
      2. drug + gene_symbol + top H-bond residues
      3. gene_symbol + "crystal structure" + "binding site"
    """
    def log(msg):
        if verbose:
            print(f"  [pubmed] {msg}")

    # Use gene symbol for searching if available (much more precise than
    # the raw PDB COMPND string like "TRANSFERASE")
    search_protein = gene_symbol if gene_symbol else protein

    # Key H-bond residues from ProLIF output
    hbond_res = [
        ix["residue"].split(".")[0]
        for ix in interactions
        if ix.get("interaction_type") in ("HBDonor", "HBAcceptor")
    ][:3]

    # For novel compounds (drug=""), use only protein-context queries.
    # Drug-name queries on a user-assigned name like "Gen-LPAR1-007"
    # return nothing useful — the protein literature is the only
    # grounding available.
    if not drug:
        queries = [
            f"{search_protein} binding site crystal structure small molecule",
            f"{search_protein} inhibitor binding pocket selectivity",
            f"{search_protein} structure function ligand",
        ]
    else:
        queries = [
            # Query 1: protein binding site context — no drug name.
            f"{search_protein} binding site crystal structure small molecule",

            # Query 2: drug + protein
            f"{drug} {search_protein} binding",

            # Query 3: drug pharmacology
            f"{drug} pharmacology mechanism receptor target",
        ]

    seen_pmids : set[str] = set()
    all_pmids  : list[str] = []

    for q in queries:
        log(f"Query: {q[:70]}...")
        pmids = _pubmed_search(q, max_results=4)
        for p in pmids:
            if p not in seen_pmids:
                all_pmids.append(p)
                seen_pmids.add(p)
        time.sleep(0.35)   # NCBI rate limit: ≤3 req/sec

    all_pmids = all_pmids[:n_papers]
    log(f"Found {len(all_pmids)} unique PMIDs — fetching abstracts...")

    papers = _pubmed_fetch(all_pmids)
    log(f"✓ {len(papers)} abstracts retrieved")

    if verbose and papers:
        for p in papers:
            print(f"    [{p.pmid}] ({p.year}) {p.title[:65]}...")

    return papers


# ════════════════════════════════════════════════════════════════════════════
# REST API BACKENDS
# ════════════════════════════════════════════════════════════════════════════

def _pubchem_cid_from_name(drug_name: str) -> int | None:
    if not _REQUESTS_AVAILABLE:
        return None
    import re as _re
    candidates = [drug_name, _re.sub(r"^[A-Z]{2,6}[-_]", "", drug_name)]
    for name in dict.fromkeys(candidates):
        try:
            r = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                f"name/{requests.utils.quote(name)}/JSON",
                timeout=8
            )
            if r.status_code == 200:
                return r.json()["PC_Compounds"][0]["id"]["id"]["cid"]
        except Exception:
            continue
    return None


def _pubchem_cid_from_smiles(smiles: str) -> int | None:
    if not _REQUESTS_AVAILABLE or not smiles:
        return None
    try:
        r = requests.post(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/JSON",
            data={"smiles": smiles}, timeout=10
        )
        if r.status_code == 200:
            return r.json()["PC_Compounds"][0]["id"]["id"]["cid"]
    except Exception:
        pass
    return None


def _fetch_pubchem_drug(drug_name: str, drug_smiles: str = "") -> str:
    if not _REQUESTS_AVAILABLE:
        return ""
    cid = _pubchem_cid_from_name(drug_name)
    if not cid and drug_smiles:
        cid = _pubchem_cid_from_smiles(drug_smiles)
    if not cid:
        return ""
    try:
        d = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"cid/{cid}/description/JSON", timeout=8
        ).json()
        for section in d.get("InformationList", {}).get("Information", []):
            text = section.get("Description", "")
            if text and len(text) > 60:
                return text[:1200]
    except Exception:
        pass
    return ""


def _fetch_uniprot_protein(gene_symbol: str) -> str:
    if not _REQUESTS_AVAILABLE:
        return ""
    try:
        r = requests.get(
            "https://rest.uniprot.org/uniprotkb/search"
            f"?query=gene:{gene_symbol}+AND+reviewed:true"
            f"&fields=function,protein_name&format=json&size=1",
            timeout=8
        )
        if r.status_code != 200:
            return ""
        entries = r.json().get("results", [])
        if not entries:
            return ""
        for c in entries[0].get("comments", []):
            if c.get("commentType") == "FUNCTION":
                texts = c.get("texts", [])
                if texts:
                    return texts[0].get("value", "")[:1200]
    except Exception:
        pass
    return ""


def _fetch_chembl_moa(drug: str, drug_smiles: str = "") -> str:
    if not _REQUESTS_AVAILABLE:
        return ""
    import re as _re

    def _mechs(name):
        try:
            r = requests.get(
                "https://www.ebi.ac.uk/chembl/api/data/mechanism"
                f"?drug_name={requests.utils.quote(name)}&format=json&limit=3",
                timeout=8
            )
            return r.json().get("mechanisms", []) if r.status_code == 200 else []
        except Exception:
            return []

    mechs = _mechs(drug)
    if not mechs:
        stripped = _re.sub(r"^[A-Z]{2,6}[-_]", "", drug)
        if stripped != drug:
            mechs = _mechs(stripped)

    if not mechs and drug_smiles:
        try:
            r = requests.get(
                "https://www.ebi.ac.uk/chembl/api/data/molecule"
                f"?molecule_structures__canonical_smiles__flexmatch="
                f"{requests.utils.quote(drug_smiles)}&format=json&limit=1",
                timeout=10
            )
            if r.status_code == 200:
                mols = r.json().get("molecules", [])
                if mols:
                    chembl_id = mols[0].get("molecule_chembl_id", "")
                    if chembl_id:
                        r2 = requests.get(
                            "https://www.ebi.ac.uk/chembl/api/data/mechanism"
                            f"?molecule_chembl_id={chembl_id}&format=json&limit=3",
                            timeout=8
                        )
                        if r2.status_code == 200:
                            mechs = r2.json().get("mechanisms", [])
        except Exception:
            pass

    return "\n".join(
        f"- {m.get('mechanism_of_action','')} (target: {m.get('target_name','')})"
        for m in mechs[:3] if m.get("mechanism_of_action")
    )


def _ddg_search(query: str, max_results: int = 5) -> list[SearchResult]:
    if not _DDGS_AVAILABLE:
        return []
    results = []
    try:
        with _DDGS_CLASS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(SearchResult(
                    title   = r.get("title", ""),
                    url     = r.get("href", ""),
                    snippet = r.get("body", ""),
                ))
    except Exception:
        pass
    return results


# ════════════════════════════════════════════════════════════════════════════
# DRUG DISPLAY NAME RESOLVER
#
# The pipeline JSON may store the drug as any of:
#   • A PDB 3-letter code             → "ON3"
#   • A PubChem substance registry ID → "RefChem:168278"
#   • A full IUPAC name               → "[6,7-BIS(...)]..."
#   • A research code                 → "ONO-3080573"
#   • A clean INN name                → "Erlotinib"
#
# Strategy (in priority order):
#   1. SMILES → PubChem CID → synonyms  (most reliable when SMILES known)
#   2. PDB code → PubChem CID → synonyms
#   3. Drug name → PubChem CID → synonyms
#   4. Return original name as-is
#
# The resolved name is stored in lit_ctx.drug_display_name and used for
# ALL downstream searches (DDG, PubMed) and in the Gemma prompt.
# ════════════════════════════════════════════════════════════════════════════

_INN_STEMS = (
    "inib", "mab", "zumab", "ximab", "umab", "tinib", "ciclib",
    "rafenib", "buvir", "previr", "navir", "lukast", "sartan",
    "pril", "olol", "dipine", "statin", "mycin", "cycline",
    "floxacin", "azole", "conazole", "thiazide", "oxacin", "cillin",
    "pam", "micin", "bactam", "penem", "semib", "tanib", "parib",
)

def _score_synonym(s: str) -> int:
    """
    Score a PubChem synonym for display quality. Lower = better.
    Returns 999 if the synonym should be skipped entirely.
    """
    import re as _re
    s = s.strip()
    # Skip: too long, has brackets/slashes, all-caps, starts with digit
    if len(s) > 40:
        return 999
    if _re.search(r'[\[\](){}/\\]', s):
        return 999
    if s.isupper() and len(s) > 6:
        return 999
    if s[0].isdigit():
        return 999
    if not _re.search(r'[a-z]', s):
        return 999
    # Skip registry IDs like "RefChem:...", "CHEMBL...", "SCHEMBL..."
    if _re.match(r'^(RefChem|SCHEMBL|CHEMBL|DTXSID|NSC|SID)\W', s, _re.I):
        return 999

    score = 50
    # INN suffix → best
    s_lower = s.lower()
    if any(s_lower.endswith(stem) for stem in _INN_STEMS):
        score -= 40
    # Short and clean → good
    if len(s) < 15:
        score -= 10
    # Contains hyphen (research codes like ONO-3080573) → decent
    if '-' in s:
        score += 5
    return score


def _best_synonym_from_cid(cid: int, verbose: bool, current_name: str) -> str:
    """
    Fetch PubChem synonyms for a CID and return the best display name.
    Returns current_name if nothing better is found.
    """
    if not _REQUESTS_AVAILABLE or not cid:
        return current_name
    try:
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"cid/{cid}/synonyms/JSON",
            timeout=8
        )
        if r.status_code != 200:
            return current_name
        syns = (
            r.json()
             .get("InformationList", {})
             .get("Information", [{}])[0]
             .get("Synonym", [])
        )
        # Score all synonyms and pick the best
        candidates = [(s, _score_synonym(s)) for s in syns[:50]]
        candidates = [(s, sc) for s, sc in candidates if sc < 999]
        if not candidates:
            return current_name
        best_name, best_score = min(candidates, key=lambda x: x[1])
        # Only substitute if it's meaningfully better than the current name
        current_score = _score_synonym(current_name)
        if best_score < current_score:
            if verbose:
                print(f"  [web_search]   ✓ Drug preferred name: '{best_name}' "
                      f"(CID {cid}, score {best_score} vs {current_score})")
            return best_name
    except Exception:
        pass
    return current_name


def _resolve_drug_display_name(
    drug: str, drug_smiles: str, cocrystal_code: str,
    result_json: dict, verbose: bool
) -> str:
    """
    Resolve the best searchable display name for a drug.

    Tries in order:
      1. SMILES → PubChem CID → best synonym
         (most reliable — SMILES is structurally unambiguous)
      2. PDB 3-letter code → PubChem CID → best synonym
         (e.g. "ON3" → CID → "ONO-3080573")
      3. Drug name → PubChem CID → best synonym
         (handles IUPAC blobs and registry IDs like RefChem:168278)
      4. Return original name

    Args:
        drug          : drug name from results JSON
        drug_smiles   : SMILES string from results JSON
        cocrystal_code: PDB 3-letter code if known (e.g. "ON3")
        result_json   : full pipeline result dict
        verbose       : print progress

    Returns:
        Best human-readable name for searches and the Gemma prompt
    """
    if not _REQUESTS_AVAILABLE:
        return drug

    cid = None

    # ── 1. SMILES lookup (most reliable) ─────────────────────────────────
    if drug_smiles:
        cid = _pubchem_cid_from_smiles(drug_smiles)
        if cid and verbose:
            print(f"  [web_search] Drug CID from SMILES: {cid}")

    # ── 2. PDB code lookup ────────────────────────────────────────────────
    if not cid and cocrystal_code and len(cocrystal_code) == 3:
        cid = _pubchem_cid_from_name(cocrystal_code)
        if cid and verbose:
            print(f"  [web_search] Drug CID from PDB code '{cocrystal_code}': {cid}")

    # ── 3. Drug name lookup ───────────────────────────────────────────────
    if not cid:
        cid = _pubchem_cid_from_name(drug)
        if cid and verbose:
            print(f"  [web_search] Drug CID from name '{drug}': {cid}")

    if not cid:
        if verbose:
            print(f"  [web_search]   ⚠ Could not resolve CID for '{drug}' — "
                  f"using original name for searches")
        return drug

    return _best_synonym_from_cid(cid, verbose, drug)




# ════════════════════════════════════════════════════════════════════════════
# NOVEL COMPOUND DETECTOR
# ════════════════════════════════════════════════════════════════════════════

import re as _novel_re
_NOVEL_NAME_PATTERNS = [
    r"^(novel|gen|new|compound|cpd|mol|drug|candidate|inhibitor)[-_]",
    r"^(gen|gm|ai|dl|vae|gan)[-_]",
    r"[-_]\d{1,5}$",
    r"^[A-Za-z]{1,4}[-_][A-Z]{2,6}[-_]\d",
]
_NOVEL_NAME_RE = [_novel_re.compile(p, _novel_re.IGNORECASE)
                  for p in _NOVEL_NAME_PATTERNS]

_KNOWN_REGISTRY_PREFIXES = (
    "ono-", "azd-", "ly-", "mk-", "bay-", "gsk-", "pf-",
    "abt-", "bms-", "jnj-", "sgi-", "ros-", "nsc-",
)

def is_novel_compound(drug_name: str, cid: int | None = None) -> bool:
    """Return True if the drug name appears to be user-assigned for a novel compound."""
    if not drug_name:
        return True

    if cid is not None:
        return False

    name_lower = drug_name.lower().strip()

    if any(name_lower.startswith(p) for p in _KNOWN_REGISTRY_PREFIXES):
        return False

    for pattern in _NOVEL_NAME_RE:
        if pattern.search(drug_name):
            return True

    if len(drug_name) <= 5 and not name_lower.replace("-","").replace("_","").isalpha():
        return True

    return False

# ════════════════════════════════════════════════════════════════════════════
# MAIN ENRICHMENT FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def enrich_docking_context(
    drug          : str,
    protein       : str,
    pdb_id        : str | None  = None,
    drug_smiles   : str | None  = None,
    result_json   : dict | None = None,
    pdb_path      : str | None  = None,  # NEW: full PDB file path
    n_papers      : int         = 8,
    verbose       : bool        = True,
    delay_between : float       = 1.2,
) -> LiteratureContext:
    """
    Gather all literature evidence for a drug–protein pair.

    Sources:
      RCSB      → authoritative gene symbol + protein title
      PubChem   → drug pharmacology (name, PDB code, or SMILES)
      UniProt   → protein function annotation
      ChEMBL    → mechanism of action
      PubMed    → peer-reviewed abstracts
      DDG       → web snippets (mechanism, biology, interaction, clinical)

    KEY FIX: Now accepts pdb_path to extract full COMPND description
    instead of using just the classification tag ("DNA").

    Args:
        drug          : drug name from pipeline JSON (may be registry ID)
        protein       : raw PDB protein string
        pdb_id        : 4-char PDB ID — enables RCSB gene symbol lookup
        drug_smiles   : SMILES — primary source for CID/name resolution
        result_json   : full pipeline result dict
        pdb_path      : path to raw PDB file — NEW: extracts full COMPND
        n_papers      : PubMed papers to fetch
        verbose       : print progress
        delay_between : seconds between DDG queries
    """
    drug_smiles = drug_smiles or ""
    ctx = LiteratureContext()

    def log(msg):
        if verbose:
            print(f"  [web_search] {msg}")

    # ── 0. Resolve best drug display name ────────────────────────────────
    cocrystal_code = (result_json.get("cocrystal_code", "") or ""
                      if result_json else "")
    log(f"Resolving drug display name for '{drug}'...")
    ctx.drug_display_name = _resolve_drug_display_name(
        drug          = drug,
        drug_smiles   = drug_smiles,
        cocrystal_code= cocrystal_code,
        result_json   = result_json or {},
        verbose       = verbose,
    )
    if ctx.drug_display_name != drug:
        log(f"  → Searches will use: '{ctx.drug_display_name}'")

    # ── 1. Resolve protein name via RCSB + PDB COMPND ────────────────────
    # KEY FIX: Now passes pdb_path so full COMPND is extracted
    log(f"Resolving protein name" + (f" for PDB {pdb_id}..." if pdb_id else "..."))
    search_name, display_name = resolve_protein_name(
        protein_raw = protein,
        result_json = result_json,
        pdb_id      = pdb_id,
        pdb_path    = pdb_path,  # NEW: enables COMPND extraction
    )
    ctx.gene_symbol           = search_name
    ctx.protein_display_name  = display_name

    if search_name != protein:
        log(f"  ✓ Resolved: '{protein}' → '{search_name}'")
        if display_name and display_name != search_name:
            log(f"    Full name: {display_name}")

        # Sanity check for mismatches
        _protein_lower  = protein.lower()
        _gene_lower     = search_name.lower()
        _display_lower  = display_name.lower()
        _plausible = True
        _mismatch_pairs = [
            (["lysophosphatidic", "lpa", "lpar"], ["egfr", "erbb"]),
            (["kinase", "egfr", "erbb"],           ["lpar", "lysophosphatidic"]),
            (["hemoglobin", "haemoglobin"],         ["egfr", "lpar", "kinase"]),
        ]
        for protein_keywords, gene_keywords in _mismatch_pairs:
            if (any(k in _protein_lower for k in protein_keywords)
                    and any(k in _gene_lower or k in _display_lower
                            for k in gene_keywords)):
                _plausible = False
                break

        if not _plausible:
            log(f"  ⚠ Gene symbol '{search_name}' may not match protein "
                f"'{protein}' — possible RCSB entity mismatch.")
            ctx.warnings.append(
                f"Gene symbol '{search_name}' may not match the intended "
                f"target protein '{protein}'. Verify the correct protein "
                f"entity in the PDB file."
            )
    else:
        log(f"  ⚠ Could not resolve gene symbol — using '{search_name}' as-is")
        ctx.warnings.append(
            f"Protein name '{protein}' could not be resolved to a gene symbol. "
            f"Provide --pdb for better results."
        )

    time.sleep(0.3)

    # ── Detect novel compound
    _drug_cid = result_json.get("_pubchem_cid") if result_json else None
    _is_novel = is_novel_compound(drug, cid=_drug_cid)

    if _is_novel:
        log(f"Novel compound detected ('{drug}' has no expected web presence).")
        ctx.drug_summary = (
            f"NOVEL COMPOUND: '{drug}' is a user-assigned name with no "
            f"public database entry. No pharmacology data is available."
        )
        if drug_smiles:
            _cid = _pubchem_cid_from_smiles(drug_smiles)
            if _cid:
                _desc = _fetch_pubchem_drug("", drug_smiles)
                if _desc:
                    ctx.drug_summary += f"\n\nClosest PubChem match (CID {_cid}):\n{_desc}"
                    log(f"  ✓ Closest PubChem match: CID {_cid}")
    else:
        # ── 2. PubChem — drug pharmacology
        log(f"PubChem: {drug[:50]}...")
        pubchem = _fetch_pubchem_drug(drug, drug_smiles)
        if pubchem:
            ctx.drug_summary = pubchem
            log(f"  ✓ {len(pubchem)} chars")
        else:
            log("  – no result")
        time.sleep(0.3)

        # ── 4. ChEMBL — mechanism of action
        log(f"ChEMBL MoA: {drug[:50]}...")
        chembl_moa = _fetch_chembl_moa(drug, drug_smiles)
        if chembl_moa:
            ctx.drug_summary = (
                f"Mechanism of action (ChEMBL):\n{chembl_moa}"
                + ("\n\n" + ctx.drug_summary if ctx.drug_summary else "")
            ).strip()
            log("  ✓ MoA found")
        else:
            log("  – no mechanism found")
        time.sleep(0.3)

    # ── 3. UniProt — protein function
    log(f"UniProt: {search_name}...")
    uniprot = _fetch_uniprot_protein(search_name)
    if uniprot:
        ctx.protein_summary = uniprot
        log(f"  ✓ {len(uniprot)} chars")
    else:
        log("  – no result")
    time.sleep(0.3)

    # ── 5. PubMed — peer-reviewed abstracts
    interactions = result_json.get("interactions", []) if result_json else []
    if _is_novel:
        log(f"PubMed: protein-only search (novel drug — skipping drug name)...")
        ctx.papers = gather_pubmed_literature(
            drug         = "",
            protein      = protein,
            gene_symbol  = search_name,
            interactions = interactions,
            n_papers     = n_papers,
            verbose      = verbose,
        )
    else:
        log(f"PubMed: searching for {ctx.drug_display_name[:30]} + {search_name}...")
        ctx.papers = gather_pubmed_literature(
            drug         = ctx.drug_display_name,
            protein      = protein,
            gene_symbol  = search_name,
            interactions = interactions,
            n_papers     = n_papers,
            verbose      = verbose,
        )

    if ctx.papers:
        paper_block = ""
        for i, p in enumerate(ctx.papers, 1):
            paper_block += (
                f"\n[Paper {i}] {p.authors} ({p.year}). {p.title}. "
                f"{p.journal}. PMID {p.pmid}.\n"
                f"Abstract: {p.abstract[:600]}...\n"
            )
        ctx.interaction_evidence = paper_block.strip()
        ctx.sources += [p.url for p in ctx.papers]

    time.sleep(0.5)

    # ── 6. DuckDuckGo — web snippets
    if _DDGS_AVAILABLE:
        import re as _re

        if not _is_novel:
            _drug_queries = [f"{ctx.drug_display_name} mechanism of action pharmacology"]
            _stripped = _re.sub(r"^[A-Z]{2,6}[-_]", "", drug)
            if _stripped != drug:
                _drug_queries.append(f"{_stripped} compound pharmacology")

            _drug_snippets = []
            for _q in _drug_queries:
                log(f"DDG: {_q[:60]}...")
                _r = _ddg_search(_q, max_results=4)
                if _r:
                    _drug_snippets += _r
                    ctx.web_sources += _r
                    ctx.sources    += [s.url for s in _r]
                    log(f"  ✓ {len(_r)} results")
                time.sleep(delay_between)

            if _drug_snippets:
                snippets = "\n".join(f"[{s.title}] {s.snippet}" for s in _drug_snippets)
                ctx.drug_summary = (
                    ctx.drug_summary + "\n\nWeb snippets:\n" + snippets
                ).strip()
        else:
            log(f"DDG drug search: skipped (novel compound — no web presence)")

        # 6b. Protein biology
        log(f"DDG: {search_name} protein biology...")
        _r = _ddg_search(f"{search_name} protein function structure disease", max_results=4)
        if _r:
            snippets = "\n".join(f"[{s.title}] {s.snippet}" for s in _r)
            ctx.protein_summary = (ctx.protein_summary + "\n\nWeb snippets:\n" + snippets).strip()
            ctx.web_sources += _r
            ctx.sources     += [s.url for s in _r]
            log(f"  ✓ {len(_r)} results")
        time.sleep(delay_between)

        # 6c. Drug–protein interaction
        if not _is_novel:
            log(f"DDG: {ctx.drug_display_name} {search_name} binding...")
            _r = _ddg_search(
                f"{ctx.drug_display_name} {search_name} binding docking interaction",
                max_results=5
            )
            if _r:
                snippets = "\n".join(f"[{s.title}] {s.snippet}" for s in _r)
                ctx.interaction_evidence = (
                    ctx.interaction_evidence + "\n\nWeb snippets:\n" + snippets
                ).strip()
                ctx.web_sources += _r
                ctx.sources     += [s.url for s in _r]
                log(f"  ✓ {len(_r)} results")
            else:
                log("  – no interaction literature found")
            time.sleep(delay_between)

            # 6d. Clinical context
            log(f"DDG: {ctx.drug_display_name} clinical use...")
            _r = _ddg_search(
                f"{ctx.drug_display_name} clinical use approved indication treatment",
                max_results=3
            )
            if _r:
                ctx.clinical_context = "\n".join(f"[{s.title}] {s.snippet}" for s in _r)
                ctx.web_sources += _r
                ctx.sources     += [s.url for s in _r]
                log(f"  ✓ {len(_r)} results")
            time.sleep(delay_between)
        else:
            log(f"DDG interaction/clinical search: skipped (novel compound)")
            ctx.warnings.append(
                f"'{drug}' is a novel compound with no published binding data. "
                f"The explanation is based on the computed 3D pose and protein "
                f"context only."
            )

    ctx.sources = list(dict.fromkeys(ctx.sources))

    log(
        f"Enrichment complete — "
        f"{len(ctx.papers)} papers, "
        f"{len(ctx.web_sources)} web sources, "
        f"gene: {ctx.gene_symbol or 'unresolved'}"
    )
    return ctx


# ════════════════════════════════════════════════════════════════════════════
# PROMPT FORMATTER
# ════════════════════════════════════════════════════════════════════════════

def format_context_for_prompt(ctx: LiteratureContext) -> str:
    """Render LiteratureContext as a structured block for the Gemma 4 prompt."""
    sections = []

    if ctx.gene_symbol:
        sections.append(
            f"PROTEIN IDENTITY: Gene symbol = {ctx.gene_symbol}"
            + (f" | Full name: {ctx.protein_display_name}"
               if ctx.protein_display_name != ctx.gene_symbol else "")
        )

    if ctx.drug_summary:
        sections.append(
            "DRUG BACKGROUND (PubChem + ChEMBL + web):\n"
            + textwrap.fill(ctx.drug_summary, width=90)
        )

    if ctx.protein_summary:
        sections.append(
            "PROTEIN BACKGROUND (UniProt + web):\n"
            + textwrap.fill(ctx.protein_summary, width=90)
        )

    if ctx.papers:
        paper_lines = []
        for i, p in enumerate(ctx.papers, 1):
            paper_lines.append(
                f"[Paper {i}] {p.authors} ({p.year}). {p.title}. "
                f"{p.journal}. PMID {p.pmid}.\n"
                f"  Abstract: {p.abstract[:500]}..."
            )
        sections.append(
            "PEER-REVIEWED LITERATURE (PubMed):\n"
            + "\n\n".join(paper_lines)
        )

    if ctx.interaction_evidence and not ctx.papers:
        sections.append(
            "INTERACTION EVIDENCE (web):\n" + ctx.interaction_evidence[:1500]
        )

    if ctx.clinical_context:
        sections.append(
            "CLINICAL CONTEXT (web):\n" + ctx.clinical_context
        )

    if ctx.warnings:
        sections.append(
            "SEARCH WARNINGS:\n"
            + "\n".join(f"  ⚠ {w}" for w in ctx.warnings)
        )

    return (
        "\n\n".join(sections)
        or "LITERATURE CONTEXT: No data gathered. Rely on training knowledge."
    )


def format_bibliography(ctx: LiteratureContext) -> str:
    """Return a formatted bibliography of all sources used."""
    lines = ["BIBLIOGRAPHY", "=" * 60]

    if ctx.papers:
        lines.append("\nPeer-reviewed papers (PubMed):")
        for i, p in enumerate(ctx.papers, 1):
            lines.append(
                f"\n[{i}] {p.authors} ({p.year}). {p.title}. "
                f"{p.journal}.\n"
                f"    PMID: {p.pmid} | {p.url}"
            )

    if ctx.web_sources:
        offset = len(ctx.papers)
        seen_urls = set()
        unique_web = []
        for s in ctx.web_sources:
            if s.url not in seen_urls:
                seen_urls.add(s.url)
                unique_web.append(s)

        lines.append("\n\nWeb sources:")
        for i, s in enumerate(unique_web, offset + 1):
            lines.append(f"\n[{i}] {s.title}\n    {s.url}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DockExplain — unified literature enrichment"
    )
    parser.add_argument("--drug",     required=True)
    parser.add_argument("--protein",  default="")
    parser.add_argument("--pdb",      default=None,
        help="PDB ID for RCSB gene symbol lookup (e.g. 1M17)")
    parser.add_argument("--pdb-file", default=None,
        help="Path to PDB file to extract full COMPND description")
    parser.add_argument("--n-papers", type=int, default=8,
        help="Number of PubMed papers to fetch (default: 8)")
    parser.add_argument("--out",      default=None)
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  DockExplain — Literature Enrichment")
    print(f"  Drug    : {args.drug}")
    print(f"  Protein : {args.protein or '(resolved from PDB)'}")
    print(f"  PDB ID  : {args.pdb or 'not provided'}")
    print(f"{'═'*60}\n")

    ctx = enrich_docking_context(
        drug      = args.drug,
        protein   = args.protein,
        pdb_id    = args.pdb,
        pdb_path  = args.pdb_file,
        n_papers  = args.n_papers,
        verbose   = True,
    )

    print(f"\n{'═'*60}")
    print(format_context_for_prompt(ctx))
    print(f"\n{format_bibliography(ctx)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "drug"                : args.drug,
                "gene_symbol"         : ctx.gene_symbol,
                "protein_display_name": ctx.protein_display_name,
                "drug_summary"        : ctx.drug_summary,
                "protein_summary"     : ctx.protein_summary,
                "interaction_evidence": ctx.interaction_evidence,
                "clinical_context"    : ctx.clinical_context,
                "papers"              : [p.__dict__ for p in ctx.papers],
                "web_sources"         : [s.__dict__ for s in ctx.web_sources],
                "warnings"            : ctx.warnings,
                "sources"             : ctx.sources,
                "bibliography"        : format_bibliography(ctx),
            }, f, indent=2)
        print(f"\n  ✓ Saved: {args.out}")