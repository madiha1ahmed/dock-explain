# input_handler.py
# Interactive input handler for DockExplain
# Accepts a RCSB Protein ID, downloads the PDB,
# detects bound ligands, and resolves the drug SMILES.
#
# Called by run_dockexplain.py before the pipeline starts.

import json
import os
import sys
import urllib.request
import urllib.parse
import time
from rdkit import Chem
from rdkit import RDLogger


# ════════════════════════════════════════════════════════════
# STEP 1: Download PDB from RCSB
# ════════════════════════════════════════════════════════════

def download_pdb(pdb_id: str, out_dir: str = "../data") -> str:
    os.makedirs(out_dir, exist_ok=True)
    pdb_id   = pdb_id.strip().upper()
    out_path = os.path.join(out_dir, f"{pdb_id}.pdb")
    if os.path.exists(out_path):
        print(f"  ✓ PDB already downloaded: {out_path}")
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id}.pdb from RCSB...")
    try:
        urllib.request.urlretrieve(url, out_path)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  ✓ Downloaded: {out_path} ({size_kb:.0f} KB)")
        return out_path
    except Exception as e:
        raise RuntimeError(
            f"Could not download PDB {pdb_id} from RCSB.\n"
            f"  URL tried: {url}\n"
            f"  Error: {e}\n"
            f"  Check the PDB ID is valid: https://www.rcsb.org/structure/{pdb_id}"
        )


# ════════════════════════════════════════════════════════════
# STEP 2: Detect co-crystal ligand(s)
# ════════════════════════════════════════════════════════════

SKIP_CODES = {
    "HOH","WAT","SO4","GOL","EDO","PEG","MPD","TRS","MES",
    "ACT","NO3","CL","NA","MG","ZN","CA","K","FE","MN","CO",
    "BME","DTT","DMS","IOD","BR","F","PO4","EPE","HED","FMT",
    "IMD","ACE","NH2","SEP","TPO","PTR","CSO","HIC","MLY",
    "ANP","ADP","ATP","GDP","GTP","FMT","ACY","ACE","EOH",
}

def detect_ligands(pdb_path: str) -> list[dict]:
    found = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            code = line[17:20].strip()
            if code and code not in SKIP_CODES:
                found[code] = found.get(code, 0) + 1
    if not found:
        return []
    return [
        {"code": code, "n_atoms": n}
        for code, n in sorted(found.items(), key=lambda x: x[1], reverse=True)
    ]


# ════════════════════════════════════════════════════════════
# STEP 3: Fetch ligand info from RCSB CCD
# ════════════════════════════════════════════════════════════

def fetch_ligand_info(code: str) -> dict:
    code = code.upper()
    info = {"code": code, "name": "", "smiles": "", "formula": ""}

    try:
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        chem = data.get("chem_comp", {})
        info["name"]    = chem.get("name", "")
        info["formula"] = chem.get("formula", "")
        for d in data.get("pdbx_chem_comp_descriptor", []):
            if d.get("type", "").upper() == "SMILES_CANONICAL":
                info["smiles"] = d.get("descriptor", ""); break
        if not info["smiles"]:
            for d in data.get("pdbx_chem_comp_descriptor", []):
                if "SMILES" in d.get("type", "").upper():
                    info["smiles"] = d.get("descriptor", ""); break
        if info["smiles"] and Chem.MolFromSmiles(info["smiles"]):
            return info
    except Exception:
        pass

    try:
        url = f"https://files.rcsb.org/ligands/download/{code}_ideal.sdf"
        with urllib.request.urlopen(url, timeout=10) as r:
            sdf = r.read().decode("utf-8")
        mol = Chem.MolFromMolBlock(sdf, removeHs=True)
        if mol:
            info["smiles"] = Chem.MolToSmiles(mol)
            return info
    except Exception:
        pass

    try:
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
               f"/name/{code}/property/IsomericSMILES,IUPACName,MolecularFormula/JSON")
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
        smiles = props.get("IsomericSMILES", "")
        if smiles and Chem.MolFromSmiles(smiles):
            info["smiles"]  = smiles
            info["name"]    = info["name"] or props.get("IUPACName", "")
            info["formula"] = info["formula"] or props.get("MolecularFormula", "")
        return info
    except Exception:
        pass

    return info


# ════════════════════════════════════════════════════════════
# DRUG NAME RESOLVER
#
# Fetches the best human-readable name from PubChem synonyms.
# Uses scoring to skip registry IDs (RefChem:..., SCHEMBL...,
# CHEMBL...) and prefer INN names or clean short synonyms.
# ════════════════════════════════════════════════════════════

_INN_STEMS = (
    "inib", "mab", "zumab", "ximab", "umab", "tinib", "ciclib",
    "rafenib", "buvir", "previr", "navir", "lukast", "sartan",
    "pril", "olol", "dipine", "statin", "mycin", "cycline",
    "floxacin", "azole", "conazole", "thiazide", "pam",
    "oxacin", "micin", "bactam", "penem", "cillin", "semib",
    "tanib", "parib", "metformin",
)

# Registry ID prefixes that should NEVER be used as display names
_REGISTRY_PREFIXES = (
    "refchem:", "schembl", "chembl", "dtxsid", "mfcd",
    "akos", "zinc", "bdbm", "sid:", "cid:",
)

def _score_synonym(name: str) -> int:
    """
    Score a synonym for display quality. Lower score = better name.
    Returns 999 if the synonym must be skipped entirely.
    """
    import re
    s = name.strip()
    if not s:
        return 999

    s_lower = s.lower()

    # Hard skip: registry IDs
    if any(s_lower.startswith(p) for p in _REGISTRY_PREFIXES):
        return 999
    # Hard skip: too long
    if len(s) > 50:
        return 999
    # Hard skip: contains brackets or slashes (IUPAC / SMILES fragments)
    if re.search(r'[\[\](){}/\\]', s):
        return 999
    # Hard skip: all uppercase and long (database accession codes)
    if s.isupper() and len(s) > 8:
        return 999
    # Hard skip: no lowercase letters at all
    if not re.search(r'[a-z]', s):
        return 999
    # Hard skip: starts with a digit (CAS numbers etc.)
    if s[0].isdigit():
        return 999

    score = 50
    # INN name → best possible
    if any(s_lower.endswith(stem) for stem in _INN_STEMS):
        score -= 40
    # Short → good
    if len(s) <= 15:
        score -= 10
    elif len(s) <= 25:
        score -= 5
    # Hyphenated research code (ONO-3080573) → decent
    if '-' in s and re.search(r'\d', s):
        score += 5
    # Contains digits only in a suffix (Erlotinib → fine, but "NSC123456" → bad)
    if re.fullmatch(r'[A-Z]{2,6}\d{4,}', s):
        score += 20   # internal code, not great

    return score


def get_preferred_name(
    cid      : int | None = None,
    fallback : str        = "",
    rcsb_name: str        = "",
) -> str:
    """
    Return the best human-readable drug name for a given PubChem CID.

    Priority:
      1. INN (approved generic name ending in known stem)
      2. Best-scored PubChem synonym (short, clean, no registry prefix)
      3. RCSB CCD name if clean (passed as rcsb_name)
      4. fallback (original user input or code)

    The RCSB CCD name is tried as a tiebreaker because for PDB ligands
    it is often more informative than PubChem synonyms (e.g. "METFORMIN"
    vs "RefChem:1051085").
    """
    import re

    best_name  = None
    best_score = 999

    if cid is not None:
        try:
            url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                f"cid/{cid}/synonyms/JSON"
            )
            with urllib.request.urlopen(url, timeout=8) as r:
                data = json.loads(r.read())

            synonyms = (
                data.get("InformationList", {})
                    .get("Information", [{}])[0]
                    .get("Synonym", [])
            )

            for syn in synonyms[:80]:
                sc = _score_synonym(syn)
                if sc < best_score:
                    best_score = sc
                    best_name  = syn

        except Exception:
            pass

    # Consider RCSB CCD name as a candidate if it's clean
    if rcsb_name:
        rcsb_score = _score_synonym(rcsb_name)
        # Accept RCSB name if it scores at least as well as the best synonym
        # (RCSB names are authoritative for PDB ligands)
        if rcsb_score <= best_score:
            best_name  = rcsb_name
            best_score = rcsb_score

    if best_name and best_score < 999:
        return best_name

    return fallback or rcsb_name or ""



def _pubchem_get(url: str, retries: int = 3, timeout: int = 12) -> dict | None:
    """
    GET a PubChem REST URL with retry logic and a neutral User-Agent.

    PubChem's CDN blocks custom User-Agent strings like "DockExplain/1.0"
    intermittently. Using Python's default urllib agent avoids this.
    Retries up to `retries` times on transient network errors with
    a short delay between attempts.

    Returns:
        Parsed JSON dict, or None on failure.
    """
    for attempt in range(retries):
        try:
            # No custom User-Agent — let urllib use its default
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None          # compound not found — don't retry
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None
    return None


def _pubchem_post(url: str, data: dict, retries: int = 3, timeout: int = 12) -> dict | None:
    """POST to a PubChem REST URL with retry logic."""
    encoded = urllib.parse.urlencode(data).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=encoded,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None
    return None


def _get_pubchem_cid(smiles: str) -> int | None:
    """Resolve a SMILES string to a PubChem CID via POST."""
    data = _pubchem_post(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/JSON",
        {"smiles": smiles},
    )
    if data:
        try:
            return data["PC_Compounds"][0]["id"]["id"]["cid"]
        except (KeyError, IndexError):
            pass
    return None


def _cid_to_smiles(cid: int) -> str:
    """Fetch the canonical SMILES for a known PubChem CID."""
    data = _pubchem_get(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
        f"/cid/{cid}/property/IsomericSMILES/JSON"
    )
    if data:
        try:
            return data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
        except (KeyError, IndexError):
            pass
    return ""


def _name_to_cid(name: str) -> int | None:
    """
    Resolve a drug name to a PubChem CID using three strategies:
      1. Exact name match  (/compound/name/<name>/cids/JSON)
      2. Synonym match     (/compound/name/<name>/cids/JSON?name_type=complete)
      3. Word search       (/compound/name/<name>/cids/JSON?name_type=word)

    Returns the first CID found, or None.
    """
    encoded = urllib.parse.quote(name, safe="")
    base    = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

    for suffix in [
        f"/name/{encoded}/cids/JSON",
        f"/name/{encoded}/cids/JSON?name_type=complete",
        f"/name/{encoded}/cids/JSON?name_type=word",
    ]:
        data = _pubchem_get(base + suffix)
        if data:
            cids = data.get("IdentifierList", {}).get("CID", [])
            if cids:
                return cids[0]
        time.sleep(0.3)   # brief pause between strategies

    return None


# ════════════════════════════════════════════════════════════
# STEP 4: Resolve any drug input → SMILES + auto name
# ════════════════════════════════════════════════════════════

def resolve_drug_input(user_input: str) -> dict:
    """
    Resolve whatever the user typed to a canonical SMILES and
    automatically fetch the best display name — no prompting needed.

    Resolution order:
      1. Direct SMILES  → validate with RDKit, look up name via CID
      2. 3-letter PDB code → RCSB CCD SMILES, look up name via CID
      3. Drug name → PubChem CID (3 sub-strategies: exact, synonym, word)
                  → SMILES from CID → preferred name from synonyms

    All PubChem requests use _pubchem_get() which retries on transient
    failures and uses the default urllib User-Agent (avoids CDN blocks).

    Returns:
        {smiles, name, code, source, cid}
    """
    user_input = user_input.strip()
    result = {"smiles": "", "name": "", "code": "", "source": "", "cid": None}

    # Silence RDKit parse-error noise for non-SMILES inputs
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    mol = Chem.MolFromSmiles(user_input)
    RDLogger.EnableLog("rdApp.error")
    RDLogger.EnableLog("rdApp.warning")

    # ── Try 1: Direct SMILES ─────────────────────────────
    if mol is not None:
        canonical = Chem.MolToSmiles(mol)
        print("  ✓ Recognised as SMILES string")
        print("  Fetching compound identity from PubChem...")
        cid  = _get_pubchem_cid(canonical)
        name = get_preferred_name(cid, fallback="")
        result.update({"smiles": canonical, "name": name,
                        "source": "user-provided SMILES", "cid": cid})
        print(f"  ✓ Identified as: {name}" if name
              else "  ⚠ Could not identify compound name from PubChem")
        return result

    # ── Try 2: 3-letter PDB code ─────────────────────────
    cleaned = user_input.strip().upper()
    if len(cleaned) == 3 and cleaned.isalnum():
        print(f"  Trying as PDB ligand code: {cleaned}...")
        info = fetch_ligand_info(cleaned)
        if info.get("smiles"):
            cid       = _get_pubchem_cid(info["smiles"])
            rcsb_name = info.get("name", "") or ""
            # Pass rcsb_name so RCSB CCD name competes with PubChem synonyms
            name = get_preferred_name(cid,
                                      fallback=rcsb_name or cleaned,
                                      rcsb_name=rcsb_name)
            result.update({"smiles": info["smiles"], "name": name,
                            "code": cleaned, "source": "RCSB CCD (3-letter code)",
                            "cid": cid})
            print(f"  ✓ Resolved: {name}")
            return result

    # ── Try 3: Drug name → CID → SMILES → preferred name ─
    # Uses _name_to_cid() which tries exact, synonym, and word
    # search in sequence — no custom User-Agent, with retry logic.
    print(f"  Searching PubChem for '{user_input}'...")
    cid = _name_to_cid(user_input)

    if cid:
        smiles = _cid_to_smiles(cid)
        if smiles:
            RDLogger.DisableLog("rdApp.error")
            mol = Chem.MolFromSmiles(smiles)
            RDLogger.EnableLog("rdApp.error")
            if mol is not None:
                canonical = Chem.MolToSmiles(mol)
                name = get_preferred_name(cid, fallback=user_input)
                result.update({"smiles": canonical, "name": name,
                                "source": "PubChem (by name)", "cid": cid})
                print(f"  ✓ Found on PubChem: {name}")
                return result

    return result


# ════════════════════════════════════════════════════════════
# STEP 5: Fetch protein metadata from RCSB
# ════════════════════════════════════════════════════════════

def fetch_protein_metadata(pdb_id: str) -> dict:
    pdb_id = pdb_id.upper()
    meta = {
        "pdb_id"    : pdb_id,
        "name"      : "",
        "organism"  : "",
        "method"    : "",
        "resolution": "",
        "deposited" : "",
    }
    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        meta["name"]      = data.get("struct", {}).get("title", "")
        exptl             = data.get("exptl", [{}])
        meta["method"]    = exptl[0].get("method", "") if exptl else ""
        reflns            = data.get("reflns", [{}])
        res               = reflns[0].get("d_resolution_high", "") if reflns else ""
        meta["resolution"]= str(res) if res else ""
        acc               = data.get("rcsb_accession_info", {})
        meta["deposited"] = acc.get("deposit_date", "")[:10]
    except Exception as e:
        print(f"  ⚠ Entry metadata fetch failed: {e}")

    try:
        url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            entity_data = json.loads(r.read())
        entity_info = entity_data.get("rcsb_polymer_entity", {})
        if not meta["name"]:
            meta["name"] = entity_info.get("pdbx_description", "")
        orgs = []
        for src in entity_data.get("rcsb_entity_source_organism", []):
            sci = src.get("ncbi_scientific_name", "")
            if sci:
                orgs.append(sci)
        meta["organism"] = ", ".join(set(orgs))
    except Exception as e:
        print(f"  ⚠ Entity metadata fetch failed: {e}")

    return meta


# ════════════════════════════════════════════════════════════
# MAIN INTERACTIVE FLOW
# ════════════════════════════════════════════════════════════

def run_input_handler(pdb_data_dir: str = "../data") -> dict:
    """
    Interactive input collection for DockExplain.

    Drug name resolution is fully automatic — PubChem synonyms are
    queried to find the best INN/generic name without asking the user.
    """
    print("\n" + "█" * 62)
    print("  DockExplain — Input Setup")
    print("█" * 62)

    # ── 1. Get PDB ID ─────────────────────────────────────
    print()
    while True:
        pdb_id = input(
            "  Enter RCSB Protein ID (e.g. 1M17, 6LU7, 4COX): "
        ).strip().upper()
        if not pdb_id:
            print("  ⚠ Please enter a PDB ID."); continue
        if not (3 <= len(pdb_id) <= 8 and pdb_id.isalnum()):
            print("  ⚠ PDB IDs are 4 alphanumeric characters. Try again.")
            continue
        break

    # ── 2. Download PDB + show metadata ───────────────────
    print(f"\n  Fetching {pdb_id} from RCSB...")
    try:
        pdb_path = download_pdb(pdb_id, out_dir=pdb_data_dir)
    except RuntimeError as e:
        print(f"\n  ✗ {e}"); sys.exit(1)

    print(f"\n  Fetching protein metadata...")
    meta = fetch_protein_metadata(pdb_id)

    print(f"\n  ┌─ Structure Summary ──────────────────────────────┐")
    print(f"  │  PDB ID     : {meta['pdb_id']:<38}│")
    print(f"  │  Protein    : {(meta['name'] or 'unknown')[:38]:<38}│")
    print(f"  │  Organism   : {(meta['organism'] or 'unknown')[:38]:<38}│")
    print(f"  │  Method     : {(meta['method'] or 'unknown')[:38]:<38}│")
    res_str = f"{meta['resolution']} Å" if meta['resolution'] else "N/A"
    print(f"  │  Resolution : {res_str:<38}│")
    print(f"  │  Deposited  : {(meta['deposited'] or 'unknown'):<38}│")
    print(f"  └──────────────────────────────────────────────────┘")

    # ── 3. Detect co-crystal ligand(s) ────────────────────
    print(f"\n  Scanning for co-crystal ligand(s)...")
    ligands     = detect_ligands(pdb_path)
    crystal_drug = None
    get_user_drug = False

    if not ligands:
        print(f"\n  ○ No drug-like ligand found — apo structure.")
        print(f"    You will need to provide the drug to dock.")
        get_user_drug = True
    else:
        print(f"\n  ● Co-crystal ligand(s) detected:")
        ligand_infos = []
        for lig in ligands:
            print(f"    Fetching info for {lig['code']}...")
            info = fetch_ligand_info(lig["code"])
            # Auto-resolve clean name via PubChem synonyms
            if info.get("smiles"):
                cid = _get_pubchem_cid(info["smiles"])
                info["preferred_name"] = get_preferred_name(
                    cid, fallback=info.get("name") or lig["code"]
                )
                info["cid"] = cid
            else:
                info["preferred_name"] = info.get("name") or lig["code"]
                info["cid"] = None
            info["n_atoms"] = lig["n_atoms"]
            ligand_infos.append(info)

        # Display each ligand
        print()
        for i, info in enumerate(ligand_infos, 1):
            display_name = info.get("preferred_name") or info.get("name", "Unknown")
            formula = info.get("formula", "")
            smiles  = info.get("smiles", "")
            n_atoms = info.get("n_atoms", 0)
            code    = info.get("code", "")

            print(f"  ┌─ Ligand {i}: {code} ─────────────────────────────┐")
            print(f"  │  Name    : {display_name[:50]:<50}│")
            print(f"  │  Formula : {formula:<50}│")
            print(f"  │  PDB code: {code:<50}│")
            print(f"  │  Atoms   : {str(n_atoms):<50}│")
            if smiles:
                s1 = smiles[:50]
                s2 = smiles[50:100] if len(smiles) > 50 else ""
                print(f"  │  SMILES  : {s1:<50}│")
                if s2:
                    print(f"  │           {s2:<50}│")
            else:
                print(f"  │  SMILES  : (could not retrieve)                  │")
            print(f"  └──────────────────────────────────────────────────┘")

        # INPUT_HANDLER FIX
# Add this section to input_handler.py, replacing lines 621-649
# This enables users to select which binding site to analyze when multiple ligands are present

# ════════════════════════════════════════════════════════════════════════════
# LIGAND SELECTION WITH MULTIPLE BINDING SITES
# ════════════════════════════════════════════════════════════════════════════

        # ── MODIFIED: Handle single AND multiple ligands ──────────────────
        if len(ligand_infos) == 1:
            # Single ligand - use it directly
            crystal_drug = ligand_infos[0]
            primary_name = crystal_drug.get("preferred_name") or crystal_drug["code"]
            
            prompt_str = (
                f"\n  Would you like to analyse the co-crystal ligand "
                f"({crystal_drug['code']} — {primary_name[:40]})?\n"
                f"  [Y] Yes, use {crystal_drug['code']}  "
                f"[N] No, I'll provide a different drug\n"
                f"  Choice: "
            )
        
        else:
            # ← NEW: Multiple ligands detected - allow user to select
            print(f"\n  ● Multiple co-crystal ligands detected ({len(ligand_infos)} binding sites):")
            print(f"    Please select which binding site you want to analyze:\n")
            
            for i, info in enumerate(ligand_infos, 1):
                display_name = info.get("preferred_name") or info.get("name", "Unknown")
                formula = info.get("formula", "")
                code = info.get("code", "")
                n_atoms = info.get("n_atoms", 0)
                
                print(f"  ┌─ Site {i}: {code} ─────────────────────────┐")
                print(f"  │  Name    : {display_name[:45]:<45}│")
                print(f"  │  Formula : {formula:<45}│")
                print(f"  │  Atoms   : {str(n_atoms):<45}│")
                print(f"  └──────────────────────────────────────────────┘")
            
            # Ligand selection loop
            while True:
                try:
                    choice_str = input(
                        f"\n  Select binding site [1-{len(ligand_infos)}]: "
                    ).strip()
                    
                    if not choice_str:
                        print("  ⚠ Please enter a number.")
                        continue
                    
                    choice_idx = int(choice_str) - 1
                    
                    if 0 <= choice_idx < len(ligand_infos):
                        crystal_drug = ligand_infos[choice_idx]
                        selected_name = (crystal_drug.get("preferred_name") or 
                                       crystal_drug.get("name") or 
                                       crystal_drug["code"])
                        print(f"\n  ✓ Selected binding site: {crystal_drug['code']} "
                              f"({selected_name})")
                        get_user_drug = False
                        break
                    else:
                        print(f"  ⚠ Please enter a number between 1 and {len(ligand_infos)}")
                        continue
                
                except ValueError:
                    print(f"  ⚠ Invalid input. Please enter a number.")
                    continue
            
            primary_name = (crystal_drug.get("preferred_name") or 
                          crystal_drug.get("name") or 
                          crystal_drug["code"])
            
            prompt_str = (
                f"\n  Would you like to analyse this selected co-crystal ligand "
                f"({crystal_drug['code']} — {primary_name[:40]})?\n"
                f"  [Y] Yes, use {crystal_drug['code']}  "
                f"[N] No, I'll provide a different drug\n"
                f"  Choice: "
            )

        # ── Rest of logic continues unchanged ─────────────────────────────
        while True:
            choice = input(prompt_str).strip().upper()
            if choice in ("Y", "YES", ""):
                get_user_drug = False
                break
            elif choice in ("N", "NO"):
                get_user_drug = True
                break
            else:
                print("  Please enter Y or N.")
    # ── 4. Resolve drug ────────────────────────────────────
    if not get_user_drug and crystal_drug:
        drug_smiles = crystal_drug.get("smiles", "")
        drug_name   = crystal_drug.get("preferred_name") or crystal_drug["code"]
        drug_code   = crystal_drug["code"]
        mode_hint   = "A"
        if not drug_smiles:
            print(f"\n  ⚠ Could not retrieve SMILES for {drug_code}.")
            print(f"    Switching to manual drug entry...")
            get_user_drug = True

    if get_user_drug:
        mode_hint = "B" if ligands else "C"
        print(
            f"\n  Enter the drug you want to dock."
            f"\n  You can provide any of the following:"
            f"\n    • A SMILES string   (e.g. CC(=O)Oc1ccccc1C(=O)O)"
            f"\n    • A PDB 3-letter code  (e.g. AQ4, STI, LIG)"
            f"\n    • A drug name          (e.g. Erlotinib, Ibuprofen)"
        )

        while True:
            user_drug_input = input(
                "\n  Enter drug SMILES / code / name: "
            ).strip()
            if not user_drug_input:
                print("  ⚠ Input cannot be empty."); continue

            # Strip internal whitespace — SMILES strings never contain
            # spaces or newlines. When a SMILES is pasted from a paper
            # or generative model output it may be line-wrapped; stripping
            # gives RDKit the full string rather than a truncated fragment.
            import re as _re
            _clean = _re.sub(r'\s+', '', user_drug_input)
            if _clean != user_drug_input:
                print("  ℹ Whitespace stripped from input (common with pasted SMILES).")
                user_drug_input = _clean

            print(f"\n  Resolving '{user_drug_input[:60]}"
                  f"{'...' if len(user_drug_input) > 60 else ''}'...")
            resolved = resolve_drug_input(user_drug_input)

            if not resolved["smiles"]:
                print(
                    f"\n  ✗ Could not resolve '{user_drug_input}'.\n"
                    f"    Alternatives:\n"
                    f"      • Paste the SMILES directly from "
                    f"https://pubchem.ncbi.nlm.nih.gov\n"
                    f"      • Try the generic name (e.g. 'Ibuprofen' not 'Advil')\n"
                    f"      • Use the 3-letter PDB code (e.g. IBP for ibuprofen)"
                )
                continue

            drug_smiles = resolved["smiles"]
            drug_code   = resolved.get("code", "")
            drug_name   = resolved["name"]

            # Validate with RDKit
            mol = Chem.MolFromSmiles(drug_smiles)
            if mol is None:
                print(f"  ✗ Resolved SMILES failed RDKit validation.")
                continue

            n_atoms = mol.GetNumAtoms()

            # Novel/generative model molecules won't be in PubChem —
            # name will be empty. Prompt the user for a display name
            # used in filenames, reports, and the Gemma 4 explanation.
            if not drug_name:
                print(f"\n  ℹ This molecule was not found in PubChem.")
                print(f"    This is expected for novel or generative model compounds.")
                print(f"    Please provide a short name for reports and filenames.")
                print(f"    Examples: 'Novel-LPAR1-001', 'GenCompound-7', 'MyDrug'")
                while True:
                    entered = input("  Drug name: ").strip()
                    if entered:
                        drug_name = entered
                        break
                    print("  ⚠ Name cannot be empty.")

            print(f"\n  ✓ Drug resolved successfully:")
            print(f"    Name   : {drug_name}")
            print(f"    SMILES : {drug_smiles[:70]}{'...' if len(drug_smiles)>70 else ''}")
            print(f"    Atoms  : {n_atoms} heavy atoms")
            print(f"    Source : {resolved['source']}")
            break

    # ── 5. Final summary + confirmation ───────────────────
    mode_labels = {
        "A": "Mode A — analyse crystal ligand (no docking)",
        "B": "Mode B — dock new drug into crystal site",
        "C": "Mode C — blind docking (apo structure)",
    }
    print(f"\n  ┌─ DockExplain Job Summary ────────────────────────┐")
    print(f"  │  PDB ID    : {pdb_id:<40}│")
    print(f"  │  Protein   : {(meta['name'] or 'unknown')[:40]:<40}│")
    print(f"  │  Drug      : {drug_name[:40]:<40}│")
    print(f"  │  PDB code  : {(drug_code or 'N/A'):<40}│")
    smiles_d = drug_smiles[:40] + ("..." if len(drug_smiles) > 40 else "")
    print(f"  │  SMILES    : {smiles_d:<40}│")
    print(f"  │  Mode      : {mode_labels.get(mode_hint,'?')[:40]:<40}│")
    print(f"  └──────────────────────────────────────────────────┘")

    print()
    # Flush any stray newlines left in stdin from multi-line paste
    # before reading the Y/n confirmation
    try:
        import sys as _sys, termios
        termios.tcflush(_sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass  # not available on Windows

    while True:
        confirm = input("  Proceed with this configuration? [Y/n]: ").strip().upper()
        if confirm in ("Y", "YES", ""):
            break
        elif confirm in ("N", "NO"):
            print("\n  Restarting input...\n")
            return run_input_handler(pdb_data_dir)
        else:
            print(f"  Please enter Y or N.")

    print(f"\n  ✓ Configuration confirmed. Starting pipeline...\n")

    return {
        "pdb_path"    : pdb_path,
        "pdb_id"      : pdb_id,
        "drug_name"   : drug_name,
        "drug_smiles" : drug_smiles,
        "drug_code"   : drug_code,
        "mode_hint"   : mode_hint,
        "protein_meta": meta,
    }