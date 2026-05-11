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


# ════════════════════════════════════════════════════════════
# STEP 1: Download PDB from RCSB
# ════════════════════════════════════════════════════════════

def download_pdb(pdb_id: str, out_dir: str = "../data") -> str:
    """
    Download the biological assembly PDB file for any
    RCSB Protein Data Bank entry.
    Returns the local path to the saved .pdb file.
    """
    os.makedirs(out_dir, exist_ok=True)
    pdb_id  = pdb_id.strip().upper()
    out_path = os.path.join(out_dir, f"{pdb_id}.pdb")

    if os.path.exists(out_path):
        print(f"  ✓ PDB already downloaded: {out_path}")
        return out_path

    # Use the RCSB PDB download URL
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
    """
    Scan HETATM records in a PDB file and return a list of
    unique drug-like ligands (skipping solvents/buffers).
    Each entry: {code, n_atoms}
    Sorted by atom count descending (most atoms = most likely drug).
    """
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

    # Sort by atom count — drug has most atoms
    ligands = [
        {"code": code, "n_atoms": n}
        for code, n in sorted(
            found.items(), key=lambda x: x[1], reverse=True
        )
    ]
    return ligands


# ════════════════════════════════════════════════════════════
# STEP 3: Fetch ligand info from RCSB CCD
# ════════════════════════════════════════════════════════════

def fetch_ligand_info(code: str) -> dict:
    """
    Fetch the full name, SMILES, formula, and description
    for a 3-letter PDB ligand code using the RCSB CCD API.
    Works for any valid PDB ligand code.
    Returns dict with keys: code, name, smiles, formula, description
    """
    code = code.upper()
    info = {
        "code"       : code,
        "name"       : "",
        "smiles"     : "",
        "formula"    : "",
        "description": "",
    }

    # ── Source 1: RCSB Data API ──────────────────────────
    try:
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{code}"
        req = urllib.request.Request(
            url, headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())

        # Name
        chem = data.get("chem_comp", {})
        info["name"]    = chem.get("name", "")
        info["formula"] = chem.get("formula", "")
        info["description"] = chem.get("pdbx_formal_charge", "")

        # SMILES from descriptor list
        descriptors = data.get("pdbx_chem_comp_descriptor", [])
        for d in descriptors:
            if d.get("type", "").upper() == "SMILES_CANONICAL":
                info["smiles"] = d.get("descriptor", "")
                break
        if not info["smiles"]:
            for d in descriptors:
                if "SMILES" in d.get("type", "").upper():
                    info["smiles"] = d.get("descriptor", "")
                    break

        if info["smiles"] and Chem.MolFromSmiles(info["smiles"]):
            return info

    except Exception:
        pass

    # ── Source 2: RCSB ideal SDF ─────────────────────────
    try:
        url = (f"https://files.rcsb.org/ligands/download/"
               f"{code}_ideal.sdf")
        with urllib.request.urlopen(url, timeout=10) as r:
            sdf = r.read().decode("utf-8")
        mol = Chem.MolFromMolBlock(sdf, removeHs=True)
        if mol:
            info["smiles"] = Chem.MolToSmiles(mol)
            return info
    except Exception:
        pass

    # ── Source 3: PubChem by CCD code ────────────────────
    try:
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
               f"/name/{code}/property/"
               f"IsomericSMILES,IUPACName,MolecularFormula/JSON")
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        props = (data.get("PropertyTable", {})
                     .get("Properties", [{}])[0])
        smiles = props.get("IsomericSMILES", "")
        if smiles and Chem.MolFromSmiles(smiles):
            info["smiles"] = smiles
            if not info["name"]:
                info["name"] = props.get("IUPACName", "")
            if not info["formula"]:
                info["formula"] = props.get("MolecularFormula", "")
        return info
    except Exception:
        pass

    return info


# ════════════════════════════════════════════════════════════
# STEP 4: Resolve any drug input to a SMILES string
# Accepts: 3-letter PDB code, drug name, or SMILES directly
# ════════════════════════════════════════════════════════════

def resolve_drug_input(user_input: str) -> dict:
    """
    Resolve whatever the user typed to a canonical SMILES.
    Suppresses RDKit stderr for non-SMILES inputs.

    Tries in order:
      1. Direct SMILES — if RDKit parses it cleanly
      2. 3-letter PDB code — fetch from RCSB CCD
      3. Drug name — search PubChem by name
      4. Drug name — search PubChem by name similarity

    Returns dict: {smiles, name, code, source}
    """
    import sys
    import io

    user_input = user_input.strip()
    result = {
        "smiles": "",
        "name"  : user_input,
        "code"  : "",
        "source": "",
    }

    # ── Try 1: Direct SMILES ─────────────────────────────
    # Suppress RDKit stderr for non-SMILES strings —
    # RDKit always prints parse errors even when we're
    # intentionally testing whether something IS a SMILES
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

    mol = Chem.MolFromSmiles(user_input)

    RDLogger.EnableLog("rdApp.error")
    RDLogger.EnableLog("rdApp.warning")

    if mol is not None:
        result["smiles"] = Chem.MolToSmiles(mol)
        result["source"] = "user-provided SMILES"
        print(f"  ✓ Recognised as SMILES string")
        return result

    # ── Try 2: 3-letter PDB code ─────────────────────────
    # PDB codes are exactly 3 alphanumeric characters
    cleaned = user_input.strip().upper()
    if len(cleaned) == 3 and cleaned.isalnum():
        print(f"  Trying as PDB ligand code: {cleaned}...")
        info = fetch_ligand_info(cleaned)
        if info.get("smiles"):
            result["smiles"] = info["smiles"]
            result["name"]   = info.get("name") or cleaned
            result["code"]   = cleaned
            result["source"] = "RCSB CCD (3-letter code)"
            print(f"  ✓ Resolved via RCSB CCD: {result['name']}")
            return result

    # ── Try 3: Drug name via PubChem name search ─────────
    print(f"  Searching PubChem for '{user_input}'...")
    try:
        # PubChem name search endpoint
        # urllib.parse.quote handles spaces and special chars
        encoded = urllib.parse.quote(user_input, safe="")
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
            f"/name/{encoded}/property/"
            f"IsomericSMILES,IUPACName,MolecularFormula/JSON"
        )
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json",
                     "User-Agent": "DockExplain/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())

        props = (data.get("PropertyTable", {})
                     .get("Properties", [{}])[0])
        smiles = props.get("IsomericSMILES", "")

        if smiles:
            # Validate
            RDLogger.DisableLog("rdApp.error")
            mol = Chem.MolFromSmiles(smiles)
            RDLogger.EnableLog("rdApp.error")

            if mol is not None:
                result["smiles"]  = Chem.MolToSmiles(mol)
                result["name"]    = (props.get("IUPACName")
                                     or user_input)
                result["formula"] = props.get("MolecularFormula", "")
                result["source"]  = "PubChem (by name)"
                print(f"  ✓ Found on PubChem: "
                      f"{props.get('IUPACName', user_input)[:60]}")
                return result

    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  ⚠ PubChem: no compound found named "
                  f"'{user_input}'")
        else:
            print(f"  ⚠ PubChem HTTP error: {e.code}")
    except Exception as e:
        print(f"  ⚠ PubChem search failed: {e}")

    # ── Try 4: PubChem autocomplete / fuzzy name ─────────
    # Sometimes brand names or abbreviations need this
    print(f"  Trying PubChem autocomplete...")
    try:
        encoded = urllib.parse.quote(user_input, safe="")
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
            f"/name/{encoded}/cids/JSON"
            f"?name_type=word"
        )
        with urllib.request.urlopen(url, timeout=10) as r:
            cid_data = json.loads(r.read())

        cids = cid_data.get("IdentifierList", {}).get("CID", [])
        if cids:
            # Fetch properties for first CID
            cid = cids[0]
            prop_url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
                f"/cid/{cid}/property/"
                f"IsomericSMILES,IUPACName,MolecularFormula/JSON"
            )
            with urllib.request.urlopen(prop_url, timeout=10) as r:
                prop_data = json.loads(r.read())

            props = (prop_data.get("PropertyTable", {})
                              .get("Properties", [{}])[0])
            smiles = props.get("IsomericSMILES", "")

            if smiles:
                RDLogger.DisableLog("rdApp.error")
                mol = Chem.MolFromSmiles(smiles)
                RDLogger.EnableLog("rdApp.error")

                if mol is not None:
                    result["smiles"]  = Chem.MolToSmiles(mol)
                    result["name"]    = (props.get("IUPACName")
                                         or user_input)
                    result["formula"] = props.get(
                        "MolecularFormula", ""
                    )
                    result["source"]  = "PubChem (word search)"
                    print(f"  ✓ Found via PubChem word search: "
                          f"CID {cid}")
                    return result

    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  ⚠ PubChem word search error: {e.code}")
    except Exception as e:
        print(f"  ⚠ PubChem word search failed: {e}")

    # All sources exhausted
    return result


# ════════════════════════════════════════════════════════════
# STEP 5: Fetch protein metadata from RCSB
# ════════════════════════════════════════════════════════════

def fetch_protein_metadata(pdb_id: str) -> dict:
    """
    Fetch human-readable metadata for any PDB entry.
    Uses multiple RCSB endpoints to get all fields reliably.
    """
    pdb_id = pdb_id.upper()
    meta = {
        "pdb_id"    : pdb_id,
        "name"      : "",
        "organism"  : "",
        "method"    : "",
        "resolution": "",
        "deposited" : "",
    }

    # ── Entry endpoint — method, resolution, date ─────────
    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        req = urllib.request.Request(
            url, headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())

        # Protein name — from struct.title (most reliable field)
        struct = data.get("struct", {})
        meta["name"] = struct.get("title", "")

        # Method
        exptl = data.get("exptl", [{}])
        meta["method"] = exptl[0].get("method", "") if exptl else ""

        # Resolution
        reflns = data.get("reflns", [{}])
        res = reflns[0].get("d_resolution_high", "") if reflns else ""
        meta["resolution"] = str(res) if res else ""

        # Deposition date
        acc = data.get("rcsb_accession_info", {})
        meta["deposited"] = acc.get("deposit_date", "")[:10]

    except Exception as e:
        print(f"  ⚠ Entry metadata fetch failed: {e}")

    # ── Entity endpoint — organism ─────────────────────────
    # The organism lives in the polymer entity, not the entry
    try:
        url = (f"https://data.rcsb.org/rest/v1/core/"
               f"polymer_entity/{pdb_id}/1")
        req = urllib.request.Request(
            url, headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            entity_data = json.loads(r.read())

        # Protein description (more specific than struct.title)
        entity_info = entity_data.get("rcsb_polymer_entity", {})
        entity_desc = entity_info.get("pdbx_description", "")
        if entity_desc and not meta["name"]:
            meta["name"] = entity_desc

        # Organism from source taxonomy
        orgs = []
        for src in entity_data.get(
            "rcsb_entity_source_organism", []
        ):
            sci = src.get("ncbi_scientific_name", "")
            if sci:
                orgs.append(sci)
        meta["organism"] = ", ".join(set(orgs))

    except Exception as e:
        print(f"  ⚠ Entity metadata fetch failed: {e}")

    # ── Fallback: use RCSB search API ─────────────────────
    if not meta["name"] or not meta["organism"]:
        try:
            url = (f"https://search.rcsb.org/rcsbsearch/v2/query"
                   f"?json=%7B%22query%22%3A%7B%22type%22%3A%22"
                   f"terminal%22%2C%22service%22%3A%22full_text%22"
                   f"%2C%22parameters%22%3A%7B%22value%22%3A%22"
                   f"{pdb_id}%22%7D%7D%7D")
            # Simpler: just call the summary endpoint
            url2 = (f"https://data.rcsb.org/rest/v1/core/"
                    f"entry/{pdb_id}")
            req = urllib.request.Request(
                url2, headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.loads(r.read())
            # Try rcsb_entry_info for organism
            info = d.get("rcsb_entry_info", {})
            if not meta["organism"]:
                tax = info.get("taxonomy_ids", [])
                if tax:
                    meta["organism"] = f"TaxID {tax[0]}"
        except Exception:
            pass

    return meta


# ════════════════════════════════════════════════════════════
# MAIN INTERACTIVE FLOW
# ════════════════════════════════════════════════════════════

def run_input_handler(pdb_data_dir: str = "../data") -> dict:
    """
    Interactive input collection for DockExplain.

    Guides the user through:
      1. Enter RCSB Protein ID
      2. Downloads PDB + shows protein metadata
      3. Detects co-crystal ligand(s) and shows their info
      4. Asks user to use crystal drug or provide their own
      5. Resolves the drug to a canonical SMILES
      6. Asks for a drug name if not already known
      7. Returns everything needed for run_pipeline()

    Returns:
        {
          pdb_path   : local path to downloaded PDB
          pdb_id     : uppercase PDB ID
          drug_name  : human-readable drug name
          drug_smiles: canonical SMILES
          drug_code  : 3-letter PDB code if known (or "")
          mode_hint  : "A" if using crystal drug, "B"/"C" otherwise
        }
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
            print("  ⚠ Please enter a PDB ID.")
            continue
        if not (3 <= len(pdb_id) <= 8 and pdb_id.isalnum()):
            print("  ⚠ PDB IDs are 4 alphanumeric characters "
                  "(e.g. 1M17). Please try again.")
            continue
        break


    # ── 2. Download PDB + show metadata ───────────────────
    print(f"\n  Fetching {pdb_id} from RCSB...")
    try:
        pdb_path = download_pdb(pdb_id, out_dir=pdb_data_dir)
    except RuntimeError as e:
        print(f"\n  ✗ {e}")
        sys.exit(1)

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
    ligands = detect_ligands(pdb_path)

    crystal_drug = None   # will hold the ligand dict if found

    if not ligands:
        # Apo structure
        print(f"\n  ○ No drug-like ligand found in {pdb_id}.")
        print(f"    This appears to be an apo (unbound) structure.")
        print(f"    You will need to provide the drug to dock.")
        get_user_drug = True

    else:
        # Show all detected ligands
        print(f"\n  ● Co-crystal ligand(s) detected:")
        ligand_infos = []
        for lig in ligands:
            print(f"    Fetching info for {lig['code']}...")
            info = fetch_ligand_info(lig["code"])
            info["n_atoms"] = lig["n_atoms"]
            ligand_infos.append(info)

        # Display each ligand
        print()
        for i, info in enumerate(ligand_infos, 1):
            name    = info.get("name", "Unknown") or "Unknown"
            formula = info.get("formula", "")
            smiles  = info.get("smiles", "")
            n_atoms = info.get("n_atoms", 0)
            code    = info.get("code", "")

            print(f"  ┌─ Ligand {i}: {code} ─────────────────────────────┐")
            print(f"  │  Name    : {name[:50]:<50}│")
            print(f"  │  Formula : {formula:<50}│")
            print(f"  │  PDB code: {code:<50}│")
            print(f"  │  Atoms   : {str(n_atoms):<50}│")
            if smiles:
                # Show SMILES split across lines if long
                s1 = smiles[:50]
                s2 = smiles[50:100] if len(smiles) > 50 else ""
                print(f"  │  SMILES  : {s1:<50}│")
                if s2:
                    print(f"  │           {s2:<50}│")
            else:
                print(f"  │  SMILES  : (could not retrieve)                  │")
            print(f"  └──────────────────────────────────────────────────┘")

        # Primary ligand = most atoms
        crystal_drug = ligand_infos[0]

        # Ask user
        if len(ligand_infos) == 1:
            prompt_str = (
                f"\n  Would you like to analyse the co-crystal ligand "
                f"({crystal_drug['code']} — "
                f"{(crystal_drug['name'] or crystal_drug['code'])[:40]})?\n"
                f"  [Y] Yes, use {crystal_drug['code']}  "
                f"[N] No, I'll provide a different drug\n"
                f"  Choice: "
            )
        else:
            prompt_str = (
                f"\n  Would you like to analyse the primary co-crystal "
                f"ligand ({crystal_drug['code']})?\n"
                f"  [Y] Yes, use {crystal_drug['code']}  "
                f"[N] No, I'll provide a different drug\n"
                f"  Choice: "
            )

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
        # User chose the crystal drug
        drug_smiles = crystal_drug.get("smiles", "")
        drug_name   = crystal_drug.get("name") or crystal_drug["code"]
        drug_code   = crystal_drug["code"]
        mode_hint   = "A"

        if not drug_smiles:
            print(f"\n  ⚠ Could not retrieve SMILES for {drug_code}.")
            print(f"    Switching to manual drug entry...")
            get_user_drug = True

    if get_user_drug:
        mode_hint = "B" if (ligands) else "C"
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
                print("  ⚠ Input cannot be empty.")
                continue

            print(f"\n  Resolving '{user_drug_input}'...")
            resolved = resolve_drug_input(user_drug_input)

            if not resolved["smiles"]:
                print(
                    f"\n  ✗ Could not resolve '{user_drug_input}'.\n"
                    f"    PubChem was searched but returned no results.\n"
                    f"    This can happen with:\n"
                    f"      • Misspelled names (try 'Ibuprofen' not 'ibuprofen')\n"
                    f"      • Brand names (try generic: 'Advil' → 'Ibuprofen')\n"
                    f"      • Abbreviations (try full name)\n"
                    f"\n    Alternatives:\n"
                    f"      • Paste the SMILES directly from "
                    f"https://pubchem.ncbi.nlm.nih.gov\n"
                    f"      • Use the 3-letter PDB code (e.g. IBP for ibuprofen)"
                )
                continue

            drug_smiles = resolved["smiles"]
            drug_code   = resolved.get("code", "")

            # Validate SMILES with RDKit
            mol = Chem.MolFromSmiles(drug_smiles)
            if mol is None:
                print(f"  ✗ Resolved SMILES failed RDKit validation.")
                continue

            n_atoms = mol.GetNumAtoms()
            print(f"\n  ✓ Drug resolved successfully:")
            print(f"    Name   : {resolved['name']}")
            print(f"    SMILES : {drug_smiles[:70]}"
                  f"{'...' if len(drug_smiles)>70 else ''}")
            print(f"    Atoms  : {n_atoms} heavy atoms")
            print(f"    Source : {resolved['source']}")
            break

        # Ask for human-readable name if not already clear
        default_name = resolved.get("name", "")
        if (not default_name or
                default_name == user_drug_input or
                len(default_name) > 60):
            # If name is empty, same as input, or a long IUPAC string,
            # ask the user for a clean display name
            print(f"\n  What would you like to call this drug?")
            print(f"  (Used in reports, filenames, and Gemma 4 prompts)")
            name_input = input(
                f"  Drug name [{default_name or 'Drug'}]: "
            ).strip()
            drug_name = name_input if name_input else (
                default_name or "Drug"
            )
        else:
            drug_name = default_name
            confirm = input(
                f"\n  Use '{drug_name}' as the drug name? [Y/n]: "
            ).strip().upper()
            if confirm in ("N", "NO"):
                drug_name = input("  Enter drug name: ").strip() or drug_name


    # ── 5. Final confirmation ──────────────────────────────
    print(f"\n  ┌─ DockExplain Job Summary ────────────────────────┐")
    print(f"  │  PDB ID    : {pdb_id:<40}│")
    print(f"  │  Protein   : {(meta['name'] or 'unknown')[:40]:<40}│")
    print(f"  │  Drug      : {drug_name[:40]:<40}│")
    code_str = drug_code or "N/A"
    print(f"  │  PDB code  : {code_str:<40}│")
    smiles_display = drug_smiles[:40] + ("..." if len(drug_smiles)>40 else "")
    print(f"  │  SMILES    : {smiles_display:<40}│")
    mode_labels = {
        "A": "Mode A — analyse crystal ligand (no docking)",
        "B": "Mode B — dock new drug into crystal site",
        "C": "Mode C — blind docking (apo structure)",
    }
    print(f"  │  Mode      : {mode_labels.get(mode_hint,'?')[:40]:<40}│")
    print(f"  └──────────────────────────────────────────────────┘")

    print()
    while True:
        confirm = input(
            "  Proceed with this configuration? [Y/n]: "
        ).strip().upper()
        if confirm in ("Y", "YES", ""):
            break
        elif confirm in ("N", "NO"):
            print("\n  Restarting input...\n")
            return run_input_handler(pdb_data_dir)
        else:
            print("  Please enter Y or N.")

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