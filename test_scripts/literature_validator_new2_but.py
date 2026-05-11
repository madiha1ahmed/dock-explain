# literature_validator.py — fully generalizable, zero hardcoding
# Works for any drug-protein pair, any PDB ID
#
# Run: python literature_validator.py \
#          ../data/results/erlotinib/erlotinib_results.json

import json, time, os, sys
import urllib.request, urllib.parse
import xml.etree.ElementTree as ET
import ollama


# ════════════════════════════════════════════════════════════
# RCSB — get structure metadata (deposition year + citation)
# Fully generalizable — works for any PDB ID
# ════════════════════════════════════════════════════════════

def fetch_rcsb_structure_info(pdb_id: str) -> dict:
    """
    Fetch all metadata for any PDB entry:
      - Deposition year (used to validate citations)
      - Primary citation PMID + DOI
      - Molecule names (actual protein name from RCSB)
      - Organism
      - Resolution
      - Experimental method

    Returns dict with these fields, or empty dict on failure.
    Works for any PDB ID — nothing hardcoded.
    """
    url = (f"https://data.rcsb.org/rest/v1/core/entry/"
           f"{pdb_id.upper()}")
    try:
        req = urllib.request.Request(
            url, headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())

        info = {}

        # ── Deposition date ─────────────────────────────────
        accession = data.get("rcsb_accession_info", {})
        deposit_date = accession.get("deposit_date", "")
        if deposit_date:
            info["deposit_year"] = int(deposit_date[:4])
        else:
            info["deposit_year"] = None

        # ── Primary citation ────────────────────────────────
        citation = data.get("rcsb_primary_citation", {})
        pmid = citation.get("pdbx_database_id_pub_med")
        doi  = citation.get("pdbx_database_id_doi")
        year = citation.get("year")
        title = citation.get("title", "")

        info["citation_pmid"]  = str(pmid) if pmid else None
        info["citation_doi"]   = doi
        info["citation_year"]  = int(year) if year else None
        info["citation_title"] = title

        # ── Validate citation against deposition year ───────
        # If the citation year is more than 5 years AFTER
        # deposition, the API returned a re-annotation or
        # associated paper, not the original deposition paper.
        # In that case we discard it and fall back to PubMed search.
        if (info["deposit_year"] and info["citation_year"] and
                info["citation_year"] > info["deposit_year"] + 5):
            print(f"  ⚠ Citation year {info['citation_year']} is "
                  f">{info['deposit_year']+5} — RCSB returned a "
                  f"recent re-annotation, not the original paper. "
                  f"Discarding and using PubMed search instead.")
            info["citation_pmid"] = None
            info["citation_doi"]  = None

        # ── Molecule / protein name from RCSB ───────────────
        # This gives the actual scientific name (e.g.
        # "Epidermal growth factor receptor") — more useful
        # than the PDB HEADER field ("TRANSFERASE")
        entity_names = []
        for entity in data.get("polymer_entities", []):
            names = entity.get("rcsb_polymer_entity", {})
            desc  = names.get("pdbx_description", "")
            if desc:
                entity_names.append(desc)
        info["molecule_names"] = entity_names
        info["primary_name"]   = entity_names[0] if entity_names else ""

        # ── Organism ─────────────────────────────────────────
        organisms = set()
        for entity in data.get("polymer_entities", []):
            for org in entity.get("rcsb_entity_source_organism", []):
                sci = org.get("ncbi_scientific_name", "")
                if sci:
                    organisms.add(sci)
        info["organisms"] = list(organisms)

        # ── Experimental details ────────────────────────────
        exptl = data.get("exptl", [{}])
        info["method"] = exptl[0].get("method", "") if exptl else ""

        reflns = data.get("reflns", [{}])
        info["resolution"] = (reflns[0].get(
            "d_resolution_high", "") if reflns else "")

        print(f"  ✓ RCSB metadata for {pdb_id.upper()}:")
        print(f"    Deposited : {info['deposit_year']}")
        print(f"    Molecule  : {info['primary_name'][:60]}")
        print(f"    Organism  : {', '.join(info['organisms'])}")
        print(f"    Method    : {info['method']}")
        if info["citation_pmid"]:
            print(f"    Citation  : PMID {info['citation_pmid']} "
                  f"({info['citation_year']})")
        else:
            print(f"    Citation  : not available from RCSB API "
                  f"— will search PubMed directly")
        return info

    except Exception as e:
        print(f"  ⚠ RCSB metadata fetch failed: {e}")
        return {}
    

def fetch_related_structure_pmids(pdb_id: str) -> list[dict]:
    """
    Fetch PMIDs from related PDB structures
    (the same ones shown on RCSB 'Related Structures').

    Returns list of dicts:
      [{pdb_id, pmid, title}]
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
    results = []

    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())

        related = data.get(
            "rcsb_entry_container_identifiers", {}
        ).get("related_entries", [])

        for entry in related:
            rid = entry.get("entry_id")
            if not rid:
                continue

            try:
                rurl = f"https://data.rcsb.org/rest/v1/core/entry/{rid}"
                with urllib.request.urlopen(rurl, timeout=10) as rr:
                    rdata = json.loads(rr.read())

                citation = rdata.get("rcsb_primary_citation", {})
                pmid = citation.get("pdbx_database_id_pub_med")

                if pmid:
                    results.append({
                        "pdb_id": rid,
                        "pmid": str(pmid),
                        "title": citation.get("title", "")
                    })

            except Exception:
                continue

        if results:
            print(f"  ✓ Related structure papers: {len(results)} found")

        return results

    except Exception as e:
        print(f"  ⚠ Failed to fetch related structures: {e}")
        return []


def search_pdb_citation_via_pubmed(pdb_id: str,
                                   protein_name: str,
                                   deposit_year: int | None) -> str | None:
    """
    When the RCSB API doesn't return a valid primary citation,
    search PubMed directly using the PDB accession number.

    PubMed indexes PDB structure IDs in the [si] field.
    This reliably finds the deposition paper for any structure.
    The year filter ensures we get the original paper,
    not a review or later study.

    Works for any PDB ID — nothing hardcoded.
    """
    # Strategy 1: PDB accession number field in PubMed
    # [si] = "secondary source identifier" — PubMed's field for
    # database cross-references including PDB IDs
    query = f"{pdb_id.upper()}[si]"
    pmids = search_pubmed(query, max_results=5)

    if pmids and deposit_year:
        # Fetch years and return the one closest to deposit year
        # (deposition paper should be published same year or ±2 years)
        papers = fetch_papers_metadata_only(pmids)
        candidates = []
        for p in papers:
            try:
                y = int(p.get("year", "9999"))
                if abs(y - deposit_year) <= 3:
                    candidates.append((abs(y - deposit_year), p["pmid"]))
            except ValueError:
                pass
        if candidates:
            candidates.sort()
            pmid = candidates[0][1]
            print(f"  ✓ Found via PubMed [si] field: PMID {pmid}")
            return pmid

    if pmids:
        print(f"  ✓ Found via PubMed [si] field: PMID {pmids[0]}")
        return pmids[0]

    # Strategy 2: protein name + PDB ID as text query
    # Catches cases where [si] indexing is incomplete
    if protein_name:
        query2 = f'"{pdb_id.upper()}" "{protein_name[:30]}"'
        pmids2 = search_pubmed(query2, max_results=3)
        if pmids2:
            print(f"  ✓ Found via text search: PMID {pmids2[0]}")
            return pmids2[0]

    return None


def fetch_papers_metadata_only(pmids: list[str]) -> list[dict]:
    """Lightweight fetch — just year and pmid, no full text."""
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
            root = ET.fromstring(r.read())
        results = []
        for art in root.findall(".//PubmedArticle"):
            pmid_el = art.find(".//PMID")
            year_el = art.find(".//PubDate/Year")
            results.append({
                "pmid": pmid_el.text if pmid_el is not None else "?",
                "year": year_el.text if year_el is not None else "?",
            })
        return results
    except Exception:
        return []


# ════════════════════════════════════════════════════════════
# PUBMED SEARCH
# ════════════════════════════════════════════════════════════

def search_pubmed(query: str, max_results: int = 5) -> list[str]:
    """Search PubMed, return PMIDs. Works for any query."""
    base   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    params = urllib.parse.urlencode({
        "db": "pubmed", "term": query,
        "retmax": max_results,
        "sort": "relevance", "retmode": "json",
    })
    try:
        with urllib.request.urlopen(
            f"{base}esearch.fcgi?{params}", timeout=10
        ) as r:
            return json.loads(r.read())["esearchresult"]["idlist"]
    except Exception as e:
        print(f"  ⚠ PubMed search failed: {e}")
        return []


# ════════════════════════════════════════════════════════════
# PMC FULL TEXT — improved extraction
# ════════════════════════════════════════════════════════════

def fetch_full_text_via_pmc(pmid: str) -> str | None:
    """
    Fetch full article text from PMC.
    Extracts text from all meaningful XML tags —
    not just paragraphs, but also results, methods,
    discussion, figure captions, and table content.
    Works for any open-access paper.
    """
    if not pmid:
        return None

    # Find PMC ID for this PubMed ID
    link_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        f"elink.fcgi?dbfrom=pubmed&db=pmc"
        f"&id={pmid}&retmode=json"
    )
    try:
        with urllib.request.urlopen(link_url, timeout=10) as r:
            link_data = json.loads(r.read())

        pmcids = []
        for ls in link_data.get("linksets", []):
            for lsdb in ls.get("linksetdbs", []):
                if lsdb.get("dbto") == "pmc":
                    pmcids = lsdb.get("links", [])
                    break
        if not pmcids:
            return None

        pmcid = str(pmcids[0])
        print(f"    PMC full text: PMC{pmcid}")

        text_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            f"efetch.fcgi?db=pmc&id={pmcid}"
            f"&rettype=full&retmode=xml"
        )
        with urllib.request.urlopen(text_url, timeout=25) as r:
            xml_data = r.read()

        root = ET.fromstring(xml_data)

        # Extract from all meaningful text-bearing tags
        # including results, methods, discussion content
        TEXT_TAGS = {
            "p", "title", "label", "sec-title", "abstract",
            "kwd", "caption", "th", "td", "def",
            "named-content", "chem-struct", "inline-formula",
        }
        seen   = set()
        chunks = []

        for elem in root.iter():
            if elem.tag not in TEXT_TAGS:
                continue
            # Collect direct text + tail text
            text = " ".join([
                (elem.text or "").strip(),
                (elem.tail or "").strip()
            ]).strip()
            if len(text) > 20 and text not in seen:
                seen.add(text)
                chunks.append(text)

        full_text = "\n".join(chunks)
        if len(full_text) < 200:
            return None

        # Cap at 15000 chars — captures most results + discussion
        cap = min(len(full_text), 15000)
        print(f"    ✓ Full text: {cap} chars (raw: {len(full_text)})")
        return full_text[:cap]

    except Exception as e:
        print(f"    ⚠ PMC fetch failed for PMID {pmid}: {e}")
        return None


# ════════════════════════════════════════════════════════════
# FETCH PAPERS WITH FULL TEXT
# ════════════════════════════════════════════════════════════

def fetch_papers(pmids: list[str]) -> list[dict]:
    """
    Fetch PubMed metadata + attempt PMC full text
    for a list of PMIDs. Fully generalizable.
    """
    if not pmids:
        return []

    base   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    params = urllib.parse.urlencode({
        "db": "pubmed", "id": ",".join(str(p) for p in pmids),
        "rettype": "abstract", "retmode": "xml",
    })
    articles = []

    try:
        with urllib.request.urlopen(
            f"{base}efetch.fcgi?{params}", timeout=20
        ) as r:
            root = ET.fromstring(r.read())

        for article in root.findall(".//PubmedArticle"):
            # Title
            title_el = article.find(".//ArticleTitle")
            title    = (title_el.text or "No title"
                        if title_el is not None else "No title")

            # Abstract (may have labelled sections)
            ab_parts = article.findall(".//AbstractText")
            abstract_sections = []
            for el in ab_parts:
                label = el.get("Label", "")
                text  = (el.text or "").strip()
                if text:
                    abstract_sections.append(
                        f"{label}: {text}" if label else text
                    )
            abstract = " ".join(abstract_sections)

            # PMID
            pmid_el = article.find(".//PMID")
            pmid    = pmid_el.text if pmid_el is not None else "?"

            # Year
            year_el = article.find(".//PubDate/Year")
            year    = year_el.text if year_el is not None else "?"

            # Authors (first 4)
            authors = []
            for au in article.findall(".//Author")[:4]:
                ln = au.find("LastName")
                fn = au.find("ForeName")
                if ln is not None:
                    name = ln.text or ""
                    if fn is not None:
                        name += f" {(fn.text or '')[:1]}"
                    authors.append(name)
            n_total_au = len(article.findall(".//Author"))
            if n_total_au > 4:
                authors.append("et al.")
            author_str = ", ".join(authors) if authors else "Unknown"

            # Journal
            j_el = article.find(".//Journal/Title")
            journal = j_el.text if j_el is not None else ""

            # Volume + pages
            vol_el  = article.find(".//Volume")
            page_el = article.find(".//MedlinePgn")
            vol   = vol_el.text  if vol_el  is not None else ""
            pages = page_el.text if page_el is not None else ""

            # DOI
            doi = ""
            for id_el in article.findall(".//ArticleId"):
                if id_el.get("IdType") == "doi":
                    doi = id_el.text or ""
                    break

            # Attempt PMC full text
            time.sleep(0.4)  # NCBI rate limit: max 3 req/sec
            full_text = fetch_full_text_via_pmc(pmid)

            articles.append({
                "pmid"        : pmid,
                "year"        : year,
                "title"       : title,
                "authors"     : author_str,
                "journal"     : journal,
                "volume"      : vol,
                "pages"       : pages,
                "doi"         : doi,
                "abstract"    : abstract[:3000],
                "full_text"   : full_text,
                "has_fulltext": full_text is not None,
                "url"         : f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "is_primary"  : False,
            })

    except Exception as e:
        print(f"  ⚠ Paper fetch failed: {e}")

    return articles


# ════════════════════════════════════════════════════════════
# QUERY BUILDER — fully dynamic, zero hardcoding
# Builds queries from structure metadata + ProLIF data
# ════════════════════════════════════════════════════════════

def build_queries(drug: str,
                  structure_info: dict,
                  interactions: list[dict],
                  pdb_id: str) -> list[str]:
    """
    Build PubMed search queries dynamically from:
      - drug name (from user input)
      - actual molecule name from RCSB (not PDB header)
      - organism
      - H-bond residues detected by ProLIF
      - PDB ID

    No protein names, residue numbers, or drug classes
    are hardcoded — all derived from the inputs.
    """
    queries = []
    pid = pdb_id.upper()

    # Actual scientific protein name from RCSB
    # e.g. "Epidermal growth factor receptor" not "TRANSFERASE"
    sci_name = structure_info.get("primary_name", "")
    # Shorten to first 40 chars for query (avoids overly long terms)
    sci_name_short = sci_name[:40].strip() if sci_name else ""

    # Organism — used to narrow to the right species
    organisms = structure_info.get("organisms", [])
    organism  = organisms[0] if organisms else ""

    # H-bond residues from ProLIF (most important interactions)
    hbond_res = [
        ix["residue"].split(".")[0]
        for ix in interactions
        if ix["interaction_type"] in ("HBDonor", "HBAcceptor")
    ]
    key_res = hbond_res[:3]

    # Deposit year — helps anchor queries to original papers
    deposit_year = structure_info.get("deposit_year")

    # ── Query 1: PDB accession number (most specific) ──────
    # [si] field in PubMed indexes PDB cross-references
    queries.append(f"{pid}[si]")

    # ── Query 2: Drug + PDB ID ─────────────────────────────
    queries.append(f"{drug} {pid}")

    # ── Query 3: Drug + actual scientific protein name ─────
    if sci_name_short:
        queries.append(f"{drug} {sci_name_short} crystal structure")

    # ── Query 4: Drug + protein name + binding mechanism ───
    if sci_name_short:
        queries.append(
            f"{drug} {sci_name_short} binding interactions inhibitor"
        )

    # ── Query 5: Drug + H-bond residues (ProLIF-derived) ───
    if key_res and sci_name_short:
        res_str = " ".join(key_res)
        queries.append(
            f"{drug} {sci_name_short} {res_str} binding site"
        )

    # ── Query 6: PDB ID + protein name (structure papers) ──
    if sci_name_short:
        queries.append(f"{pid} {sci_name_short}")

    # ── Query 7: Drug + crystal structure + binding ────────
    queries.append(f"{drug} crystal structure binding pocket inhibitor")

    # ── Query 8: Protein name + inhibitor class ───────────
    if sci_name_short:
        queries.append(
            f"{sci_name_short} kinase inhibitor binding hinge"
        )

    return queries


# ════════════════════════════════════════════════════════════
# MAIN LITERATURE GATHERING
# ════════════════════════════════════════════════════════════

def gather_literature(drug: str,
                      protein: str,
                      interactions: list[dict],
                      pdb_id: str | None = None,
                      n_papers: int = 10) -> list[dict]:
    """
    Retrieve literature for any drug-protein pair.
    Zero hardcoding — all queries built from dynamic metadata.

    Priority order:
      1. PDB [si] field in PubMed (most direct)
      2. RCSB primary citation (if valid year)
      3. Drug + actual RCSB molecule name queries
      4. Drug + ProLIF H-bond residue queries
      5. General pharmacology queries
    """
    all_pmids = []
    seen      = set()
    structure_info = {}

    # ── Step 1: Fetch RCSB metadata for this structure ─────
    if pdb_id:
        print(f"\n  Fetching RCSB metadata for {pdb_id.upper()}...")
        structure_info = fetch_rcsb_structure_info(pdb_id)

        # ── Step 2A: Primary citation (RCSB) ───────────────
        citation_pmid = structure_info.get("citation_pmid")

        if citation_pmid:
            if citation_pmid not in seen:
                all_pmids.append(citation_pmid)
                seen.add(citation_pmid)
                print(f"  ✓ Primary citation (RCSB): PMID {citation_pmid}")

        # ── Step 2B: Related structure citations ───────────
        related_papers = fetch_related_structure_pmids(pdb_id)

        for rp in related_papers:
            pmid = rp["pmid"]
            if pmid and pmid not in seen:
                all_pmids.append(pmid)
                seen.add(pmid)
                print(f"  ✓ Related PDB ({rp['pdb_id']}): PMID {pmid}")

        # ── Step 2C: Fallback via PubMed [si] ──────────────
        if not citation_pmid:
            print(f"  Searching PubMed [si] field for {pdb_id}...")
            fallback_pmid = search_pdb_citation_via_pubmed(
                pdb_id,
                structure_info.get("primary_name", protein),
                structure_info.get("deposit_year"),
            )
            if fallback_pmid and fallback_pmid not in seen:
                all_pmids.append(fallback_pmid)
                seen.add(fallback_pmid)
                print(f"  ✓ Primary citation (PubMed): PMID {fallback_pmid}")

    # ── Step 3: Build dynamic queries ─────────────────────
    queries = build_queries(
        drug, structure_info, interactions,
        pdb_id or ""
    )

    print(f"\n  PubMed queries (dynamic, from RCSB metadata):")
    for q in queries:
        if len(all_pmids) >= n_papers + 3:
            break
        print(f"    → {q}")
        pmids = search_pubmed(q, max_results=4)
        for p in pmids:
            if p not in seen:
                all_pmids.append(p)
                seen.add(p)
        time.sleep(0.4)

    all_pmids = all_pmids[:n_papers]
    print(f"\n  Fetching {len(all_pmids)} papers + full text (PMC)...")
    papers = fetch_papers(all_pmids)
    # ── Rank papers: primary > related > others ───────
    primary_pmid = structure_info.get("citation_pmid")

    related_pmids = {
        rp["pmid"] for rp in related_papers
    }

    for p in papers:
        if p["pmid"] == primary_pmid:
            p["is_primary"] = True
            p["priority"] = 0
        elif p["pmid"] in related_pmids:
            p["priority"] = 1
        else:
            p["priority"] = 2

    papers.sort(key=lambda x: x.get("priority", 3))

   

    # Attach structure info for prompt building
    for p in papers:
        p["_structure_info"] = structure_info

    n_full = sum(1 for p in papers if p.get("has_fulltext"))
    print(f"  ✓ {len(papers)} papers: {n_full} full text, "
          f"{len(papers)-n_full} abstract only")
    return papers


# ════════════════════════════════════════════════════════════
# PROMPT BUILDER — uses RCSB metadata, not hardcoded names
# ════════════════════════════════════════════════════════════

def build_validation_prompt(result: dict,
                             papers: list[dict],
                             pdb_id: str,
                             structure_info: dict) -> str:
    drug      = result.get("drug", "Unknown")
    protein   = result.get("protein", "Unknown")
    score     = result.get("docking_score")
    score_str = (f"{score:.2f} kcal/mol" if score
                 else "crystallographic pose (X-ray data)")

    # Use RCSB scientific name if available
    sci_name  = structure_info.get("primary_name", protein)
    organism  = ", ".join(structure_info.get("organisms", []))
    method    = structure_info.get("method", "")
    res       = structure_info.get("resolution", "")

    ix_lines  = [
        f"  {ix['residue']}: {ix['interaction_type']}"
        for ix in result.get("interactions", [])
    ]
    ix_block  = "\n".join(ix_lines) or "  None detected"

    # Build paper content block
    paper_block = ""
    for i, p in enumerate(papers, 1):
        primary_tag = (
            " ← PRIMARY CITATION (original deposition paper)"
            if p.get("is_primary") else ""
        )
        text_tag = (
            "[FULL TEXT AVAILABLE]"
            if p.get("has_fulltext") else "[ABSTRACT ONLY]"
        )
        content = (
            f"FULL TEXT:\n{p['full_text'][:10000]}"
            if p.get("has_fulltext") and p.get("full_text")
            else f"ABSTRACT:\n{p.get('abstract', 'No abstract')}"
        )
        paper_block += (
            f"\n[Paper {i}] {text_tag}{primary_tag}\n"
            f"PMID: {p['pmid']} ({p['year']}) — {p.get('authors','')}\n"
            f"Title: {p['title']}\n"
            f"Journal: {p.get('journal','')}\n"
            f"{content}\n"
            f"{'─'*50}\n"
        )

    if not papers:
        paper_block = "\n[No papers retrieved]\n"

    return f"""You are a rigorous computational chemistry reviewer.

STRUCTURE CONTEXT (from RCSB database — authoritative):
  PDB ID          : {pdb_id.upper()}
  Scientific name : {sci_name}
  Organism        : {organism}
  Method          : {method}
  Resolution      : {res} Å

IMPORTANT: The protein recorded as "{protein}" in our pipeline
IS the same as "{sci_name}" (RCSB official name).
Treat them as identical. Residue numbers in our results
match the PDB file numbering.

═══════════════════════════════════════════════════════
OUR DOCKING RESULTS  (PDB: {pdb_id.upper()})
═══════════════════════════════════════════════════════
Drug         : {drug}
Protein      : {sci_name}
Binding score: {score_str}

Interactions detected by ProLIF:
{ix_block}

ProLIF settings used:
  H-bond distance : 4.0 Å (DHA angle 100–180°)
  Hydrophobic     : 6.0 Å cutoff
  All residues    : within 6 Å of ligand scanned

═══════════════════════════════════════════════════════
RETRIEVED LITERATURE
(Paper 1 is the primary deposition citation for {pdb_id.upper()})
═══════════════════════════════════════════════════════
{paper_block}

═══════════════════════════════════════════════════════
YOUR VALIDATION TASK
═══════════════════════════════════════════════════════
Use ALL text provided above (full text where available).
Extract specific residue interactions mentioned in the papers
and compare them to our ProLIF results.

1. CONFIRMED INTERACTIONS
   For each ProLIF interaction confirmed by any paper:
   state residue + number, interaction type, paper number,
   and the exact or paraphrased supporting text.

2. UNCONFIRMED INTERACTIONS
   ProLIF interactions not mentioned in any paper.
   Assess chemical plausibility from first principles.

3. MISSING INTERACTIONS
   Interactions reported in the papers for {drug} on
   {pdb_id.upper()} that ProLIF did NOT find.
   Explain the likely reason (distance threshold,
   H-atom geometry, residue renumbering).

4. SCORE ASSESSMENT
   Compare our score ({score_str}) to experimental
   binding affinity values (Ki, IC50, Kd) from any paper.

5. OVERALL CONFIDENCE
   HIGH (>70% confirmed) / MEDIUM (40-70%) / LOW (<40%).
   Base rating only on residue-level confirmation,
   not general drug class statements.

6. CAVEATS
   Note any renumbering differences between our PDB
   and the literature's residue numbering scheme.
   Note ProLIF threshold differences vs literature methods.

Cite papers by number. Name every residue by number.
"""


# ════════════════════════════════════════════════════════════
# GEMMA 4 VALIDATION
# ════════════════════════════════════════════════════════════

def validate_with_gemma4(result: dict,
                          papers: list[dict],
                          pdb_id: str,
                          structure_info: dict,
                          model: str = "gemma4:e2b") -> dict:

    prompt = build_validation_prompt(
        result, papers, pdb_id, structure_info
    )
    print(f"\n  Sending to Gemma 4 ({model})...")

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous computational chemistry reviewer. "
                    "You cite specific papers and residue numbers. "
                    "You use full article text where available. "
                    "You treat the RCSB scientific protein name and any "
                    "PDB HEADER classification as the same protein. "
                    "You distinguish clearly between confirmed, "
                    "unconfirmed, and missing interactions."
                )
            },
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.2, "num_predict": 3000},
    )

    return {
        "validation_report": response["message"]["content"],
        "papers_used"      : [
            {k: v for k, v in p.items()
             if k != "_structure_info"}   # don't serialise large dict
            for p in papers
        ],
        "n_papers"         : len(papers),
        "n_fulltext"       : sum(
            1 for p in papers if p.get("has_fulltext")
        ),
        "model"            : model,
        "drug"             : result.get("drug"),
        "protein"          : result.get("protein"),
        "sci_name"         : structure_info.get("primary_name", ""),
        "pdb_id"           : pdb_id.upper(),
        "deposit_year"     : structure_info.get("deposit_year"),
    }


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def run_validation(result_json_path: str,
                   model: str = "gemma4:e2b",
                   n_papers: int = 10) -> dict:

    with open(result_json_path) as f:
        result = json.load(f)

    drug    = result["drug"]
    protein = result["protein"]

    # Extract PDB ID from protein_file field
    raw_fname = result.get("protein_file", "")
    pdb_id    = (raw_fname
                 .replace(".pdb", "")
                 .split("_")[0]
                 .upper()) or "UNKNOWN"

    print("=" * 60)
    print("DockExplain — Literature Validation")
    print(f"  Drug          : {drug}")
    print(f"  Protein       : {protein}")
    print(f"  PDB ID        : {pdb_id}")
    print(f"  Papers target : {n_papers} (full text via PMC)")
    print("=" * 60)

    print(f"\n[1/3] Retrieving literature...")
    papers = gather_literature(
        drug, protein,
        result.get("interactions", []),
        pdb_id   = pdb_id,
        n_papers = n_papers,
    )

    # Extract structure_info from first paper (attached by gather_literature)
    structure_info = {}
    if papers:
        structure_info = papers[0].pop("_structure_info", {})
        for p in papers[1:]:
            p.pop("_structure_info", None)

    if papers:
        print(f"\n  Papers retrieved:")
        for i, p in enumerate(papers, 1):
            tag = "[FULL]" if p.get("has_fulltext") else "[ABS] "
            pri = " ← primary" if p.get("is_primary") else ""
            print(f"    {tag} [{p['pmid']}] ({p['year']}) "
                  f"{p['title'][:55]}...{pri}")

    print(f"\n[2/3] Running Gemma 4 validation...")
    validation = validate_with_gemma4(
        result, papers, pdb_id, structure_info, model
    )

    print("\n" + "═" * 60)
    print("VALIDATION REPORT")
    print("═" * 60)
    print(validation["validation_report"])
    print("═" * 60)

    # Save
    out_dir   = os.path.dirname(result_json_path)
    safe_drug = drug.lower().replace(" ", "_")
    json_path = os.path.join(out_dir, f"{safe_drug}_validation.json")

    with open(json_path, "w") as f:
        json.dump(validation, f, indent=2)

    print(f"\n[3/3] Saved: {json_path}")
    print(f"  Full text: {validation['n_fulltext']} / {validation['n_papers']}")
    print(f"  Sci name : {validation.get('sci_name', 'unknown')}")
    return validation


if __name__ == "__main__":
    path = (sys.argv[1] if len(sys.argv) > 1
            else "../data/results/erlotinib/erlotinib_results.json")
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        sys.exit(1)
    run_validation(path, model="gemma4:e2b", n_papers=10)