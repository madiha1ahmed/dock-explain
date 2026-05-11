# literature_validator.py
# Fetches relevant PubMed abstracts and uses Gemma 4 to
# validate docking results against published literature.
#
# Run: python literature_validator.py \
#          ../data/results/mode_a/erlotinib_results.json

import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import ollama
import sys
import os



# ════════════════════════════════════════════════════════════
# STEP 1: Fetch relevant abstracts from PubMed
# Uses NCBI E-utilities — free, no API key needed
# ════════════════════════════════════════════════════════════

def search_pubmed(query: str, max_results: int = 5) -> list[str]:
    """
    Search PubMed and return a list of PMIDs.
    Uses the NCBI E-utilities esearch endpoint.
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    params = urllib.parse.urlencode({
        "db"     : "pubmed",
        "term"   : query,
        "retmax" : max_results,
        "sort"   : "relevance",
        "retmode": "json",
    })
    url = f"{base}esearch.fcgi?{params}"

    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        pmids = data["esearchresult"]["idlist"]
        return pmids
    except Exception as e:
        print(f"  ⚠ PubMed search failed: {e}")
        return []


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """
    Fetch titles and abstracts for a list of PMIDs.
    Uses the NCBI E-utilities efetch endpoint.
    Returns list of dicts with title, abstract, pmid.
    """
    if not pmids:
        return []

    base   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    params = urllib.parse.urlencode({
        "db"     : "pubmed",
        "id"     : ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    })
    url = f"{base}efetch.fcgi?{params}"

    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            xml_data = r.read()
    except Exception as e:
        print(f"  ⚠ Abstract fetch failed: {e}")
        return []

    # Parse XML
    articles = []
    try:
        root = ET.fromstring(xml_data)
        for article in root.findall(".//PubmedArticle"):
            # Title
            title_el = article.find(
                ".//ArticleTitle"
            )
            title = (title_el.text or "No title"
                     if title_el is not None else "No title")

            # Abstract — may have multiple AbstractText sections
            abstract_parts = article.findall(
                ".//AbstractText"
            )
            abstract = " ".join(
                (el.text or "") for el in abstract_parts
                if el.text
            )

            # PMID
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else "?"

            # Year
            year_el = article.find(".//PubDate/Year")
            year = year_el.text if year_el is not None else "?"

            if abstract:   # skip articles with no abstract
                articles.append({
                    "pmid"    : pmid,
                    "year"    : year,
                    "title"   : title,
                    "abstract": abstract[:1500],  # cap length
                    "url"     : f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
    except Exception as e:
        print(f"  ⚠ XML parsing failed: {e}")

    return articles


def gather_literature(drug: str, protein: str,
                      interactions: list[dict],
                      n_papers: int = 10) -> list[dict]:
    """
    Build targeted PubMed queries from the docking results
    and fetch the most relevant abstracts.

    Searches for:
      1. Drug + protein + binding interactions
      2. Drug + specific key residues found by ProLIF
      3. Drug + protein + crystal structure
    """
    # Extract the most important residues (H-bonds first)
    hbond_residues = [
        ix["residue"].split(".")[0]
        for ix in interactions
        if ix["interaction_type"] in ("HBDonor","HBAcceptor")
    ]
    key_residues = hbond_residues[:3]   # top 3 H-bond residues

    # Build queries from specific to general
    queries = []

    # Query 1: Drug + protein + binding mode
    queries.append(
        f"{drug} {protein} binding interactions docking"
    )

    # Query 2: Drug + specific residues detected by ProLIF
    if key_residues:
        res_str = " ".join(key_residues)
        queries.append(
            f"{drug} {protein} {res_str} binding site"
        )

    # Query 3: Crystal structure analysis
    queries.append(
        f"{drug} {protein} crystal structure ATP binding pocket"
    )

    print(f"\n  PubMed queries:")
    all_pmids = []
    seen      = set()

    for q in queries:
        print(f"    → {q}")
        pmids = search_pubmed(q, max_results=3)
        for p in pmids:
            if p not in seen:
                all_pmids.append(p)
                seen.add(p)
        time.sleep(0.35)   # NCBI rate limit: 3 requests/second

    # Keep top n_papers unique results
    all_pmids = all_pmids[:n_papers]
    print(f"  Found {len(all_pmids)} unique papers")

    if not all_pmids:
        return []

    abstracts = fetch_abstracts(all_pmids)
    print(f"  Fetched {len(abstracts)} abstracts")
    return abstracts


# ════════════════════════════════════════════════════════════
# STEP 2: Build the validation prompt
# ════════════════════════════════════════════════════════════

def build_validation_prompt(result: dict,
                             abstracts: list[dict]) -> str:
    """
    Build a prompt that asks Gemma 4 to compare our
    computational results against the literature abstracts.
    """
    drug    = result["drug"]
    protein = result["protein"]
    score   = result.get("docking_score")
    score_str = (f"{score:.2f} kcal/mol" if score
                 else "crystallographic pose")

    # Format interactions
    interaction_lines = []
    for ix in result.get("interactions", []):
        interaction_lines.append(
            f"  {ix['residue']}: {ix['interaction_type']}"
        )
    interaction_block = "\n".join(interaction_lines)

    # Format abstracts
    abstract_block = ""
    for i, ab in enumerate(abstracts, 1):
        abstract_block += (
            f"\n[Paper {i}] PMID:{ab['pmid']} ({ab['year']})\n"
            f"Title: {ab['title']}\n"
            f"Abstract: {ab['abstract']}\n"
            f"URL: {ab['url']}\n"
        )

    if not abstracts:
        abstract_block = (
            "\n[No abstracts retrieved — "
            "validate from general chemistry knowledge]\n"
        )

    prompt = f"""You are a computational chemistry expert validating 
docking simulation results against published scientific literature.

═══════════════════════════════════════
OUR COMPUTATIONAL RESULTS
═══════════════════════════════════════
Drug         : {drug}
Protein      : {protein}
Binding score: {score_str}

Interactions detected by ProLIF:
{interaction_block}

═══════════════════════════════════════
RELEVANT PUBLISHED LITERATURE
═══════════════════════════════════════
{abstract_block}

═══════════════════════════════════════
YOUR VALIDATION TASK
═══════════════════════════════════════
Compare our computational results against the literature above.
Structure your response as follows:

1. CONFIRMED INTERACTIONS
   List each interaction our pipeline found that IS supported 
   by the literature. Quote the specific paper and residue.
   Example: "MET769 H-bond — confirmed by [Paper 1], which 
   reports this as the primary anchor interaction."

2. UNCONFIRMED INTERACTIONS
   List interactions our pipeline found that are NOT mentioned
   in any of the retrieved abstracts. Note whether they are
   still chemically plausible even if unconfirmed.

3. MISSING INTERACTIONS
   List any interactions mentioned in the literature that our
   pipeline did NOT detect. Suggest why they might have been
   missed (e.g. hydrogen placement, distance threshold).

4. SCORE ASSESSMENT
   Compare our binding score ({score_str}) to any affinity
   values mentioned in the literature. Is our score in the
   expected range? Explain any discrepancy.

5. OVERALL CONFIDENCE
   Give an overall confidence rating:
     HIGH   — >70% of our interactions confirmed by literature
     MEDIUM — 40-70% confirmed
     LOW    — <40% confirmed or major discrepancies found
   
   Justify the rating in 2-3 sentences.

6. CAVEATS
   Note any important limitations of this validation:
   - Differences in protein construct, mutation status, or 
     experimental conditions between our structure and the 
     literature
   - Threshold differences between our ProLIF settings and
     literature methods
   - Any residue renumbering that may affect comparison

Be specific. Cite paper numbers. Name residues by number.
Do not be vague. If the literature doesn't address a point,
say so explicitly rather than speculating.
"""
    return prompt


# ════════════════════════════════════════════════════════════
# STEP 3: Run Gemma 4 validation
# ════════════════════════════════════════════════════════════

def validate_with_gemma4(result: dict,
                          abstracts: list[dict],
                          model: str = "gemma4:e2b") -> dict:
    """
    Ask Gemma 4 to validate our results against literature.
    Returns validation report dict.
    """
    prompt = build_validation_prompt(result, abstracts)

    print(f"\n  Sending to Gemma 4 for validation...")

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous computational chemistry reviewer. "
                    "You validate docking results against published literature "
                    "with scientific precision. You always cite specific papers "
                    "and residue numbers. You distinguish clearly between what "
                    "is confirmed, unconfirmed, and contradicted."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.2,   # very low — we want factual, not creative
            "num_predict": 2500,
        }
    )

    return {
        "validation_report" : response["message"]["content"],
        "papers_used"       : [
            {
                "pmid" : ab["pmid"],
                "year" : ab["year"],
                "title": ab["title"],
                "url"  : ab["url"],
            }
            for ab in abstracts
        ],
        "n_papers"          : len(abstracts),
        "model"             : model,
        "drug"              : result["drug"],
        "protein"           : result["protein"],
    }


# ════════════════════════════════════════════════════════════
# MAIN — run full validation pipeline
# ════════════════════════════════════════════════════════════

def run_validation(result_json_path: str,
                   model: str = "gemma4:e2b",
                   n_papers: int = 10) -> dict:
    """
    Full validation pipeline:
      1. Load docking result JSON
      2. Search PubMed for relevant literature
      3. Fetch abstracts
      4. Ask Gemma 4 to validate against literature
      5. Save validation report

    Args:
        result_json_path : path to erlotinib_results.json
        model            : Ollama model name
        n_papers         : number of PubMed papers to fetch
    """
    # Load result
    with open(result_json_path) as f:
        result = json.load(f)

    drug    = result["drug"]
    protein = result["protein"]
    n_ix    = result["n_interactions"]

    print("=" * 58)
    print("DockExplain — Literature Validation")
    print(f"  Drug        : {drug}")
    print(f"  Protein     : {protein}")
    print(f"  Interactions: {n_ix}")
    print("=" * 58)

    # Step 1: Gather literature
    print(f"\n[1/3] Searching PubMed...")
    abstracts = gather_literature(
        drug, protein,
        result.get("interactions", []),
        n_papers=n_papers
    )

    # Show what we found
    if abstracts:
        print(f"\n  Papers retrieved:")
        for ab in abstracts:
            print(f"    [{ab['pmid']}] ({ab['year']}) {ab['title'][:70]}...")
    else:
        print("  No abstracts found — "
              "will validate from Gemma 4 training knowledge")

    # Step 2: Validate with Gemma 4
    print(f"\n[2/3] Running Gemma 4 validation...")
    validation = validate_with_gemma4(result, abstracts, model)

    # Step 3: Print and save
    print("\n" + "═" * 58)
    print("VALIDATION REPORT")
    print("═" * 58)
    print(validation["validation_report"])
    print("═" * 58)

    # Save alongside the original results
    out_dir   = os.path.dirname(result_json_path)
    safe_drug = drug.lower().replace(" ","_")
    out_path  = os.path.join(
        out_dir, f"{safe_drug}_validation.json"
    )
    txt_path  = os.path.join(
        out_dir, f"{safe_drug}_validation.txt"
    )

    with open(out_path, "w") as f:
        json.dump(validation, f, indent=2)

    with open(txt_path, "w") as f:
        f.write(f"DockExplain — Literature Validation\n")
        f.write(f"Drug: {drug} | Protein: {protein}\n")
        f.write("=" * 60 + "\n\n")
        f.write(validation["validation_report"])
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("Papers used:\n")
        for p in validation["papers_used"]:
            f.write(f"  [{p['pmid']}] ({p['year']}) {p['title']}\n")
            f.write(f"  {p['url']}\n\n")

    print(f"\n[3/3] Saved:")
    print(f"  ✓ JSON: {out_path}")
    print(f"  ✓ TXT : {txt_path}")

    return validation


if __name__ == "__main__":
    json_path = (
        sys.argv[1] if len(sys.argv) > 1
        else "../data/results/mode_a/erlotinib_results.json"
    )

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        print("Run dockexplain_pipeline.py first")
        sys.exit(1)

    run_validation(json_path, model="gemma4:e2b", n_papers=10)