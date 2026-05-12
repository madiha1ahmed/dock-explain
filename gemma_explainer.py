# gemma_explainer.py
# DockExplain — Gemma 4 plain-language explanation of docking results
#
# Changes from previous version:
#   - Integrates web_search_enricher.py for literature context
#   - Explicitly passes PyMOL interaction data into the prompt so
#     Gemma 4 knows which residues are visualized and how they appear
#   - Adds a LITERATURE GROUNDING section to the prompt
#   - Adds a PYMOL VISUALIZATION section so explanation references
#     what the researcher will actually see in the 3D view
#
# Run standalone:
#   python gemma_explainer.py \
#       ../data/results/erlotinib/erlotinib_results.json
#
# Or import into run_dockexplain.py:
#   from gemma_explainer import explain_docking_result, save_explanation

import json
import os
import sys
import re
import ollama
from collections import defaultdict

# ── Local module ─────────────────────────────────────────────────────────────
from web_search_enricher import (
    enrich_docking_context,
    format_context_for_prompt,
    format_bibliography,
    LiteratureContext,
)

# Import the protonation summary helper
try:
    from dockexplain_pipeline import protonation_note as _protonation_note
except ImportError:
    def _protonation_note(ph: float) -> str:
        if abs(ph - 7.4) < 0.1:
            return f"pH {ph:.1f} — standard physiological conditions."
        return (
            f"pH {ph:.1f} — non-physiological. Titratable residues "
            "(ASP pKa 3.9, GLU pKa 4.3, HIS pKa 6.0, CYS pKa 8.3) "
            "may change protonation state compared to pH 7.4. "
            "Electrostatic interactions involving these residues "
            "should be interpreted with caution."
        )


# ════════════════════════════════════════════════════════════════════════════
# INTERACTION GUIDANCE
# Atom-level explanations for multi-role residues.
# ════════════════════════════════════════════════════════════════════════════

def _get_multi_role_guidance(resname: str, types: list) -> str:
    type_set = set(types)
    hbond  = type_set & {"HBDonor", "HBAcceptor"}
    hydro  = type_set & {"Hydrophobic"}
    pi     = type_set & {"PiStacking", "EdgeToFace"}
    ionic  = type_set & {"Cationic", "Anionic"}

    if resname == "MET" and hbond and hydro:
        return (
            "Methionine has a sulfur atom (S) that acts as a weak "
            "H-bond acceptor, AND a hydrophobic carbon chain "
            "(CH₂-CH₂-S). The sulfur anchors the drug via H-bonding "
            "while the carbon chain makes hydrophobic contact."
        )
    if resname == "LYS" and hbond and hydro:
        return (
            "Lysine has a positively charged amine (NH₃⁺) at the "
            "tip that H-bonds, AND a long aliphatic chain (-(CH₂)₄-) "
            "that is hydrophobic."
        )
    if resname in ("ASP", "GLU") and ionic and hydro:
        return (
            f"{resname} carries a negatively charged carboxylate "
            "(COO⁻) for ionic attraction while its CH₂ backbone "
            "makes weak hydrophobic contact."
        )
    if resname in ("PHE", "TYR") and pi and hydro:
        return (
            f"{resname} has a flat aromatic ring that π-stacks with "
            "the drug's ring system, AND the ring carbons contribute "
            "hydrophobic surface — both from the same aromatic ring."
        )
    if resname == "THR" and hbond and hydro:
        return (
            "Threonine has a hydroxyl (-OH) that H-bonds AND a "
            "methyl group (-CH₃) that makes hydrophobic contact."
        )
    if resname == "CYS" and hbond and hydro:
        return (
            "Cysteine has a thiol (-SH) that weak H-bonds via sulfur "
            "AND a short aliphatic carbon (-CH₂-) that is hydrophobic."
        )
    # Generic fallback
    type_descriptions = {
        "HBDonor"    : "hydrogen bond donation (-NH or -OH group)",
        "HBAcceptor" : "hydrogen bond acceptance (electronegative atom)",
        "Hydrophobic": "hydrophobic contact (non-polar carbon atoms)",
        "PiStacking" : "π–π ring stacking (aromatic ring face-to-face)",
        "EdgeToFace" : "T-shaped ring interaction",
        "Cationic"   : "electrostatic attraction (positive charge)",
        "Anionic"    : "electrostatic attraction (negative charge)",
        "CationPi"   : "cation-π interaction",
        "PiCation"   : "π-cation interaction",
    }
    roles = " AND ".join(type_descriptions.get(t, t) for t in types)
    return (
        f"This residue contributes via {roles}. "
        "Explain the specific atoms responsible for each role."
    )


# ════════════════════════════════════════════════════════════════════════════
# PYMOL CONTEXT BUILDER
# Translates the interaction fingerprint into a PyMOL-awareness block.
# This tells Gemma 4 exactly what is colour-coded in the 3D view
# so its explanation can reference the visualization directly.
# ════════════════════════════════════════════════════════════════════════════

# Mirrors the colour scheme in pymol_visualize.py
_PYMOL_COLOR_LABELS = {
    "HBDonor"    : "dark blue sticks",
    "HBAcceptor" : "blue sticks",
    "Hydrophobic": "amber/orange sticks",
    "PiStacking" : "purple sticks",
    "EdgeToFace" : "light purple sticks",
    "Cationic"   : "dark red sticks",
    "Anionic"    : "pink sticks",
    "CationPi"   : "cyan sticks",
    "PiCation"   : "teal sticks",
}

_PYMOL_DISTANCE_TYPES = {
    "HBDonor", "HBAcceptor", "Cationic", "Anionic", "CationPi", "PiCation"
}


def _build_pymol_context_block(interactions: list) -> str:
    """
    Describe what is visible in the PyMOL session so Gemma 4 can
    ground its explanation in what the researcher actually sees.

    The interactions list comes from the same ProLIF fingerprint that
    pymol_visualize.py uses — so this block is always in sync with
    the 3D view.
    """
    if not interactions:
        return "No interactions detected — PyMOL shows the ligand only."

    residue_map = defaultdict(list)
    for ix in interactions:
        residue_map[ix["residue"]].append(ix["interaction_type"])

    lines = []
    has_distance_dashes = False

    for residue, types in residue_map.items():
        primary = types[0]
        color   = _PYMOL_COLOR_LABELS.get(primary, "green sticks")
        has_dash = any(t in _PYMOL_DISTANCE_TYPES for t in types)
        if has_dash:
            has_distance_dashes = True

        dash_note = " + dashed distance line" if has_dash else ""
        multi_note = (
            f" [MULTI-ROLE: {'+'.join(types)}]" if len(types) > 1 else ""
        )
        lines.append(
            f"  • {residue}: shown as {color}{dash_note}{multi_note}"
        )

    summary = "\n".join(lines)

    notes = []
    notes.append("The protein backbone is shown as a semi-transparent blue-grey cartoon.")
    notes.append(f"The drug ({_get_drug_name_placeholder()}) is shown as green sticks with spheres.")
    if has_distance_dashes:
        notes.append(
            "Dashed lines connect the drug to polar-interaction residues "
            "(H-bonds, ionic contacts) — representing the actual distances "
            "measured by PyMOL."
        )
    notes.append(
        "A translucent surface shows the binding pocket shape "
        "(residues within 8 Å of the ligand)."
    )
    notes.append(
        "Every interacting residue is labelled with its name and "
        "interaction type directly in the 3D view."
    )

    return (
        "PyMOL 3D Visualization — what the researcher sees:\n"
        + summary
        + "\n\nVisualization notes:\n"
        + "\n".join(f"  - {n}" for n in notes)
    )


def _get_drug_name_placeholder():
    # This is filled dynamically in build_prompt; placeholder used here
    return "the drug"


# ════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_prompt(
    result  : dict,
    lit_ctx : LiteratureContext | None = None,
) -> str:
    """
    Build a rich, structured prompt from the pipeline JSON plus
    optional literature context from web_search_enricher.

    Args:
        result  : dict loaded from *_results.json
        lit_ctx : LiteratureContext from enrich_docking_context(),
                  or None to skip literature grounding

    Returns:
        Full prompt string for Gemma 4
    """
    drug         = result.get("drug", "Unknown drug")
    protein      = result.get("protein", "Unknown protein")
    score        = result.get("docking_score")
    interactions = result.get("interactions", [])
    mode         = result.get("mode", "?")
    ph           = result.get("ph", 7.4)
    ph_note      = _protonation_note(ph)

    # ── Override names with resolved values from web enrichment ─────────
    # The pipeline JSON stores raw PDB strings ("TRANSFERASE", "KINASE").
    # lit_ctx.gene_symbol and lit_ctx.protein_display_name hold the names
    # actually used in all the web searches — use those throughout the
    # prompt so Gemma 4 reasons about the correct, identified protein.
    # Similarly, if the pipeline stored a long IUPAC name but lit_ctx
    # has a cleaner display name, prefer that for readability.
    if lit_ctx is not None:
        if lit_ctx.gene_symbol:
            resolved_protein = (
                lit_ctx.protein_display_name
                if lit_ctx.protein_display_name
                else lit_ctx.gene_symbol.upper()
            )
            if resolved_protein != protein:
                print(f"  [build_prompt] Protein name resolved: "
                      f"'{protein}' → '{resolved_protein}' "
                      f"(gene: {lit_ctx.gene_symbol})")
            protein = resolved_protein

        # ── Drug name substitution — ONLY for genuine IUPAC-to-INN mappings ──
        # Guard: only substitute if the original name looks like a full IUPAC
        # string (long, starts with bracket/paren). Do NOT substitute for
        # short identifiers like "RefChem:168278" or "ONO-3080573" — these are
        # intentional user-provided names, not IUPAC blobs needing simplification.
        # Substituting them risks silently replacing a correct identifier with
        # a structurally different compound's INN name.
        lit_drug = getattr(lit_ctx, "drug_display_name", "")
        _original_drug = result.get("drug", drug)   # raw name from pipeline JSON
        _looks_iupac   = (
            len(_original_drug) > 30
            and any(c in _original_drug for c in "([")
            and _original_drug[0] in "(["
        )
        if lit_drug and lit_drug != drug and _looks_iupac:
            print(f"  [build_prompt] Drug name resolved: "
                  f"'{drug}' → '{lit_drug}' (IUPAC → INN via PubChem)")
            drug = lit_drug
        elif lit_drug and lit_drug != drug and not _looks_iupac:
            # Name differs but original is not IUPAC — do NOT substitute,
            # just log so the researcher can see what PubChem returned
            print(f"  [build_prompt] PubChem suggested '{lit_drug}' for "
                  f"'{drug}' — NOT substituted (original is not an IUPAC name). "
                  f"Verify compound identity before interpreting results.")

    score_str = (
        f"{score:.2f} kcal/mol"
        if score is not None
        else "crystallographic pose (experimental binding mode)"
    )

    # ── Group interactions by residue ────────────────────────────────────
    residue_map = defaultdict(list)
    for ix in interactions:
        residue_map[ix["residue"]].append(ix["interaction_type"])

    PRIORITY = {
        "HBDonor":1, "HBAcceptor":1,
        "Cationic":2, "Anionic":2,
        "PiStacking":3, "EdgeToFace":3,
        "CationPi":3, "PiCation":3,
        "Hydrophobic":4,
    }

    def top_priority(types):
        return min(PRIORITY.get(t, 99) for t in types)

    sorted_residues = sorted(
        residue_map.items(),
        key=lambda x: top_priority(x[1])
    )

    # ── Interaction block ────────────────────────────────────────────────
    interaction_lines   = []
    multi_role_residues = []

    for residue, types in sorted_residues:
        types_str = " + ".join(types)
        if len(types) > 1:
            multi_role_residues.append((residue, types))
            interaction_lines.append(
                f"  {residue}: {types_str}"
                f"  ← MULTI-ROLE RESIDUE (explain each role separately)"
            )
        else:
            interaction_lines.append(f"  {residue}: {types_str}")

    interaction_block = "\n".join(interaction_lines) or "  No interactions detected."

    # ── Multi-role guidance ──────────────────────────────────────────────
    multi_role_section = ""
    if multi_role_residues:
        lines = []
        for residue, types in multi_role_residues:
            resname = residue.strip().split(".")[0]
            m = re.match(r"([A-Z]+)\d+", resname)
            resname_code = m.group(1) if m else resname
            guidance = _get_multi_role_guidance(resname_code, types)
            lines.append(f"  {residue}: {guidance}")

        multi_role_section = (
            "\n\nIMPORTANT — Multi-role residues (explain each role with atoms):\n"
            + "\n".join(lines)
            + "\n\nFor each: name the specific atoms causing each interaction, "
            "and explain why the same residue can play two different roles."
        )

    # ── Novel compound notice ────────────────────────────────────────────
    # Detect novel compound FIRST — before novel_notice is built
    _novel_flag = (
        lit_ctx is not None
        and any("novel compound" in w.lower() for w in lit_ctx.warnings)
    )
    if not _novel_flag:
        try:
            from web_search_enricher import is_novel_compound
            _novel_flag = is_novel_compound(drug)
        except Exception:
            pass

    if _novel_flag:
        novel_notice = (
            "\n⚠ NOVEL COMPOUND NOTICE:\n"
            f"  '{drug}' is a user-assigned name for a molecule not present\n"
            "  in any public database (e.g. from a generative model or\n"
            "  unpublished research). There is NO published pharmacology,\n"
            "  mechanism of action, or binding data for this specific molecule.\n"
            "  Your explanation MUST:\n"
            "    - Describe interactions purely from the computed 3D pose\n"
            "    - Use the protein context papers to describe the BINDING SITE\n"
            "    - NOT claim any known mechanism of action for the drug\n"
            "    - NOT cite literature as evidence of how THIS drug binds\n"
            "    - Explicitly state in the opening that this is a novel molecule\n"
            "    - In Section 7 (Plausibility), note that Criterion A and C\n"
            "      cannot be assessed since the drug's therapeutic context is unknown"
        )
    else:
        novel_notice = ""

    # ── PyMOL visualization block ────────────────────────────────────────
    # The interactions in this block come from the same ProLIF fingerprint
    # that pymol_visualize.py uses to generate the 3D view.
    # Gemma's explanation should refer to what the researcher sees.
    pymol_block = _build_pymol_context_block(interactions).replace(
        _get_drug_name_placeholder(), drug
    )

    # ── Literature context block ─────────────────────────────────────────
    LIT_BLOCK_MAX = 3500

    if lit_ctx is not None:
        lit_block_full = format_context_for_prompt(lit_ctx)
        if len(lit_block_full) > LIT_BLOCK_MAX:
            lit_block = (
                lit_block_full[:LIT_BLOCK_MAX]
                + f"\n\n[Literature truncated to {LIT_BLOCK_MAX} chars "
                f"to fit model context window. Full bibliography in output file.]"
            )
        else:
            lit_block = lit_block_full
    else:
        lit_block = (
            "LITERATURE CONTEXT: Web search was not run for this session. "
            f"Rely on your training knowledge about {drug} and {protein}."
        )

    # ── Assemble full prompt ─────────────────────────────────────────────
    prompt = f"""You are an expert computational chemist explaining molecular
docking results to a mixed audience — from biology students to
medicinal chemists. Your goal is clear, specific, and insightful science.

═══════════════════════════════════════════════════════════════
DOCKING RESULT
═══════════════════════════════════════════════════════════════
Drug              : {drug}
Protein           : {protein}
Binding score     : {score_str}
Pipeline mode     : {mode}
Total interactions: {len(interactions)} (detected by ProLIF fingerprinting)

ProLIF interaction fingerprint
(These are the same residues highlighted in the PyMOL 3D view):
{interaction_block}
{multi_role_section}
{novel_notice}

pH AND PROTONATION CONTEXT
═══════════════════════════════════════════════════════════════
The protein was prepared at pH {ph:.1f} using PDBFixer with standard
Henderson-Hasselbalch protonation (textbook pKa values, not protein-
microenvironment corrected).

{ph_note}

CRITICAL INSTRUCTION FOR pH {ph:.1f}:
For every interaction involving a titratable residue (ASP, GLU, HIS,
CYS, LYS, ARG), explicitly state:
  1. Whether the residue's charge state at pH {ph:.1f} enables this interaction
  2. Whether this interaction would DIFFER at physiological pH 7.4
  3. If pH {ph:.1f} is non-physiological, flag which interactions are
     pH-conditional and may not exist in a normal cellular context

═══════════════════════════════════════════════════════════════
PYMOL 3D VISUALIZATION CONTEXT
═══════════════════════════════════════════════════════════════
The researcher is looking at a PyMOL session generated from the
same interaction data above. Your explanation should complement
what they see in 3D. Reference the colours and labels they will
observe — this bridges the visual and the molecular reasoning.

{pymol_block}

═══════════════════════════════════════════════════════════════
LITERATURE GROUNDING
═══════════════════════════════════════════════════════════════
The drug being analyzed is: {drug}
The protein target is     : {protein}

⚠ IMPORTANT — read before using these papers:
If the papers below discuss a DIFFERENT drug (e.g. a known inhibitor
of this protein, not {drug}), that means {drug} has limited published
binding data for {protein}. In that case:
  - Use the papers ONLY for protein pocket and binding site context
  - Do NOT attribute those papers' findings to {drug}
  - In Section 5 (Literature Cross-Check), name which drug the papers
    actually cover, state this explicitly, and distinguish what is
    known about {drug} vs what is inferred from protein context alone

{lit_block}

═══════════════════════════════════════════════════════════════
DATA REMINDER — refer to these specific values in your response
═══════════════════════════════════════════════════════════════
Drug              : {drug}
Protein           : {protein}
Binding score     : {score_str}
Interactions detected:
{interaction_block}

You MUST reference the specific residue names, interaction types,
and binding score above. Do not write generic placeholders.

═══════════════════════════════════════════════════════════════
YOUR TASK — Write a structured explanation (~650 words)
═══════════════════════════════════════════════════════════════

1. OPENING (2-3 sentences)
   Plain-English summary of the binding. What does {score_str} mean?
   What is the dominant interaction type?

2. KEY INTERACTIONS (one paragraph per type found)
   For each interaction type:
   - Name the specific residue(s) and atoms
   - Explain the physical/chemical property causing it
   - Explain its contribution to binding strength
   - Reference the colour the researcher sees in PyMOL
     (e.g. "the blue sticks of MET769 indicate an H-bond")

3. MULTI-ROLE RESIDUES (mandatory if any flagged above)
   For every MULTI-ROLE residue:
   - Each role gets its own sentence
   - Name the atoms responsible for each role
   - Explain why the same residue can play two roles
   - This is the most chemically insightful section

4. BINDING POCKET CHARACTER
   Describe the overall pocket (e.g. "a deep hydrophobic cleft
   lined with aromatic residues"). Explain why {drug}'s molecular
   structure suits this pocket. Ground in the literature evidence
   above if it confirms the pocket type.

5. LITERATURE CROSS-CHECK
   Compare the detected ProLIF interactions against what the
   literature evidence above says about how {drug} binds {protein}.
   Do the computational interactions match experimentally known
   contacts? Flag any surprises or confirmations explicitly.

6. CLINICAL RELEVANCE (2-3 sentences)
   Why does binding here matter therapeutically? Use the clinical
   context from literature to ground this — do not speculate beyond
   what the evidence supports.

7. BIOLOGICAL PLAUSIBILITY ASSESSMENT (mandatory for ALL modes)

   This section must always be written, regardless of pipeline mode.
   Evaluate the drug–protein pairing systematically using the five
   criteria below. For each criterion, write one sentence giving a
   verdict (PLAUSIBLE / UNLIKELY / NOT APPLICABLE) and a reason.

   ── Criterion A: Drug class vs. protein class ──────────────────
   Does the drug's known mechanism of action match the protein's
   function? Examples of mismatches:
     • A kinase inhibitor (e.g. Erlotinib, designed for EGFR tyrosine
       kinase) docked into a lipid-transport protein or GPCR
     • An Alzheimer's cholinesterase inhibitor docked into a cancer
       kinase target
     • An antibiotic docked into a human metabolic enzyme
   State clearly: is this drug designed for this class of protein?

   ── Criterion B: Pocket type compatibility ─────────────────────
   Does the protein have a well-defined small-molecule binding pocket
   appropriate for this drug?
     • Lipid-binding proteins (e.g. LPA acyltransferases, fatty acid
       carriers) have large, greasy, lipid-shaped cavities — small
       polar drugs like kinase inhibitors will sit awkwardly
     • Structural proteins, pure scaffolding proteins, or coiled-coil
       domains typically lack druggable pockets entirely
     • Ion channels require very specific size and charge complementarity
   State whether the detected binding pocket suits the drug's shape
   and polarity. Reference the interaction profile above as evidence.

   ── Criterion C: Disease area alignment ────────────────────────
   Are the drug's approved clinical indication(s) and the protein's
   disease role in the same therapeutic area?
     • Erlotinib treats NSCLC (non-small cell lung cancer) via EGFR;
       docking it into a lipid metabolism enzyme has no oncology rationale
     • A CNS drug docked into a peripheral inflammatory target may still
       have rationale (repurposing) — state this if applicable
   Use the clinical context from literature to make this judgment.
   Flag explicitly if this looks like an unintentional or test pairing.

   ── Criterion D: Interaction profile consistency ───────────────
   Does the detected interaction profile (from ProLIF above) match
   what you would expect for a genuine drug–protein complex?
     • Kinase inhibitors typically form 1-3 H-bonds with the hinge
       region and sit in a hydrophobic adenine pocket
     • If the profile shows ONLY hydrophobic contacts and no H-bonds
       for a drug that normally forms H-bonds, that is suspicious
     • If the docking score is weak (> -6 kcal/mol) AND the interaction
       profile is atypical, flag this combination explicitly
   Be specific: name which interactions are expected and which are missing.

   ── Criterion E: Overall verdict ───────────────────────────────
   Summarise the assessment in 2-3 sentences:
     • If ALL criteria pass: state the pairing is biologically plausible
       and interactions should be interpreted at face value
     • If 1-2 criteria fail: state the pairing is UNCERTAIN — interactions
       may be real but should be interpreted cautiously; suggest what
       experimental validation would confirm or refute the binding
     • If 3+ criteria fail: state clearly that this pairing is likely
       NOT biologically meaningful, that the detected interactions are
       probably artefactual, and explain what a more appropriate
       protein target for this drug would be (if known from literature)

   IMPORTANT: Do not soften a negative verdict with vague hedging.
   A clear "this pairing is biologically implausible because..." is
   more useful to the researcher than a cautious non-answer.

STYLE RULES:
- Define every chemistry term on first use
- Reference specific PyMOL colours/labels where relevant
- Cite PubMed papers by number when they support a claim:
  e.g. "MET769 forms a key H-bond [Paper 1]" or "consistent with [Paper 2]"
  If no paper supports a claim, say so rather than inventing a citation.
- Never say "as shown above" — each section is self-contained
- Be specific: name residues, atoms, interaction types
- Avoid vague phrases like "interacts with" — say HOW
- The protein was protonated at pH 7.4 (physiological standard).
  If you mention a charged residue (ASP, GLU, HIS, LYS, ARG), briefly
  note that its charge state assumes physiological pH 7.4. Do not
  speculate about interactions at other pH values — pH-dependent
  analysis is a planned future enhancement.
- Target: 700-800 words total (the plausibility section adds length)
"""
    # Inject ph, ph_note, and novel_notice into prompt placeholders
    prompt = prompt.replace("{ph:.1f}", f"{ph:.1f}")
    prompt = prompt.replace("{ph_note}", ph_note)
    prompt = prompt.replace("{novel_notice}", novel_notice)
    return prompt


# ════════════════════════════════════════════════════════════════════════════
# GEMMA 4 CALL
# ════════════════════════════════════════════════════════════════════════════

def explain_docking_result(
    result         : dict,
    model          : str  = "gemma4:e2b",
    verbose        : bool = True,
    run_web_search : bool = True,
    lit_ctx        : LiteratureContext | None = None,
) -> dict:
    """
    Generate a plain-language explanation via Gemma 4.

    The explanation is grounded in:
      1. ProLIF interaction fingerprint (source of truth for interactions)
      2. PyMOL visualization context (same interactions, described visually)
      3. Literature context from web_search_enricher (drug/protein/clinical)

    Args:
        result         : dict loaded from *_results.json
        model          : Ollama model name
        verbose        : print to terminal
        run_web_search : if True, auto-run web enrichment
                         (set False if you pass lit_ctx manually)
        lit_ctx        : pre-fetched LiteratureContext; if None and
                         run_web_search=True, will be fetched automatically

    Returns:
        dict with keys:
            explanation      — full text from Gemma 4
            prompt_used      — exact prompt sent
            model            — model used
            drug, protein    — names
            n_interactions   — count
            multi_role       — list of multi-role residues
            literature_used  — True/False
            sources          — list of URLs used for grounding
    """
    drug    = result.get("drug", "Unknown")
    protein = result.get("protein", "Unknown")

    # ── Step 1: Web search enrichment ───────────────────────────────────
    if lit_ctx is None and run_web_search:
        if verbose:
            print(f"\n  [Step 1/2] Running web search enrichment...")
        # Pass SMILES, full result dict, and PDB ID so the drug name resolver
        # can use the most reliable paths: SMILES → PubChem CID → synonym, and
        # RCSB CCD lookup for PDB 3-letter ligand codes (e.g. F86 → Remdesivir).
        # Without these, vendor catalog IDs like "orb1691391" can't be resolved.
        _drug_smiles = (
            result.get("smiles", "")
            or result.get("drug_smiles", "")
            or result.get("ligand_smiles", "")
        )
        _pdb_id = (
            result.get("pdb_id", "")
            or result.get("pdb", "")
            or result.get("structure_id", "")
        )
        lit_ctx = enrich_docking_context(
            drug        = drug,
            protein     = protein,
            drug_smiles = _drug_smiles,
            result_json = result,
            pdb_id      = _pdb_id,
            verbose     = verbose,
        )

    # ── Step 2: Build prompt ─────────────────────────────────────────────
    prompt = build_prompt(result, lit_ctx)

    # ── Identify multi-role residues ─────────────────────────────────────
    residue_map = defaultdict(list)
    for ix in result.get("interactions", []):
        residue_map[ix["residue"]].append(ix["interaction_type"])
    multi_role = [r for r, t in residue_map.items() if len(t) > 1]

    # Compute the resolved names that build_prompt() will actually use
    _raw_drug    = result.get("drug", drug)
    _raw_protein = result.get("protein", protein)
    _display_drug = (
        getattr(lit_ctx, "drug_display_name", "")
        if lit_ctx and getattr(lit_ctx, "drug_display_name", "")
           and len(getattr(lit_ctx, "drug_display_name", "")) < len(_raw_drug)
        else _raw_drug
    )
    _display_protein = (
        (lit_ctx.protein_display_name or lit_ctx.gene_symbol or _raw_protein)
        if lit_ctx else _raw_protein
    )

    if verbose:
        print(f"\n  [Step 2/2] Sending to Gemma 4 ({model})...")
        print(f"  Drug (raw)        : {_raw_drug}")
        if _display_drug != _raw_drug:
            print(f"  Drug (Gemma sees) : {_display_drug}  ← resolved via PubChem")
        print(f"  Protein (raw)     : {_raw_protein}")
        if _display_protein != _raw_protein:
            print(f"  Protein (Gemma)   : {_display_protein}  ← resolved via RCSB")
        print(f"  Interactions      : {result.get('n_interactions', 0)}")
        print(f"  Literature sources: {len(lit_ctx.sources) if lit_ctx else 0}")
        if multi_role:
            print(f"  Multi-role        : {', '.join(multi_role)}")
        print(f"  {'─'*54}")

    # ── Gemma 4 call ─────────────────────────────────────────────────────
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are DockExplain — an AI that explains molecular "
                    "docking results accurately and accessibly. "
                    "CRITICAL IDENTITY RULE: The drug named in the DOCKING "
                    "RESULT and DATA REMINDER sections is the ONLY drug being "
                    "studied. NEVER replace it with a drug name from the "
                    "literature — even if papers repeatedly mention a different "
                    "drug bound to the same protein. Literature papers provide "
                    "protein/pocket context, not drug identity. If the papers "
                    "cover a different compound, say so explicitly and use them "
                    "only for protein context. "
                    "You ground your explanations in the provided literature "
                    "evidence, name specific residues and atoms, define every "
                    "chemistry term on first use, and reference what the "
                    "researcher sees in the PyMOL 3D visualization. "
                    "You never use vague language like 'interacts with'. "
                    "You always say HOW: H-bond, hydrophobic contact, "
                    "π-stacking, ionic attraction, etc."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature" : 0.3,
            "num_predict" : 5000,
            "num_ctx"     : 16384,   # extend context window for long prompts
        }
    )

    explanation = response["message"]["content"]

    if verbose:
        print(f"\n{'═'*58}")
        print("  DockExplain — Gemma 4 Explanation")
        print(f"{'═'*58}")
        print(explanation)
        print(f"{'═'*58}")

    # Build bibliography from all sources in lit_ctx
    bibliography = ""
    if lit_ctx is not None:
        try:
            bibliography = format_bibliography(lit_ctx)
        except Exception:
            bibliography = ""

    return {
        "explanation"     : explanation,
        "prompt_used"     : prompt,
        "model"           : model,
        "drug"            : drug,
        "protein"         : protein,
        "n_interactions"  : result.get("n_interactions", 0),
        "multi_role"      : multi_role,
        "literature_used" : lit_ctx is not None,
        "sources"         : lit_ctx.sources if lit_ctx else [],
        "n_papers"        : len(lit_ctx.papers) if lit_ctx else 0,
        "bibliography"    : bibliography,
    }


# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════

def save_explanation(explained: dict, out_path: str):
    """Save explanation dict to JSON + plain text."""
    with open(out_path, "w") as f:
        json.dump(explained, f, indent=2)
    print(f"  ✓ Explanation JSON saved: {out_path}")

    txt_path = out_path.replace(".json", ".txt")
    with open(txt_path, "w") as f:
        drug    = explained.get("drug", "")
        protein = explained.get("protein", "")
        n       = explained.get("n_interactions", 0)
        model   = explained.get("model", "")
        sources = explained.get("sources", [])

        f.write(f"DockExplain — {drug} binding to {protein}\n")
        f.write(f"Model: {model} | Interactions explained: {n}\n")
        if sources:
            f.write(f"Literature sources used: {len(sources)}\n")
            for s in sources[:8]:          # cap at 8 in the text file
                f.write(f"  • {s}\n")
        f.write("=" * 60 + "\n\n")
        f.write(explained.get("explanation", ""))

        multi = explained.get("multi_role", [])
        if multi:
            f.write(f"\n\n[Multi-role residues: {', '.join(multi)}]")

        # Append bibliography if available
        bib = explained.get("bibliography", "")
        if bib:
            f.write("\n\n" + "=" * 60 + "\n")
            f.write(bib)

    print(f"  ✓ Explanation TXT saved: {txt_path}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DockExplain — Gemma 4 explanation with web search"
    )
    parser.add_argument(
        "json", nargs="?",
        default="../data/results/erlotinib/erlotinib_results.json",
        help="Path to *_results.json from dockexplain_pipeline"
    )
    parser.add_argument(
        "--model", default="gemma4:e2b",
        help="Ollama model (default: gemma4:e2b)"
    )
    parser.add_argument(
        "--no-web-search", action="store_true",
        help="Skip web search (use Gemma training knowledge only)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"Error: {args.json} not found")
        print("Run dockexplain_pipeline.py first")
        sys.exit(1)

    with open(args.json) as f:
        result = json.load(f)

    print(f"\n{'═'*58}")
    print("  DockExplain — Gemma 4 Explainer")
    print(f"  Drug        : {result.get('drug')}")
    print(f"  Protein     : {result.get('protein')}")
    print(f"  Interactions: {result.get('n_interactions', 0)}")
    print(f"  Web search  : {'disabled' if args.no_web_search else 'enabled'}")
    print(f"{'═'*58}")

    explained = explain_docking_result(
        result,
        model          = args.model,
        run_web_search = not args.no_web_search,
    )

    out_dir   = os.path.dirname(args.json)
    safe_drug = result.get("drug", "drug").lower().replace(" ", "_")
    out_path  = os.path.join(out_dir, f"{safe_drug}_explanation.json")
    save_explanation(explained, out_path)
