# gemma_explainer.py
# Week 3: Gemma 4 plain-language explanation of docking results
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
import ollama
from collections import defaultdict


# ════════════════════════════════════════════════════════════
# INTERACTION GUIDANCE
# Atom-level explanations for multi-role residues.
# Gemma 4 uses these to explain WHY the same residue
# contributes two different interaction types.
# ════════════════════════════════════════════════════════════

def _get_multi_role_guidance(resname: str, types: list) -> str:
    """
    Return atom-level guidance for specific multi-role residues.
    Covers the most common cases seen in drug-protein binding.
    """
    type_set = set(types)
    hbond    = type_set & {"HBDonor", "HBAcceptor"}
    hydro    = type_set & {"Hydrophobic"}
    pi       = type_set & {"PiStacking", "EdgeToFace"}
    ionic    = type_set & {"Cationic", "Anionic"}

    # Methionine: sulfur H-bonds, carbon chain hydrophobic
    if resname == "MET" and hbond and hydro:
        return (
            "Methionine has a sulfur atom (S) that acts as a weak "
            "H-bond acceptor, AND a hydrophobic carbon chain "
            "(CH₂-CH₂-S). Explain that the sulfur anchors the drug "
            "via H-bonding while the carbon chain makes greasy "
            "hydrophobic contact — two completely different atoms, "
            "two different physical forces, same residue."
        )

    # Lysine: charged amine H-bonds + aliphatic chain hydrophobic
    if resname == "LYS" and hbond and hydro:
        return (
            "Lysine has a positively charged amine (NH₃⁺) at the "
            "tip that H-bonds, AND a long aliphatic chain (-(CH₂)₄-) "
            "that is hydrophobic. Explain that the charged tip "
            "interacts via H-bond while the hydrocarbon tail makes "
            "hydrophobic contact."
        )

    # Aspartate/Glutamate: charged carboxylate + hydrophobic backbone
    if resname in ("ASP", "GLU") and ionic and hydro:
        return (
            f"{resname} carries a negatively charged carboxylate "
            "(COO⁻) that creates ionic/electrostatic attraction, "
            "while its CH₂ backbone makes weak hydrophobic contact. "
            "Explain that opposite charges attract the drug's "
            "positive region while the backbone contributes to "
            "hydrophobic packing."
        )

    # Phenylalanine/Tyrosine: pi-stacking ring + hydrophobic
    if resname in ("PHE", "TYR") and pi and hydro:
        return (
            f"{resname} has a flat aromatic ring that π-stacks with "
            "the drug's ring system, AND the ring carbons are "
            "hydrophobic. Explain that π-stacking is a specific "
            "geometric interaction between electron clouds, while "
            "hydrophobic contact is a more general surface effect — "
            "both arising from the same aromatic ring."
        )

    # Threonine: hydroxyl H-bonds, methyl hydrophobic
    if resname == "THR" and hbond and hydro:
        return (
            "Threonine has a hydroxyl group (-OH) that H-bonds, "
            "AND a methyl group (-CH₃) that makes hydrophobic "
            "contact. The -OH is polar and attracted to polar parts "
            "of the drug, while -CH₃ is greasy — both on the same "
            "residue but different atoms."
        )

    # Cysteine: sulfur H-bonds + hydrophobic carbon
    if resname == "CYS" and hbond and hydro:
        return (
            "Cysteine has a thiol group (-SH) that can donate or "
            "accept weak H-bonds via its sulfur, AND a short "
            "aliphatic carbon (-CH₂-) that is hydrophobic. "
            "Explain both contributions separately."
        )

    # Generic fallback for any other combination
    type_descriptions = {
        "HBDonor"    : "hydrogen bond donation (-NH or -OH group)",
        "HBAcceptor" : "hydrogen bond acceptance (electronegative atom)",
        "Hydrophobic": "hydrophobic contact (non-polar carbon atoms)",
        "PiStacking" : "π–π ring stacking (aromatic ring face-to-face)",
        "EdgeToFace" : "T-shaped ring interaction (aromatic ring edge)",
        "Cationic"   : "electrostatic attraction (positive charge)",
        "Anionic"    : "electrostatic attraction (negative charge)",
        "CationPi"   : "cation-π interaction",
        "PiCation"   : "π-cation interaction",
    }
    roles = " AND ".join(
        type_descriptions.get(t, t) for t in types
    )
    return (
        f"This residue simultaneously contributes via {roles}. "
        "Explain the specific atoms responsible for each role "
        "and why this dual contribution makes the residue "
        "especially important for binding."
    )


# ════════════════════════════════════════════════════════════
# PROMPT BUILDER
# Constructs the structured prompt sent to Gemma 4.
# Groups multi-role residues, ranks by importance,
# and embeds atom-level guidance for dual-role residues.
# ════════════════════════════════════════════════════════════

def build_prompt(result: dict) -> str:
    """
    Build a rich, structured prompt from the pipeline JSON.

    Args:
        result : dict loaded from *_results.json

    Returns:
        Full prompt string for Gemma 4
    """
    drug         = result.get("drug", "Unknown drug")
    protein      = result.get("protein", "Unknown protein")
    score        = result.get("docking_score")
    interactions = result.get("interactions", [])
    mode         = result.get("mode", "?")

    score_str = (
        f"{score:.2f} kcal/mol" if score
        else "crystallographic pose (experimental binding mode)"
    )

    # ── Group interactions by residue ───────────────────────
    residue_map = defaultdict(list)
    for ix in interactions:
        residue_map[ix["residue"]].append(ix["interaction_type"])

    # ── Sort by interaction priority ────────────────────────
    # H-bonds first (most important), hydrophobic last
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

    # ── Build interaction block ─────────────────────────────
    interaction_lines   = []
    multi_role_residues = []

    for residue, types in sorted_residues:
        types_str = " + ".join(types)
        if len(types) > 1:
            multi_role_residues.append((residue, types))
            interaction_lines.append(
                f"  {residue}: {types_str}"
                f"  ← MULTI-ROLE RESIDUE (explain each role)"
            )
        else:
            interaction_lines.append(f"  {residue}: {types_str}")

    interaction_block = "\n".join(interaction_lines)
    if not interaction_block:
        interaction_block = "  No interactions detected."

    # ── Build multi-role guidance ───────────────────────────
    multi_role_section = ""
    if multi_role_residues:
        lines = []
        for residue, types in multi_role_residues:
            # Extract residue name e.g. "MET" from "MET769.A"
            resname = residue.split(".")[0]
            import re
            m = re.match(r"([A-Z]+)\d+", resname)
            resname_code = m.group(1) if m else resname
            guidance = _get_multi_role_guidance(resname_code, types)
            lines.append(f"  {residue}: {guidance}")
        multi_role_section = (
            "\n\nIMPORTANT — Multi-role residues require special attention:\n"
            + "\n".join(lines)
            + "\n\nFor each of these, explicitly explain WHY the same "
            "residue contributes two different interaction types. "
            "Reference the specific atoms involved (e.g. sulfur vs "
            "carbon chain for MET). This is a key insight that "
            "non-experts need explained clearly."
        )

    # ── Assemble full prompt ────────────────────────────────
    prompt = f"""You are an expert computational chemist explaining molecular
docking results to a mixed audience — from biology students to
medicinal chemists. Your goal is to make the science clear,
specific, and genuinely insightful.

═══════════════════════════════════════════════════════
DOCKING RESULT TO EXPLAIN
═══════════════════════════════════════════════════════
Drug         : {drug}
Protein      : {protein}
Binding score: {score_str}
Pipeline mode: {mode}
Total interactions detected: {len(interactions)}

Interactions detected by ProLIF:
{interaction_block}
{multi_role_section}

═══════════════════════════════════════════════════════
YOUR TASK — Write a structured explanation
═══════════════════════════════════════════════════════
Write a clear, structured explanation of WHY {drug} binds
to {protein}. Your explanation must follow this structure:

1. OPENING (2-3 sentences)
   Summarise the binding in plain English. Mention the score
   and what it means (more negative = stronger binding).
   State the dominant interaction type driving the binding.

2. KEY INTERACTIONS (one paragraph per interaction type found)
   For each interaction type present:
   - Name the specific residue(s) involved
   - Explain the physical/chemical property causing it
   - Explain its contribution to binding strength
   - Use a simple analogy if it helps clarity

3. MULTI-ROLE RESIDUES (mandatory section if any are flagged above)
   For every residue flagged as MULTI-ROLE:
   - Explain each role separately in its own sentence
   - Name the specific atoms responsible for each role
   - Explain why the same residue can play two different roles
   - This is the most important section for chemical insight

4. BINDING POCKET CHARACTER
   Describe the overall character of the binding site
   (e.g. "a deep hydrophobic cave lined with aromatic residues")
   and explain why {drug}'s molecular structure is well-suited
   for it. Be specific about matching structural features.

5. CLINICAL RELEVANCE (2-3 sentences)
   Connect this binding to real-world drug action.
   Why does blocking/activating this site matter therapeutically?
   What disease does this treat and why does binding here help?

6. SANITY CHECK (Mode B / Mode C only)

    If Pipeline mode is Mode B or Mode C, first evaluate whether
    the drug–protein pair is biologically and chemically plausible.

    If the pairing is NOT meaningful (e.g. the protein is not a known
    target class for the drug, lacks an appropriate binding pocket,
    or belongs to a completely different functional system):

    - Clearly state that the docking result may not represent a
    biologically relevant binding event
    - Explain WHY in chemical/biological terms:
    • mismatch in protein function (e.g. enzyme vs lipid carrier)
    • absence of a well-defined small-molecule binding pocket
    • drug designed for a specific target class (e.g. kinase inhibitors)
        but applied to an unrelated protein
    - Give a concrete example using the current pair
    - Still describe the interactions briefly, but explicitly frame them
    as potentially incidental or artefactual rather than meaningful

    If the pairing IS plausible, proceed normally without this warning.

STYLE RULES:
- Explain every chemistry term the first time you use it
  e.g. "hydrogen bond (a weak attraction between...)"
- Never say "as shown above" — each section is self-contained
- Be specific: name residues, name atoms, name interaction types
- Avoid vague phrases like "interacts with" — say HOW it interacts
- Target length: 600-750 words total
- Use plain English first, then technical terms in parentheses
"""
    return prompt


# ════════════════════════════════════════════════════════════
# GEMMA 4 CALL
# ════════════════════════════════════════════════════════════

def explain_docking_result(
    result  : dict,
    model   : str  = "gemma4:e2b",
    verbose : bool = True,
) -> dict:
    """
    Send the interaction fingerprint to Gemma 4 and get back
    a plain-language explanation.

    Args:
        result  : dict loaded from *_results.json
        model   : Ollama model name
        verbose : if True, print the explanation to terminal

    Returns:
        dict with keys:
            explanation     — full text from Gemma 4
            prompt_used     — exact prompt sent
            model           — model used
            drug            — drug name
            protein         — protein name
            n_interactions  — number of interactions explained
            multi_role      — list of multi-role residues flagged
    """
    prompt = build_prompt(result)

    # Identify multi-role residues for the return dict
    residue_map = defaultdict(list)
    for ix in result.get("interactions", []):
        residue_map[ix["residue"]].append(ix["interaction_type"])
    multi_role = [
        r for r, t in residue_map.items() if len(t) > 1
    ]

    if verbose:
        print(f"\n  Sending to Gemma 4 ({model})...")
        print(f"  Drug         : {result.get('drug')}")
        print(f"  Protein      : {result.get('protein')}")
        print(f"  Interactions : {result.get('n_interactions', 0)}")
        if multi_role:
            print(f"  Multi-role   : {', '.join(multi_role)}")
        print(f"  {'─'*54}")

    # Call Gemma 4 via Ollama
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are DockExplain — an AI that explains "
                    "molecular docking results in clear, accurate, "
                    "and accessible language. You always name specific "
                    "residues and atoms. You never use vague language. "
                    "You explain every chemistry term you use. "
                    "You are especially attentive to residues that "
                    "contribute multiple interaction types — you "
                    "always explain each role and name the atoms "
                    "responsible."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature" : 0.3,    # low = factual, not creative
            "num_predict" : 4500,    # ~550 words
        }
    )

    explanation = response["message"]["content"]

    if verbose:
        print(f"\n{'═'*58}")
        print("  DockExplain — Gemma 4 Explanation")
        print(f"{'═'*58}")
        print(explanation)
        print(f"{'═'*58}")

    return {
        "explanation"   : explanation,
        "prompt_used"   : prompt,
        "model"         : model,
        "drug"          : result.get("drug"),
        "protein"       : result.get("protein"),
        "n_interactions": result.get("n_interactions", 0),
        "multi_role"    : multi_role,
    }


# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════

def save_explanation(explained: dict, out_path: str):
    """Save explanation dict to JSON."""
    with open(out_path, "w") as f:
        json.dump(explained, f, indent=2)
    print(f"  ✓ Explanation JSON saved: {out_path}")

    # Also save as plain text for easy reading
    txt_path = out_path.replace(".json", ".txt")
    with open(txt_path, "w") as f:
        drug    = explained.get("drug", "")
        protein = explained.get("protein", "")
        n       = explained.get("n_interactions", 0)
        model   = explained.get("model", "")
        f.write(f"DockExplain — {drug} binding to {protein}\n")
        f.write(f"Model: {model} | Interactions explained: {n}\n")
        f.write("=" * 60 + "\n\n")
        f.write(explained.get("explanation", ""))
        multi = explained.get("multi_role", [])
        if multi:
            f.write(f"\n\n[Multi-role residues: {', '.join(multi)}]")
    print(f"  ✓ Explanation TXT saved: {txt_path}")


# ════════════════════════════════════════════════════════════
# CLI — run standalone
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DockExplain — Gemma 4 explanation"
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
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"Error: {args.json} not found")
        print("Run dockexplain_pipeline.py first")
        sys.exit(1)

    with open(args.json) as f:
        result = json.load(f)

    print(f"Loaded: {args.json}")
    print(f"Drug   : {result.get('drug')}")
    print(f"Protein: {result.get('protein')}")
    print(f"Interactions: {result.get('n_interactions', 0)}")

    # Generate explanation
    explained = explain_docking_result(result, model=args.model)

    # Save alongside the results JSON
    out_dir   = os.path.dirname(args.json)
    safe_drug = result.get("drug","drug").lower().replace(" ","_")
    out_path  = os.path.join(out_dir, f"{safe_drug}_explanation.json")
    save_explanation(explained, out_path)