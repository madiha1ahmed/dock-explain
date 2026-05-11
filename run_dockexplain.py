# run_dockexplain.py
# ─────────────────────────────────────────────────────────────
# DockExplain — unified end-to-end workflow
#
# Runs all components in sequence:
#   1. dockexplain_pipeline  → docking + ProLIF
#   2. validation_suite      → 5-tier structural QC
#   3. pymol_visualize       → PyMOL script + PNG
#   4. web_search_enricher   → drug/protein + PubMed literature
#   5. gemma_explainer       → plain-language explanation + bibliography
#   6. report_generator      → PDF report
#
# Usage:
#   python run_dockexplain.py --pdb 1M17
#   python run_dockexplain.py --pdb 1M17 --drug Erlotinib
#   python run_dockexplain.py --pdb 1M17 --drug Erlotinib \
#       --no-web-search           # skip web enrichment
#       --skip-pymol              # if PyMOL not installed
#       --skip-literature         # if offline
#       --exhaustiveness 32       # better docking
# ─────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
import time
import traceback
import warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════
# TOP-LEVEL IMPORTS
# ════════════════════════════════════════════════════════════

def import_components():
    """
    Import all DockExplain modules.
    Returns dict of imported callables; missing modules are
    reported but do not abort — they are skipped gracefully.
    """
    components = {}
    errors     = []

    # ── Core pipeline ──────────────────────────────────────
    try:
        from dockexplain_pipeline import run_pipeline
        components["run_pipeline"] = run_pipeline
    except ImportError as e:
        errors.append(f"dockexplain_pipeline: {e}")

    
    # ── PyMOL visualization ────────────────────────────────
    try:
        from pymol_visualize import visualize
        components["visualize"] = visualize
    except ImportError as e:
        errors.append(f"pymol_visualize: {e}")

    # ── Web search enricher ────────────────────────────────
    try:
        from web_search_enricher import (
            enrich_docking_context,
            format_context_for_prompt,
        )
        components["enrich_docking_context"]  = enrich_docking_context
        components["format_context_for_prompt"] = format_context_for_prompt
    except ImportError as e:
        errors.append(f"web_search_enricher: {e}")

    # ── Gemma 4 explanation ────────────────────────────────
    # Uses the updated gemma_explainer.py that accepts lit_ctx
    try:
        from gemma_explainer import explain_docking_result, save_explanation
        components["explain"]          = explain_docking_result
        components["save_explanation"] = save_explanation
    except ImportError as e:
        errors.append(f"gemma_explainer: {e}")

    # ── PDF report generator ───────────────────────────────
    try:
        from report_generator import generate_report
        components["generate_report"] = generate_report
    except ImportError as e:
        errors.append(f"report_generator: {e}")

    if errors:
        print("\n⚠  Some components could not be imported:")
        for err in errors:
            print(f"   • {err}")
        print("\n   Missing components will be skipped.")
        print("   Check that all .py files are in the same directory.\n")

    return components


# ════════════════════════════════════════════════════════════
# STEP RUNNER
# ════════════════════════════════════════════════════════════

class StepResult:
    """Result of a single pipeline step."""
    def __init__(self, name):
        self.name    = name
        self.status  = None   # "ok" | "warn" | "skip" | "fail"
        self.message = ""
        self.data    = None
        self.elapsed = 0.0

    def ok(self, data=None, msg=""):
        self.status  = "ok"
        self.data    = data
        self.message = msg
        return self

    def warn(self, msg):
        self.status  = "warn"
        self.message = msg
        return self

    def skip(self, msg):
        self.status  = "skip"
        self.message = msg
        return self

    def fail(self, msg):
        self.status  = "fail"
        self.message = msg
        return self


def run_step(name: str, fn, *args, **kwargs) -> StepResult:
    """
    Run a pipeline step with timing and error capture.
    Returns a StepResult regardless of outcome.
    """
    result = StepResult(name)
    t0 = time.time()
    try:
        data = fn(*args, **kwargs)
        result.elapsed = time.time() - t0
        return result.ok(data=data)
    except Exception as e:
        result.elapsed = time.time() - t0
        tb = traceback.format_exc()
        return result.fail(f"{type(e).__name__}: {e}\n{tb}")


# ════════════════════════════════════════════════════════════
# FILE LOCATORS
# ════════════════════════════════════════════════════════════

def locate_clean_pdb(out_dir: str, prot_base: str) -> str | None:
    candidates = [
        os.path.join(out_dir, f"{prot_base}_clean.pdb"),
        os.path.join(out_dir, f"{prot_base}.pdb"),
    ]
    if os.path.isdir(out_dir):
        for fname in sorted(os.listdir(out_dir)):
            if fname.endswith("_clean.pdb"):
                candidates.append(os.path.join(out_dir, fname))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def locate_best_pose_sdf(out_dir: str, safe_drug: str) -> str | None:
    for path in [
        os.path.join(out_dir, f"{safe_drug}_best_pose.sdf"),
        os.path.join(out_dir, "best_pose.sdf"),
    ]:
        if os.path.exists(path):
            return path
    return None


def locate_results_json(out_dir: str, safe_drug: str) -> str | None:
    for path in [
        os.path.join(out_dir, f"{safe_drug}_results.json"),
        os.path.join(out_dir, "results.json"),
    ]:
        if os.path.exists(path):
            return path
    return None


# ════════════════════════════════════════════════════════════
# MAIN WORKFLOW
# ════════════════════════════════════════════════════════════

def run_dockexplain(
    protein_pdb      : str,
    drug_name        : str,
    drug_smiles      : str | None = None,
    pdb_id           : str | None = None,    # 4-char PDB ID for RCSB lookup
    out_dir          : str        = "../data/results/",
    exhaustiveness   : int        = 8,
    ollama_model     : str        = "gemma4:e2b",
    n_papers         : int        = 10,
    skip_pymol       : bool       = False,
    skip_web_search  : bool       = False,   # ← new
    skip_explanation : bool       = False,
    pymol_headless   : bool       = False,
    selected_ligand_code: str | None = None
) -> dict:
    """
    Full DockExplain workflow for any protein–drug pair.

    Step order:
      1. Docking + ProLIF  (dockexplain_pipeline)
      2. Structural QC     (validation_suite)
      3. PyMOL             (pymol_visualize)
      4. PubMed RAG        (literature_validator)
      5. Web enrichment    (web_search_enricher)   ← NEW
      6. Gemma explanation (gemma_explainer)
      7. PDF report        (report_generator)

    The web enrichment result (LiteratureContext) is passed
    directly into Gemma — so the data is fetched exactly once.

    Args:
        protein_pdb     : raw PDB file path
        drug_name       : human-readable name (e.g. "Erlotinib")
        drug_smiles     : SMILES string (optional)
        out_dir         : output directory
        exhaustiveness  : AutoDock Vina search depth
        ollama_model    : Ollama model name
        n_papers        : PubMed papers for literature validation
        skip_pymol      : skip PyMOL (if not installed)
        skip_literature : skip PubMed RAG (if offline)
        skip_web_search : skip web enrichment for Gemma
        skip_explanation: skip Gemma explanation
        pymol_headless  : headless render only (no GUI)

    Returns:
        dict mapping step name → StepResult
    """
    safe_drug = drug_name.lower().replace(" ", "_")
    prot_base = os.path.splitext(os.path.basename(protein_pdb))[0]
    os.makedirs(out_dir, exist_ok=True)

    components = import_components()
    steps      = {}
    t_total    = time.time()

    print("\n" + "█" * 66)
    print("  DockExplain — Full Workflow")
    print(f"  Protein : {os.path.basename(protein_pdb)}")
    print(f"  Drug    : {drug_name}")
    print(f"  Mode    : {'SMILES provided' if drug_smiles else 'auto-detect from PDB'}")
    print(f"  Output  : {out_dir}")
    print("█" * 66)

    # Shared state — populated across steps and passed forward
    json_path  = None
    clean_pdb  = None
    best_sdf   = None
    lit_ctx    = None   # LiteratureContext from Step 5, passed to Step 6


    # ════════════════════════════════════════════════════════
    # STEP 1 — Docking pipeline
    # ════════════════════════════════════════════════════════
    print(f"\n{'─'*66}")
    print(f"  STEP 1/7 — Docking pipeline")
    print(f"{'─'*66}")

    if "run_pipeline" not in components:
        steps["pipeline"] = StepResult("pipeline").skip(
            "dockexplain_pipeline.py not found"
        )
        print("  ⚠ Skipped — dockexplain_pipeline.py not importable")
    else:
        result = run_step(
            "pipeline",
            components["run_pipeline"],
            raw_protein_pdb = protein_pdb,
            drug_name       = drug_name,
            drug_smiles     = drug_smiles,
            out_dir         = out_dir,
            exhaustiveness  = exhaustiveness,
            selected_ligand_code = selected_ligand_code,
        )
        steps["pipeline"] = result

        if result.status == "fail":
            print(f"\n  ✗ Pipeline failed:\n{result.message}")
            print("  Cannot continue without pipeline output.")
            _print_summary(steps, time.time() - t_total)
            return steps

        pipeline_result = result.data
        print(f"\n  ✓ Pipeline complete in {result.elapsed:.1f}s")
        print(f"    Mode         : {pipeline_result.get('mode','?')}")
        print(f"    Docking score: {pipeline_result.get('docking_score') or 'crystal pose'}")
        print(f"    Interactions : {pipeline_result.get('n_interactions', 0)}")

    # Locate files written by Step 1
    clean_pdb = locate_clean_pdb(out_dir, prot_base)
    best_sdf  = locate_best_pose_sdf(out_dir, safe_drug)
    json_path = locate_results_json(out_dir, safe_drug)


    # ════════════════════════════════════════════════════════
    # STEP 2 — Structural validation removed
    # ════════════════════════════════════════════════════════
    steps["validation"] = StepResult("validation").skip(
        "Validation step removed for deployment"
    )


    # ════════════════════════════════════════════════════════
    # STEP 3 — PyMOL visualization
    # ════════════════════════════════════════════════════════
    print(f"\n{'─'*66}")
    print(f"  STEP 3/7 — PyMOL visualization")
    print(f"{'─'*66}")

    if skip_pymol:
        steps["pymol"] = StepResult("pymol").skip("--skip-pymol flag set")
        print("  — Skipped (--skip-pymol)")
    elif "visualize" not in components:
        steps["pymol"] = StepResult("pymol").skip(
            "pymol_visualize.py not found"
        )
        print("  ⚠ Skipped — pymol_visualize.py not importable")
    elif not json_path:
        steps["pymol"] = StepResult("pymol").skip("No results JSON")
        print("  ⚠ Skipped — results JSON not found")
    else:
        result = run_step(
            "pymol",
            _run_pymol_safely,
            visualize_fn = components["visualize"],
            json_path    = json_path,
            clean_pdb    = clean_pdb,
            interactive  = not pymol_headless,
        )
        steps["pymol"] = result

        if result.status == "ok":
            print(f"  ✓ PyMOL complete in {result.elapsed:.1f}s")
            print(f"    Script : {result.data}")
        else:
            print(f"  ⚠ PyMOL error: {result.message[:200]}")


    # ════════════════════════════════════════════════════════
    # STEP 5 — Web search enrichment for Gemma
    #
    # Fetches: PubChem (drug), UniProt (protein), ChEMBL (MoA),
    # DuckDuckGo (mechanism, biology, interaction, clinical).
    #
    # The returned LiteratureContext is passed directly into
    # Step 6 so the data is only fetched once.
    # ════════════════════════════════════════════════════════
    print(f"\n{'─'*66}")
    print(f"  STEP 4/6 — Web search enrichment")
    print(f"{'─'*66}")

    if skip_web_search:
        steps["web_search"] = StepResult("web_search").skip(
            "--no-web-search flag set"
        )
        print("  — Skipped (--no-web-search)")
        print("    Gemma will rely on training knowledge only.")
    elif "enrich_docking_context" not in components:
        steps["web_search"] = StepResult("web_search").skip(
            "web_search_enricher.py not found or duckduckgo-search "
            "not installed.\n"
            "  Install: pip install duckduckgo-search requests beautifulsoup4"
        )
        print("  ⚠ Skipped — web_search_enricher.py not importable")
        print("    Install: pip install duckduckgo-search requests beautifulsoup4")
    else:
        # Load the full results dict — passed to enricher so it can
        # extract gene symbols from pipeline metadata fields
        _pipeline_result = None
        if json_path and os.path.exists(json_path):
            with open(json_path) as f:
                _tmp = json.load(f)
            _drug_name       = _tmp.get("drug",    drug_name)
            _protein_name    = _tmp.get("protein", "Unknown protein")
            _pipeline_result = _tmp
        else:
            _drug_name    = drug_name
            _protein_name = "Unknown protein"

        print(f"  Drug    : {_drug_name}")
        print(f"  Protein : {_protein_name}")
        if pdb_id:
            print(f"  PDB ID  : {pdb_id} (RCSB gene lookup enabled)")
        print(f"  Sources : PubChem · UniProt · ChEMBL · DuckDuckGo")

        # Pass drug_smiles so proprietary/research drug codes that
        # PubChem can't resolve by name can be looked up by structure
        _drug_smiles = (
            _pipeline_result.get("drug_smiles", "") if _pipeline_result
            else drug_smiles or ""
        )

        result = run_step(
            "web_search",
            components["enrich_docking_context"],
            drug        = _drug_name,
            protein     = _protein_name,
            pdb_id      = pdb_id,
            drug_smiles = _drug_smiles,
            result_json = _pipeline_result,
            verbose     = True,
        )
        steps["web_search"] = result

        if result.status == "ok":
            lit_ctx = result.data    # LiteratureContext, passed to Step 6
            n_sources = len(lit_ctx.sources)
            n_warnings = len(lit_ctx.warnings)
            print(f"\n  ✓ Web enrichment complete in {result.elapsed:.1f}s")
            print(f"    Sources gathered : {n_sources}")
            print(f"    PubMed papers    : {len(lit_ctx.papers)}")
            # Show resolved identities clearly so user knows what Gemma receives
            _resolved_drug = getattr(lit_ctx, "drug_display_name", "") or _drug_name
            if _resolved_drug != _drug_name:
                print(f"    Drug name        : '{_drug_name}'")
                print(f"                       → '{_resolved_drug}'  ← Gemma will use this name")
            else:
                print(f"    Drug name        : {_resolved_drug}")
            _resolved_prot = lit_ctx.protein_display_name or lit_ctx.gene_symbol or _protein_name
            if _resolved_prot != _protein_name:
                print(f"    Protein          : '{_protein_name}'")
                print(f"                       → '{_resolved_prot}'  ← Gemma will use this name")
            else:
                print(f"    Gene symbol      : {lit_ctx.gene_symbol or 'unresolved'}")
            print(f"    Drug summary     : "
                  f"{'yes' if lit_ctx.drug_summary else 'not found'}")
            print(f"    Protein summary  : "
                  f"{'yes' if lit_ctx.protein_summary else 'not found'}")
            print(f"    Interaction data : "
                  f"{'yes' if lit_ctx.interaction_evidence else 'not found'}")
            print(f"    Clinical context : "
                  f"{'yes' if lit_ctx.clinical_context else 'not found'}")
            if n_warnings:
                for w in lit_ctx.warnings:
                    print(f"    ⚠ {w}")

            # Save enrichment context alongside other outputs
            if json_path:
                enrich_path = json_path.replace(
                    "_results.json", "_web_enrichment.json"
                )
                with open(enrich_path, "w") as f:
                    json.dump({
                        "drug"                : _drug_name,
                        "protein"             : _protein_name,
                        "drug_summary"        : lit_ctx.drug_summary,
                        "protein_summary"     : lit_ctx.protein_summary,
                        "interaction_evidence": lit_ctx.interaction_evidence,
                        "clinical_context"    : lit_ctx.clinical_context,
                        "warnings"            : lit_ctx.warnings,
                        "sources"             : lit_ctx.sources,
                    }, f, indent=2)
                print(f"    Saved: {enrich_path}")
        else:
            # Non-fatal — Gemma still runs, just without enrichment
            print(f"  ⚠ Web enrichment failed: {result.message[:200]}")
            print("    Gemma will proceed with training knowledge only.")
            lit_ctx = None


    # ════════════════════════════════════════════════════════
    # STEP 6 — Gemma 4 plain-language explanation
    #
    # Receives lit_ctx from Step 5 directly — no re-fetch.
    # run_web_search=False prevents gemma_explainer from
    # running its own search since we already have the data.
    # ════════════════════════════════════════════════════════
    print(f"\n{'─'*66}")
    print(f"  STEP 5/6 — Gemma 4 explanation")
    print(f"{'─'*66}")

    if skip_explanation:
        steps["explanation"] = StepResult("explanation").skip(
            "--skip-explanation flag set"
        )
        print("  — Skipped (--skip-explanation)")
    elif "explain" not in components:
        steps["explanation"] = StepResult("explanation").skip(
            "gemma_explainer.py not found"
        )
        print("  ⚠ Skipped — gemma_explainer.py not importable")
    elif not json_path:
        steps["explanation"] = StepResult("explanation").skip(
            "No results JSON"
        )
        print("  ⚠ Skipped — results JSON not found")
    else:
        with open(json_path) as f:
            pipeline_result = json.load(f)

        # lit_ctx is either a LiteratureContext from Step 5,
        # or None if web search was skipped / failed.
        # run_web_search=False prevents a duplicate search.
        result = run_step(
            "explanation",
            components["explain"],
            result         = pipeline_result,
            model          = ollama_model,
            verbose        = True,
            run_web_search = False,   # data already in lit_ctx
            lit_ctx        = lit_ctx,
        )
        steps["explanation"] = result

        if result.status == "ok":
            explained = result.data

            out_path = os.path.join(
                out_dir, f"{safe_drug}_explanation.json"
            )
            save_fn = components.get("save_explanation")
            if save_fn:
                save_fn(explained, out_path)

            n_sources = len(explained.get("sources", []))
            print(f"\n  ✓ Explanation complete in {result.elapsed:.1f}s")
            print(f"    Literature sources used: {n_sources}")
            print(f"    Saved: {out_path}")
        else:
            print(f"  ⚠ Explanation error: {result.message[:200]}")


    # ════════════════════════════════════════════════════════
    # STEP 7 — PDF report
    # ════════════════════════════════════════════════════════
    print(f"\n{'─'*66}")
    print(f"  STEP 6/6 — PDF report")
    print(f"{'─'*66}")

    if "generate_report" not in components:
        steps["pdf"] = StepResult("pdf").skip(
            "explanation_pdf_generator.py not found or reportlab not installed.\n"
            "  Install: pip install reportlab"
        )
        print("  ⚠ Skipped — run: pip install reportlab")
    elif not json_path:
        steps["pdf"] = StepResult("pdf").skip("No results JSON")
        print("  ⚠ Skipped — results JSON not found")
    else:
        # The PDF generator needs the explanation JSON (contains Gemma text
        # + bibliography). Falls back to results JSON if not found.
        explanation_json = os.path.join(
            out_dir, f"{safe_drug}_explanation.json"
        )
        pdf_input = explanation_json if os.path.exists(explanation_json) else json_path

        result = run_step(
            "pdf",
            components["generate_report"],
            pdf_input,
        )
        steps["pdf"] = result

        if result.status == "ok":
            print(f"  ✓ PDF complete in {result.elapsed:.1f}s")
            print(f"    Saved: {result.data}")
        else:
            print(f"  ⚠ PDF error: {result.message[:200]}")


    # ════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════
    _print_summary(steps, time.time() - t_total)
    _save_run_manifest(steps, out_dir, safe_drug,
                       protein_pdb, drug_name, drug_smiles)
    return steps


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def _run_pymol_safely(visualize_fn, json_path, clean_pdb, interactive):
    """
    Ensure clean_pdb is present in the results directory before
    calling visualize(), which scans the directory for *_clean.pdb.
    """
    import shutil
    out_dir  = os.path.dirname(json_path)
    dest_pdb = None

    if clean_pdb and os.path.dirname(
        os.path.abspath(clean_pdb)
    ) != os.path.abspath(out_dir):
        dest_pdb = os.path.join(out_dir, os.path.basename(clean_pdb))
        if not os.path.exists(dest_pdb):
            shutil.copy2(clean_pdb, dest_pdb)

    return visualize_fn(json_path, interactive=interactive)


def _print_summary(steps: dict, elapsed: float):
    """Print a colour-coded summary of all step outcomes."""
    ICONS = {"ok": "✓", "warn": "⚠", "skip": "—", "fail": "✗"}
    COLS  = {
        "ok"  : "\033[92m",
        "warn": "\033[93m",
        "fail": "\033[91m",
        "skip": "\033[90m",
    }
    RESET = "\033[0m"

    STEP_LABELS = {
        "pipeline"   : "Docking pipeline    ",
        "validation" : "Structural QC       ",
        "pymol"      : "PyMOL visualization ",
        "web_search" : "Web enrichment      ",   # ← new
        "explanation": "Gemma 4 explanation ",
        "pdf"        : "PDF report          ",
    }

    print(f"\n{'█'*66}")
    print("  DockExplain — Run Summary")
    print(f"{'─'*66}")

    any_fail = False
    all_ok   = True

    for key, label in STEP_LABELS.items():
        if key not in steps:
            print(f"  — {label} : not run")
            continue
        r     = steps[key]
        icon  = ICONS.get(r.status, "?")
        col   = COLS.get(r.status, "")
        t_str = f"{r.elapsed:.1f}s" if r.elapsed > 0 else ""
        msg   = r.message[:55] if r.status != "ok" else ""
        print(f"  {col}{icon}{RESET} {label} : "
              f"{col}{r.status.upper():<5}{RESET} "
              f"{t_str:<8} {msg}")
        if r.status == "fail":
            any_fail = True
        if r.status not in ("ok", "skip"):
            all_ok = False

    print(f"{'─'*66}")
    print(f"  Total time: {elapsed:.1f}s")
    if all_ok:
        print(f"  \033[92mAll steps completed successfully.\033[0m")
    elif any_fail:
        print(f"  \033[91mSome steps failed — check output above.\033[0m")
    else:
        print(f"  \033[93mCompleted with warnings or skipped steps.\033[0m")
    print(f"{'█'*66}\n")


def _save_run_manifest(steps: dict, out_dir: str, safe_drug: str,
                       protein_pdb: str, drug_name: str,
                       drug_smiles: str | None):
    """Save a JSON manifest of the entire run for auditing."""
    import datetime
    manifest = {
        "timestamp"  : datetime.datetime.now().isoformat(),
        "drug"       : drug_name,
        "drug_smiles": drug_smiles,
        "protein_pdb": protein_pdb,
        "out_dir"    : out_dir,
        "steps"      : {
            k: {
                "status" : v.status,
                "elapsed": round(v.elapsed, 2),
                "message": v.message[:300] if v.status != "ok" else "",
            }
            for k, v in steps.items()
        }
    }
    path = os.path.join(out_dir, f"{safe_drug}_run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Run manifest: {path}")


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="DockExplain — drug-protein docking analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive (recommended)
  python run_dockexplain.py --pdb 1M17

  # Non-interactive
  python run_dockexplain.py --pdb 1M17 --drug Erlotinib

  # With explicit SMILES
  python run_dockexplain.py --pdb 1M17 --drug Ibuprofen \\
      --smiles "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

  # Offline / no web search
  python run_dockexplain.py --pdb 1M17 --drug Erlotinib \\
      --no-web-search --skip-literature

  # Fast run (no visualization, no PDF)
  python run_dockexplain.py --pdb 1M17 --drug Erlotinib \\
      --skip-pymol --no-web-search --skip-literature
        """
    )

    parser.add_argument("--pdb",    default=None, metavar="ID",
        help="RCSB PDB ID (e.g. 1M17). Downloaded automatically.")
    parser.add_argument("--drug",   default=None, metavar="NAME",
        help="Drug name (prompted interactively if omitted)")
    parser.add_argument("--smiles", default=None, metavar="SMILES",
        help="Drug SMILES (auto-fetched for crystal drugs)")
    parser.add_argument("--out",    default=None, metavar="DIR",
        help="Output directory (default: ../data/results/<drug>/)")
    parser.add_argument("--exhaustiveness", type=int, default=8,
        metavar="N", help="Vina exhaustiveness (default: 8)")
    parser.add_argument("--model",  default="gemma4:e2b", metavar="MODEL",
        help="Ollama model (default: gemma4:e2b)")
    parser.add_argument("--papers", type=int, default=10, metavar="N",
        help="PubMed papers for RAG validation (default: 10)")

    # ── Skip flags ─────────────────────────────────────────
    parser.add_argument("--skip-pymol",
        action="store_true", help="Skip PyMOL visualization")
    parser.add_argument("--no-web-search",
        action="store_true",
        help="Skip web enrichment for Gemma (use training knowledge only)")
    parser.add_argument("--skip-explanation",
        action="store_true", help="Skip Gemma explanation")
    parser.add_argument("--headless",
        action="store_true", help="PyMOL headless render (no GUI)")

    args = parser.parse_args()

    # ── Resolve inputs ─────────────────────────────────────
    if args.pdb:
        from input_handler import (
            download_pdb, detect_ligands, fetch_ligand_info,
            resolve_drug_input,
        )

        pdb_id   = args.pdb.strip().upper()
        pdb_path = download_pdb(pdb_id, out_dir="../data")

        if args.smiles:
            drug_smiles = args.smiles
            drug_name   = args.drug or pdb_id
        elif args.drug:
            resolved    = resolve_drug_input(args.drug)
            drug_smiles = resolved["smiles"]
            drug_name   = args.drug
            if not drug_smiles:
                ligands = detect_ligands(pdb_path)
                if ligands:
                    info        = fetch_ligand_info(ligands[0]["code"])
                    drug_smiles = info.get("smiles", "")
        else:
            from input_handler import run_input_handler
            job         = run_input_handler(pdb_data_dir="../data")
            pdb_path    = job["pdb_path"]
            pdb_id      = job.get("pdb_id", pdb_id)  # prefer job value
            drug_name   = job["drug_name"]
            drug_smiles = job["drug_smiles"]
    else:
        from input_handler import run_input_handler
        job         = run_input_handler(pdb_data_dir="../data")
        pdb_path    = job["pdb_path"]
        pdb_id      = job.get("pdb_id", "")      # ← always set
        drug_name   = job["drug_name"]
        drug_smiles = job["drug_smiles"]

    out_dir = args.out or os.path.join(
        "../data/results", drug_name.lower().replace(" ", "_")
    )

    run_dockexplain(
        protein_pdb      = pdb_path,
        drug_name        = drug_name,
        drug_smiles      = drug_smiles or None,
        pdb_id           = pdb_id if pdb_id else None,
        out_dir          = out_dir,
        exhaustiveness   = args.exhaustiveness,
        ollama_model     = args.model,
        n_papers         = args.papers,
        skip_pymol       = args.skip_pymol,
        skip_web_search  = args.no_web_search,
        skip_explanation = args.skip_explanation,
        pymol_headless   = args.headless,
    )
