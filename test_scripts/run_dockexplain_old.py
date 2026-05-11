# run_dockexplain.py
# ─────────────────────────────────────────────────────────────
# DockExplain — unified end-to-end workflow
#
# Runs all components in sequence:
#   1. dockexplain_pipeline  → docking + ProLIF
#   2. validation_suite      → 5-tier structural QC
#   3. pymol_visualize       → PyMOL script + PNG
#   4. literature_validator  → PubMed RAG validation
#   5. gemma_explainer       → plain-language explanation
#
# Usage:
#   python run_dockexplain.py \
#       --protein ../data/1M17.pdb \
#       --drug    Erlotinib
#
#   python run_dockexplain.py \
#       --protein ../data/1M17.pdb \
#       --smiles  "CC(C)Cc1ccc(cc1)C(C)C(=O)O" \
#       --drug    Ibuprofen
#
#   python run_dockexplain.py \
#       --protein ../data/my_apo_protein.pdb \
#       --smiles  "CC(=O)Oc1ccccc1C(=O)O" \
#       --drug    Aspirin \
#       --skip-pymol          # if PyMOL not installed
#       --skip-literature     # if offline
#       --exhaustiveness 32   # for better docking
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
# TOP-LEVEL IMPORTS — all components loaded once here
# ════════════════════════════════════════════════════════════

def import_components():
    """
    Import all DockExplain modules.
    Returns dict of imported functions, or raises with
    a clear message if any dependency is missing.
    """
    components = {}
    errors     = []

    # ── Core pipeline ──────────────────────────────────────
    try:
        from dockexplain_pipeline import run_pipeline
        components["run_pipeline"] = run_pipeline
    except ImportError as e:
        errors.append(f"dockexplain_pipeline: {e}")

    # ── Validation suite ───────────────────────────────────
    try:
        from validation_suite import (
            run_all_checks, print_report, save_report
        )
        components["run_all_checks"] = run_all_checks
        components["print_report"]   = print_report
        components["save_report"]    = save_report
    except ImportError as e:
        errors.append(f"validation_suite: {e}")

    # ── PyMOL visualization ────────────────────────────────
    try:
        from pymol_visualize import visualize
        components["visualize"] = visualize
    except ImportError as e:
        errors.append(f"pymol_visualize: {e}")

    # ── Literature validation ──────────────────────────────
    try:
        from literature_validator import run_validation
        components["run_validation"] = run_validation
    except ImportError as e:
        errors.append(f"literature_validator: {e}")

    # ── Gemma 4 explanation ────────────────────────────────
    try:
        from python_scripts.gemma_explainer_old import explain_docking_result, save_explanation
        components["explain"]          = explain_docking_result
        components["save_explanation"] = save_explanation
    except ImportError as e:
        errors.append(f"gemma_explainer: {e}")

    if errors:
        print("\n⚠  Some components could not be imported:")
        for err in errors:
            print(f"   • {err}")
        print("\n   Missing components will be skipped.")
        print("   Check that all .py files are in the same directory.\n")

    return components


# ════════════════════════════════════════════════════════════
# STEP RUNNER — uniform error handling for every step
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
# LOCATE FILES — robust file finding for cross-step handoffs
# ════════════════════════════════════════════════════════════

def locate_clean_pdb(out_dir: str, prot_base: str) -> str | None:
    """
    Find the cleaned protein PDB in the output directory.
    Tries multiple naming patterns to be robust to
    different input filenames.
    """
    candidates = [
        os.path.join(out_dir, f"{prot_base}_clean.pdb"),
        os.path.join(out_dir, f"{prot_base}.pdb"),
    ]
    # Also scan directory for any *_clean.pdb
    if os.path.isdir(out_dir):
        for fname in sorted(os.listdir(out_dir)):
            if fname.endswith("_clean.pdb"):
                candidates.append(os.path.join(out_dir, fname))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def locate_best_pose_sdf(out_dir: str, safe_drug: str) -> str | None:
    """Find the best pose SDF file."""
    candidates = [
        os.path.join(out_dir, f"{safe_drug}_best_pose.sdf"),
        os.path.join(out_dir, "best_pose.sdf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def locate_results_json(out_dir: str, safe_drug: str) -> str | None:
    """Find the pipeline results JSON."""
    candidates = [
        os.path.join(out_dir, f"{safe_drug}_results.json"),
        os.path.join(out_dir, "results.json"),
    ]
    for path in candidates:
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
    out_dir          : str        = "../data/results/",
    exhaustiveness   : int        = 8,
    ollama_model     : str        = "gemma4:e2b",
    n_papers         : int        = 10,
    skip_pymol       : bool       = False,
    skip_literature  : bool       = False,
    skip_explanation : bool       = False,
    pymol_headless   : bool       = False,
) -> dict:
    """
    Full DockExplain workflow for any protein-drug pair.

    Args:
        protein_pdb      : raw PDB file (straight from RCSB)
        drug_name        : human-readable name (e.g. "Erlotinib")
        drug_smiles      : SMILES string (optional for co-crystal PDBs)
        out_dir          : output directory
        exhaustiveness   : AutoDock Vina search depth (8=fast, 32=thorough)
        ollama_model     : Gemma 4 model (gemma4:e2b for 8GB RAM)
        n_papers         : number of PubMed papers for literature validation
        skip_pymol       : skip PyMOL visualization (if not installed)
        skip_literature  : skip PubMed search (if offline)
        skip_explanation : skip Gemma 4 explanation (if Ollama not running)
        pymol_headless   : headless render only (no GUI)

    Returns:
        dict with all step results
    """
    # ── Setup ──────────────────────────────────────────────
    safe_drug = drug_name.lower().replace(" ", "_")
    prot_base = os.path.splitext(
        os.path.basename(protein_pdb)
    )[0]
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


    # ════════════════════════════════════════════════════════
    # STEP 1: Docking pipeline (runs modes A/B/C automatically)
    # ════════════════════════════════════════════════════════
    print(f"\n{'─'*66}")
    print(f"  STEP 1/5 — Docking pipeline")
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
            raw_protein_pdb  = protein_pdb,
            drug_name        = drug_name,
            drug_smiles      = drug_smiles,
            out_dir          = out_dir,
            exhaustiveness   = exhaustiveness,
        )
        steps["pipeline"] = result

        if result.status == "fail":
            print(f"\n  ✗ Pipeline failed:\n{result.message}")
            print("  Cannot continue without pipeline output.")
            _print_summary(steps, time.time()-t_total)
            return steps

        pipeline_result = result.data
        print(f"\n  ✓ Pipeline complete in {result.elapsed:.1f}s")
        print(f"    Mode         : {pipeline_result.get('mode','?')}")
        print(f"    Docking score: "
              f"{pipeline_result.get('docking_score') or 'crystal pose'}")
        print(f"    Interactions : {pipeline_result.get('n_interactions',0)}")


    # ════════════════════════════════════════════════════════
    # STEP 2: Structural validation (5-tier)
    # ════════════════════════════════════════════════════════
    print(f"\n{'─'*66}")
    print(f"  STEP 2/5 — Structural validation (5 tiers)")
    print(f"{'─'*66}")

    # Locate the files produced by step 1
    clean_pdb  = locate_clean_pdb(out_dir, prot_base)
    best_sdf   = locate_best_pose_sdf(out_dir, safe_drug)
    json_path  = locate_results_json(out_dir, safe_drug)

    if not all([clean_pdb, best_sdf, json_path]):
        missing = [
            name for name, val in [
                ("clean_pdb", clean_pdb),
                ("best_pose.sdf", best_sdf),
                ("results.json", json_path),
            ] if not val
        ]
        steps["validation"] = StepResult("validation").skip(
            f"Missing files from step 1: {missing}"
        )
        print(f"  ⚠ Skipped — files not found: {missing}")
    elif "run_all_checks" not in components:
        steps["validation"] = StepResult("validation").skip(
            "validation_suite.py not found"
        )
        print("  ⚠ Skipped — validation_suite.py not importable")
    else:
        result = run_step(
            "validation",
            components["run_all_checks"],
            raw_pdb     = protein_pdb,
            clean_pdb   = clean_pdb,
            ligand_sdf  = best_sdf,
            result_json = json_path,
        )
        steps["validation"] = result

        if result.status == "ok":
            report = result.data
            components["print_report"](report)
            components["save_report"](report, out_dir)
            conf  = report.confidence_score()
            overall = report.overall_status()
            print(f"\n  ✓ Validation complete in {result.elapsed:.1f}s")
            print(f"    Confidence score : {conf}%")
            print(f"    Overall status   : {overall}")
            if overall == "FAIL":
                print("  ⚠ Validation FAILED — results may be unreliable.")
                print("    Review the report before proceeding.")
        else:
            print(f"  ⚠ Validation error: {result.message[:200]}")


    # ════════════════════════════════════════════════════════
    # STEP 3: PyMOL visualization
    # ════════════════════════════════════════════════════════
    print(f"\n{'─'*66}")
    print(f"  STEP 3/5 — PyMOL visualization")
    print(f"{'─'*66}")

    if skip_pymol:
        steps["pymol"] = StepResult("pymol").skip(
            "--skip-pymol flag set"
        )
        print("  — Skipped (--skip-pymol)")
    elif "visualize" not in components:
        steps["pymol"] = StepResult("pymol").skip(
            "pymol_visualize.py not found"
        )
        print("  ⚠ Skipped — pymol_visualize.py not importable")
    elif not json_path:
        steps["pymol"] = StepResult("pymol").skip(
            "No results JSON from step 1"
        )
        print("  ⚠ Skipped — results JSON not found")
    else:
        # Pass clean_pdb explicitly to avoid filename-based lookup issues
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
    # STEP 4: Literature validation (RAG)
    # ════════════════════════════════════════════════════════
    print(f"\n{'─'*66}")
    print(f"  STEP 4/5 — Literature validation (PubMed + Gemma 4)")
    print(f"{'─'*66}")

    if skip_literature:
        steps["literature"] = StepResult("literature").skip(
            "--skip-literature flag set"
        )
        print("  — Skipped (--skip-literature)")
    elif "run_validation" not in components:
        steps["literature"] = StepResult("literature").skip(
            "literature_validator.py not found"
        )
        print("  ⚠ Skipped — literature_validator.py not importable")
    elif not json_path:
        steps["literature"] = StepResult("literature").skip(
            "No results JSON from step 1"
        )
        print("  ⚠ Skipped — results JSON not found")
    else:
        result = run_step(
            "literature",
            components["run_validation"],
            result_json_path = json_path,
            model            = ollama_model,
            n_papers         = n_papers,
        )
        steps["literature"] = result

        if result.status == "ok":
            val = result.data
            print(f"  ✓ Literature validation complete "
                  f"in {result.elapsed:.1f}s")
            print(f"    Papers found : {val.get('n_papers', 0)}")
        else:
            print(f"  ⚠ Literature validation error: "
                  f"{result.message[:200]}")


    # ════════════════════════════════════════════════════════
    # STEP 5: Gemma 4 plain-language explanation
    # ════════════════════════════════════════════════════════
    print(f"\n{'─'*66}")
    print(f"  STEP 5/5 — Gemma 4 explanation")
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
            "No results JSON from step 1"
        )
        print("  ⚠ Skipped — results JSON not found")
    else:
        with open(json_path) as f:
            pipeline_result = json.load(f)

        result = run_step(
            "explanation",
            components["explain"],
            result  = pipeline_result,
            model   = ollama_model,
            verbose = True,
        )
        steps["explanation"] = result

        if result.status == "ok":
            explained = result.data

            # Save explanation
            save_fn  = components.get("save_explanation")
            out_path = os.path.join(
                out_dir, f"{safe_drug}_explanation.json"
            )
            if save_fn:
                save_fn(explained, out_path)

            # Also save plain text
            txt_path = out_path.replace(".json", ".txt")
            with open(txt_path, "w") as f:
                f.write(
                    f"DockExplain — {drug_name} binding to "
                    f"{pipeline_result.get('protein','')}\n"
                )
                f.write("=" * 60 + "\n\n")
                f.write(explained.get("explanation", ""))

            print(f"\n  ✓ Explanation complete in {result.elapsed:.1f}s")
            print(f"    Saved: {out_path}")
            print(f"    Saved: {txt_path}")
        else:
            print(f"  ⚠ Explanation error: {result.message[:200]}")


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

def _run_pymol_safely(visualize_fn, json_path,
                      clean_pdb, interactive):
    """
    Wrapper that temporarily copies clean_pdb into
    the results directory so pymol_visualize.py finds it
    via its directory scan. Returns the .pml path.
    """
    import shutil
    out_dir  = os.path.dirname(json_path)
    dest_pdb = None

    # If clean_pdb isn't already in out_dir, copy it
    if clean_pdb and os.path.dirname(
        os.path.abspath(clean_pdb)
    ) != os.path.abspath(out_dir):
        dest_pdb = os.path.join(
            out_dir, os.path.basename(clean_pdb)
        )
        if not os.path.exists(dest_pdb):
            shutil.copy2(clean_pdb, dest_pdb)

    pml_path = visualize_fn(
        json_path, interactive=interactive
    )
    return pml_path


def _print_summary(steps: dict, elapsed: float):
    """Print a concise summary of all step outcomes."""
    ICONS = {"ok":"✓", "warn":"⚠", "skip":"—", "fail":"✗"}
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
        "literature" : "Literature (RAG)    ",
        "explanation": "Gemma 4 explanation ",
    }

    print(f"\n{'█'*66}")
    print("  DockExplain — Run Summary")
    print(f"{'─'*66}")

    all_ok   = True
    any_fail = False

    for key, label in STEP_LABELS.items():
        if key not in steps:
            print(f"  — {label} : not run")
            continue
        r = steps[key]
        icon  = ICONS.get(r.status, "?")
        col   = COLS.get(r.status, "")
        t_str = f"{r.elapsed:.1f}s" if r.elapsed > 0 else ""
        msg   = r.message[:50] if r.status != "ok" else ""
        print(f"  {col}{icon}{RESET} {label} : "
              f"{col}{r.status.upper():<5}{RESET} "
              f"{t_str:<8} {msg}")
        if r.status == "fail":
            any_fail = True
        if r.status != "ok":
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


def _save_run_manifest(steps: dict, out_dir: str,
                       safe_drug: str, protein_pdb: str,
                       drug_name: str, drug_smiles: str | None):
    """
    Save a run manifest JSON — useful for auditing and
    re-running specific steps.
    """
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
                "message": (v.message[:300]
                            if v.status != "ok" else ""),
            }
            for k, v in steps.items()
        }
    }
    path = os.path.join(out_dir, f"{safe_drug}_run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Run manifest: {path}")


# ════════════════════════════════════════════════════════
    # STEP 6: PDF report generation
    # ════════════════════════════════════════════════════════
    print(f"\n{'─'*66}")
    print(f"  STEP 6/6 — PDF report generation")
    print(f"{'─'*66}")

    try:
        from report_generator import generate_report
        if json_path:
            result = run_step(
                "pdf",
                generate_report,
                json_path,
            )
            steps["pdf"] = result
            if result.status == "ok":
                print(f"  ✓ PDF: {result.data}")
            else:
                print(f"  ⚠ PDF error: {result.message[:200]}")
        else:
            steps["pdf"] = StepResult("pdf").skip("No results JSON")
    except ImportError:
        steps["pdf"] = StepResult("pdf").skip(
            "report_generator.py not found or reportlab not installed.\n"
            "Install: conda install -c conda-forge reportlab -y"
        )
        print("  ⚠ Skipped — run: conda install -c conda-forge reportlab")


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

# In run_dockexplain.py — replace the __main__ block entirely

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DockExplain — drug-protein docking analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended — just provide the PDB ID)
  python run_dockexplain.py --pdb 1M17

  # Non-interactive (provide everything upfront)
  python run_dockexplain.py --pdb 1M17 --drug Erlotinib
  python run_dockexplain.py --pdb 1M17 --drug Ibuprofen \\
      --smiles "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
        """
    )
    parser.add_argument("--pdb", default=None,
        metavar="ID",
        help="RCSB Protein ID (e.g. 1M17). "
             "PDB file downloaded automatically.")
    parser.add_argument("--drug", default=None,
        metavar="NAME",
        help="Drug name (optional — prompted interactively if omitted)")
    parser.add_argument("--smiles", default=None,
        metavar="SMILES",
        help="Drug SMILES (optional — auto-fetched for crystal drugs)")
    parser.add_argument("--out", default=None,
        metavar="DIR",
        help="Output directory (default: ../data/results/<drug>/)")
    parser.add_argument("--exhaustiveness", type=int, default=8,
        metavar="N",
        help="Vina exhaustiveness (default: 8)")
    parser.add_argument("--model", default="gemma4:e2b",
        metavar="MODEL",
        help="Ollama model (default: gemma4:e2b)")
    parser.add_argument("--papers", type=int, default=10,
        metavar="N",
        help="PubMed papers for literature validation (default: 10)")
    parser.add_argument("--skip-pymol",       action="store_true")
    parser.add_argument("--skip-literature",  action="store_true")
    parser.add_argument("--skip-explanation", action="store_true")
    parser.add_argument("--headless",         action="store_true")
    args = parser.parse_args()

    # ── Resolve inputs ─────────────────────────────────────
    if args.pdb:
        # Non-interactive path: PDB ID provided on command line
        from python_scripts.input_handler_old import (
            download_pdb, detect_ligands, fetch_ligand_info,
            fetch_protein_metadata, resolve_drug_input
        )

        pdb_id   = args.pdb.strip().upper()
        pdb_path = download_pdb(pdb_id, out_dir="../data")

        if args.smiles:
            drug_smiles = args.smiles
            drug_name   = args.drug or pdb_id
        elif args.drug:
            # Resolve drug name/code to SMILES
            resolved    = resolve_drug_input(args.drug)
            drug_smiles = resolved["smiles"]
            drug_name   = args.drug
            if not drug_smiles:
                # Try as ligand code from the crystal structure
                ligands = detect_ligands(pdb_path)
                if ligands:
                    info        = fetch_ligand_info(ligands[0]["code"])
                    drug_smiles = info.get("smiles", "")
                    drug_name   = args.drug
        else:
            # No drug provided — run full interactive handler
            from python_scripts.input_handler_old import run_input_handler
            job         = run_input_handler(pdb_data_dir="../data")
            pdb_path    = job["pdb_path"]
            pdb_id      = job["pdb_id"]
            drug_name   = job["drug_name"]
            drug_smiles = job["drug_smiles"]

    else:
        # Fully interactive
        from python_scripts.input_handler_old import run_input_handler
        job         = run_input_handler(pdb_data_dir="../data")
        pdb_path    = job["pdb_path"]
        pdb_id      = job["pdb_id"]
        drug_name   = job["drug_name"]
        drug_smiles = job["drug_smiles"]

    # Output directory
    out_dir = args.out or os.path.join(
        "../data/results",
        drug_name.lower().replace(" ", "_")
    )

    # Run the full pipeline
    run_dockexplain(
        protein_pdb      = pdb_path,
        drug_name        = drug_name,
        drug_smiles      = drug_smiles if drug_smiles else None,
        out_dir          = out_dir,
        exhaustiveness   = args.exhaustiveness,
        ollama_model     = args.model,
        n_papers         = args.papers,
        skip_pymol       = args.skip_pymol,
        skip_literature  = args.skip_literature,
        skip_explanation = args.skip_explanation,
        pymol_headless   = args.headless,
    )
    