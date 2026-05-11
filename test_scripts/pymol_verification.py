# pymol_verification.py
#
# Headless PyMOL verification of PDBFixer output.
# Runs as subprocess — works whether PyMOL is in the same
# conda env or a separate one (set PYMOL_EXECUTABLE env var).
#
# Verifies:
#   1. RMSD between raw and cleaned PDB (must be ~0)
#   2. Lysine NZ protonation (must have 3 H)
#   3. Aspartate/Glutamate carboxylate (must have 0 H)
#   4. Histidine tautomer states (reports which N is protonated)
#   5. Renders binding pocket PNG for visual inspection
#
# Run: python pymol_verification.py \
#          --raw ../data/1M17.pdb \
#          --clean ../data/results/mode_a/1M17_clean.pdb \
#          --pose ../data/results/mode_a/erlotinib_best_pose.sdf \
#          --residues 694,702,721,738,742,764,768,769,820,831 \
#          --out ../data/results/mode_a/

import argparse, json, os, subprocess, sys, tempfile


PYMOL_EXE = os.environ.get("PYMOL_EXECUTABLE", "pymol")


PYMOL_SCRIPT_TEMPLATE = r"""
# Auto-generated PyMOL verification script
import json
from pymol import cmd

RAW       = "{raw}"
CLEAN     = "{clean}"
POSE      = "{pose}"
RESIDUES  = "{residues}"        # comma-separated resi numbers
OUT_JSON  = "{out_json}"
OUT_PNG   = "{out_png}"

report = {{
    "rmsd_overall"   : None,
    "rmsd_pocket"    : None,
    "lysines"        : [],     # list of {{resi, n_hydrogens, ok}}
    "carboxylates"   : [],     # list of {{resi, residue, n_hydrogens, ok}}
    "histidines"     : [],     # list of {{resi, tautomer}}
    "pocket_residues": [r.strip() for r in RESIDUES.split(",") if r.strip()],
    "errors"         : [],
}}

try:
    cmd.load(RAW,   "raw")
    cmd.load(CLEAN, "clean")

    # ── Overall RMSD ────────────────────────────────────────
    # align returns tuple: (rmsd, n_atoms, n_cycles, ...)
    try:
        r = cmd.align("clean", "raw")
        report["rmsd_overall"] = round(float(r[0]), 4)
    except Exception as e:
        report["errors"].append(f"overall align failed: {{e}}")

    # ── Pocket RMSD ─────────────────────────────────────────
    if RESIDUES:
        sel_expr = "+".join(report["pocket_residues"])
        try:
            r = cmd.align(f"clean and resi {{sel_expr}}",
                          f"raw   and resi {{sel_expr}}")
            report["rmsd_pocket"] = round(float(r[0]), 4)
        except Exception as e:
            report["errors"].append(f"pocket align failed: {{e}}")

    # ── Lysine NZ protonation ───────────────────────────────
    # Collect every LYS in the pocket list (or all if not specified)
    lys_sel = "clean and resn LYS and name CA"
    if RESIDUES:
        lys_sel += f" and resi {{'+'.join(report['pocket_residues'])}}"

    lys_resis = []
    cmd.iterate(lys_sel, "lys_resis.append(resi)",
                space={{"lys_resis": lys_resis}})

    for resi in lys_resis:
        n_h = cmd.count_atoms(
            f"clean and resi {{resi}} and elem H and neighbor name NZ"
        )
        report["lysines"].append({{
            "resi"         : resi,
            "n_hydrogens"  : n_h,
            "ok"           : n_h == 3,
            "expected"     : 3,
        }})

    # ── Aspartate / Glutamate carboxylate ───────────────────
    for resn, o_names in [("ASP", "OD1 OD2"), ("GLU", "OE1 OE2")]:
        sel = f"clean and resn {{resn}} and name CA"
        if RESIDUES:
            sel += f" and resi {{'+'.join(report['pocket_residues'])}}"
        resis = []
        cmd.iterate(sel, "resis.append(resi)",
                    space={{"resis": resis}})

        for resi in resis:
            # count H bonded to either carboxylate oxygen
            o1, o2 = o_names.split()
            n_h = cmd.count_atoms(
                f"clean and resi {{resi}} and elem H "
                f"and (neighbor name {{o1}} or neighbor name {{o2}})"
            )
            report["carboxylates"].append({{
                "resi"        : resi,
                "residue"     : resn,
                "n_hydrogens" : n_h,
                "ok"          : n_h == 0,
                "expected"    : 0,
            }})

    # ── Histidine tautomer detection ────────────────────────
    his_sel = "clean and resn HIS and name CA"
    if RESIDUES:
        his_sel += f" and resi {{'+'.join(report['pocket_residues'])}}"
    his_resis = []
    cmd.iterate(his_sel, "his_resis.append(resi)",
                space={{"his_resis": his_resis}})

    for resi in his_resis:
        h_nd1 = cmd.count_atoms(
            f"clean and resi {{resi}} and elem H and neighbor name ND1"
        )
        h_ne2 = cmd.count_atoms(
            f"clean and resi {{resi}} and elem H and neighbor name NE2"
        )
        if h_nd1 and h_ne2:     tautomer = "HIP (both protonated, +1 charge)"
        elif h_nd1:             tautomer = "HID (ND1 protonated, neutral)"
        elif h_ne2:             tautomer = "HIE (NE2 protonated, neutral)"
        else:                   tautomer = "unprotonated (unusual)"
        report["histidines"].append({{
            "resi"     : resi,
            "h_nd1"    : h_nd1,
            "h_ne2"    : h_ne2,
            "tautomer" : tautomer,
        }})

    # ── Render binding pocket image ─────────────────────────
    if POSE and POSE != "None":
        try:
            cmd.load(POSE, "lig")
            cmd.hide("everything")
            cmd.show("cartoon", "clean")
            cmd.color("cyan", "clean")
            cmd.show("sticks", "lig")
            cmd.color("yellow", "lig and elem C")
            if RESIDUES:
                pocket = f"clean and resi {{'+'.join(report['pocket_residues'])}}"
                cmd.show("sticks", pocket + " and not name C+N+O")
                cmd.color("salmon", pocket + " and elem C")
            cmd.zoom("lig", 6)
            cmd.bg_color("white")
            cmd.set("ray_opaque_background", 0)
            cmd.ray(1200, 900)
            cmd.png(OUT_PNG, dpi=150)
        except Exception as e:
            report["errors"].append(f"render failed: {{e}}")

except Exception as e:
    report["errors"].append(f"fatal: {{e}}")

with open(OUT_JSON, "w") as f:
    json.dump(report, f, indent=2)

print(f"PYMOL_VERIFICATION_DONE {{OUT_JSON}}")
cmd.quit()
"""


def run_pymol_verification(raw_pdb, clean_pdb, pose_sdf,
                           pocket_residues, out_dir):
    """
    Run headless PyMOL verification. Returns the report dict.
    
    pocket_residues : list of resi numbers (str or int) to check
    """
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "pymol_verification.json")
    out_png  = os.path.join(out_dir, "pocket_view.png")

    residues_str = ",".join(str(r) for r in pocket_residues)

    script = PYMOL_SCRIPT_TEMPLATE.format(
        raw      = os.path.abspath(raw_pdb),
        clean    = os.path.abspath(clean_pdb),
        pose     = os.path.abspath(pose_sdf) if pose_sdf else "None",
        residues = residues_str,
        out_json = os.path.abspath(out_json),
        out_png  = os.path.abspath(out_png),
    )

    # Write script to temp file — PyMOL runs it with -cq
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                      delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        print(f"  Running headless PyMOL: {PYMOL_EXE} -cq {script_path}")
        result = subprocess.run(
            [PYMOL_EXE, "-cq", script_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  ⚠ PyMOL stderr: {result.stderr[:500]}")
    finally:
        os.unlink(script_path)

    if not os.path.exists(out_json):
        raise RuntimeError(
            f"PyMOL did not produce {out_json}. "
            f"Is PyMOL installed? Set PYMOL_EXECUTABLE env var "
            f"if pymol is not on PATH. stderr: {result.stderr[:300]}"
        )

    with open(out_json) as f:
        report = json.load(f)

    return report, out_png


def print_verification_summary(report, png_path):
    """Pretty-print the verification report."""
    print("\n" + "═" * 58)
    print("PyMOL STRUCTURE VERIFICATION")
    print("═" * 58)

    # RMSD
    overall = report.get("rmsd_overall")
    pocket  = report.get("rmsd_pocket")
    print(f"\n  RMSD (raw vs clean):")
    print(f"    Overall : {overall} Å  "
          f"{'✓' if overall is not None and overall < 0.5 else '⚠'}")
    print(f"    Pocket  : {pocket} Å  "
          f"{'✓' if pocket is not None and pocket < 0.5 else '⚠'}")

    # Lysines
    if report["lysines"]:
        print(f"\n  Lysine NZ protonation (expect 3 H):")
        for lys in report["lysines"]:
            tag = "✓" if lys["ok"] else "✗"
            print(f"    LYS{lys['resi']:>4}  "
                  f"NZ has {lys['n_hydrogens']} H  {tag}")

    # Carboxylates
    if report["carboxylates"]:
        print(f"\n  Asp/Glu carboxylate (expect 0 H):")
        for c in report["carboxylates"]:
            tag = "✓" if c["ok"] else "✗"
            print(f"    {c['residue']}{c['resi']:>4}  "
                  f"carboxylate has {c['n_hydrogens']} H  {tag}")

    # Histidines
    if report["histidines"]:
        print(f"\n  Histidine tautomers:")
        for h in report["histidines"]:
            print(f"    HIS{h['resi']:>4}  → {h['tautomer']}")
    else:
        print(f"\n  No histidines in pocket — no tautomer concerns.")

    # Errors
    if report.get("errors"):
        print(f"\n  ⚠ Warnings:")
        for e in report["errors"]:
            print(f"    {e}")

    # Image
    if os.path.exists(png_path):
        print(f"\n  Pocket rendering: {png_path}")

    # Overall pass/fail
    ok_rmsd = (report.get("rmsd_overall") is not None
               and report["rmsd_overall"] < 0.5)
    ok_lys  = all(l["ok"] for l in report["lysines"])
    ok_cbx  = all(c["ok"] for c in report["carboxylates"])

    print("\n  " + "─" * 54)
    if ok_rmsd and ok_lys and ok_cbx:
        print("  ✓ VERIFICATION PASSED — structure is chemically sound")
    else:
        print("  ⚠ VERIFICATION ISSUES — review warnings above")
    print("═" * 58 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",      required=True)
    parser.add_argument("--clean",    required=True)
    parser.add_argument("--pose",     default=None)
    parser.add_argument("--residues", default="",
        help="Comma-separated pocket resi numbers")
    parser.add_argument("--out",      default="./")
    args = parser.parse_args()

    residues = [r.strip() for r in args.residues.split(",") if r.strip()]
    report, png = run_pymol_verification(
        args.raw, args.clean, args.pose, residues, args.out
    )
    print_verification_summary(report, png)