# pymol_visualize.py
# Generates a PyMOL script from DockExplain results
# Then runs it automatically if PyMOL is installed
#
# Run:
#   python pymol_visualize.py ../data/results/mode_a/erlotinib_results.json
#
# Headless:
#   python pymol_visualize.py ../data/results/mode_a/erlotinib_results.json --headless
#
# Prerequisites:
#   brew install pymol
#   or
#   conda install -c conda-forge pymol-open-source

import json
import os
import sys
import re
import subprocess
import shutil
from collections import defaultdict


# ════════════════════════════════════════════════════════════
# COLOUR SCHEME
# Professional PyMOL colours for each interaction type
# ════════════════════════════════════════════════════════════

INTERACTION_COLORS = {
    "HBDonor"    : ("hbond_donor",    "0x1565C0"),   # dark blue
    "HBAcceptor" : ("hbond_acceptor", "0x1976D2"),   # blue
    "Hydrophobic": ("hydrophobic",    "0xFF8F00"),   # amber
    "PiStacking" : ("pi_stack",       "0x6A1B9A"),   # purple
    "EdgeToFace" : ("edge_face",      "0x8E24AA"),   # light purple
    "Cationic"   : ("cationic",       "0xC62828"),   # dark red
    "Anionic"    : ("anionic",        "0xAD1457"),   # pink
    "CationPi"   : ("cation_pi",      "0x00838F"),   # cyan
    "PiCation"   : ("pi_cation",      "0x00695C"),   # teal
}


def hex_to_rgb(hex_color: str) -> str:
    """
    Convert a PyMOL hex color string to an RGB float-list string.

    PyMOL set_color requires:
        [r, g, b]

    Not:
        0x1565C0

    Input:
        "0x1976D2" or "#1976D2"

    Output:
        "[0.098, 0.463, 0.824]"
    """
    hex_str = str(hex_color).replace("0x", "").replace("#", "").strip()

    if len(hex_str) != 6:
        # Safe fallback: green
        hex_str = "4CAF50"

    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0

    return f"[{r:.3f}, {g:.3f}, {b:.3f}]"


# Interactions that should show distance measurements
SHOW_DISTANCE = {
    "HBDonor", "HBAcceptor",
    "Cationic", "Anionic",
    "CationPi", "PiCation",
}


def parse_residue(residue_str: str):
    """
    Parse 'MET769.A' → (resname='MET', resi='769', chain='A')
    Handles both 'MET769.A' and 'MET769' formats.
    """
    parts = residue_str.strip().split(".")
    chain = parts[1] if len(parts) > 1 else "A"

    m = re.match(r"([A-Z]+)(\d+)", parts[0])
    if not m:
        return None, None, None

    return m.group(1), m.group(2), chain


def generate_pymol_script(result_json_path: str,
                          interactive: bool = True) -> tuple[str, str]:
    """
    Generate a .pml PyMOL script from DockExplain results.

    Args:
        result_json_path : path to *_results.json
        interactive      : True = high-quality GUI render
                           False = fast headless web/server render

    Returns:
        (pml_path, out_dir)
    """
    with open(result_json_path, encoding="utf-8") as f:
        result = json.load(f)

    drug         = result["drug"]
    protein      = result["protein"]
    interactions = result.get("interactions", [])
    score        = result.get("docking_score")
    score_str    = f"{score:.3f} kcal/mol" if score else "crystal pose"
    out_dir      = os.path.dirname(result_json_path)
    safe_drug    = drug.lower().replace(" ", "_")

    # ── Locate structure files ────────────────────────────
    protein_pdb = None

    for fname in os.listdir(out_dir):
        if fname.endswith("_clean.pdb"):
            protein_pdb = os.path.join(out_dir, fname)
            break

    if not protein_pdb:
        parent = os.path.dirname(out_dir)
        if os.path.isdir(parent):
            for fname in os.listdir(parent):
                if "clean" in fname and fname.endswith(".pdb"):
                    protein_pdb = os.path.join(parent, fname)
                    break

    ligand_sdf = os.path.join(out_dir, f"{safe_drug}_best_pose.sdf")

    if not protein_pdb or not os.path.exists(protein_pdb):
        raise FileNotFoundError(
            f"Cleaned protein PDB not found in {out_dir}\n"
            "Run dockexplain_pipeline.py first."
        )

    if not os.path.exists(ligand_sdf):
        raise FileNotFoundError(
            f"Ligand SDF not found: {ligand_sdf}\n"
            "Run dockexplain_pipeline.py first."
        )

    # Convert SDF to PDB for PyMOL
    ligand_pdb = ligand_sdf.replace(".sdf", ".pdb")
    _sdf_to_pdb(ligand_sdf, ligand_pdb)

    # ── Group interactions by residue ─────────────────────
    residue_map = defaultdict(list)

    for ix in interactions:
        residue = ix.get("residue")
        itype   = ix.get("interaction_type")

        if residue and itype:
            residue_map[residue].append(itype)

    # ── Output paths ──────────────────────────────────────
    pml_path     = os.path.join(out_dir, f"{safe_drug}_pymol.pml")
    session_path = os.path.join(out_dir, f"{safe_drug}_session.pse")
    image_path   = os.path.join(out_dir, f"{safe_drug}_visualization.png")

    protein_pdb_abs = os.path.abspath(protein_pdb)
    ligand_pdb_abs  = os.path.abspath(ligand_pdb)
    session_abs     = os.path.abspath(session_path)
    image_abs       = os.path.abspath(image_path)

    # ════════════════════════════════════════════════════════
    # BUILD THE PyMOL SCRIPT
    # ════════════════════════════════════════════════════════
    lines = []

    def cmd(s=""):
        lines.append(s)

    # ── Header ────────────────────────────────────────────
    cmd("# ═══════════════════════════════════════════════")
    cmd(f"# DockExplain — {drug} ↔ {protein}")
    cmd(f"# Binding score: {score_str}")
    cmd(f"# Interactions: {len(interactions)}")
    cmd("# ═══════════════════════════════════════════════")
    cmd()

    # ── Global settings ───────────────────────────────────
    cmd("# ── Global display settings ──────────────────────")
    cmd("set bg_rgb, [0.98, 0.99, 1.00]")          # light professional background
    cmd("set ray_shadows, 0")
    cmd("set ray_trace_fog, 0")
    cmd("set ray_opaque_background, 1")
    cmd("set depth_cue, 1")
    cmd("set fog_start, 0.45")
    cmd("set antialias, 2")
    cmd("set ray_trace_mode, 1")
    cmd("set stick_radius, 0.15")
    cmd("set sphere_scale, 0.25")
    cmd("set label_size, 14")
    cmd("set label_color, black")
    cmd("set label_font_id, 7")
    cmd("set label_bg_color, white")
    cmd("set label_bg_transparency, 0.25")
    cmd("set dash_radius, 0.06")
    cmd("set dash_gap, 0.3")
    cmd("set dash_length, 0.5")
    cmd("set distance_exclusion, 0")
    cmd()

    # ── Load structures ───────────────────────────────────
    cmd("# ── Load protein and ligand ──────────────────────")
    cmd(f'load {protein_pdb_abs}, protein')
    cmd(f'load {ligand_pdb_abs}, ligand')
    cmd()

    # ── Protein representation ────────────────────────────
    cmd("# ── Protein styling ──────────────────────────────")
    cmd("hide everything, protein")
    cmd("show cartoon, protein")
    cmd("color slate, protein")
    cmd("set cartoon_transparency, 0.30, protein")
    cmd("set cartoon_fancy_helices, 1")
    cmd("set cartoon_smooth_loops, 1")
    cmd()

    # ── Ligand representation ─────────────────────────────
    cmd("# ── Ligand styling ───────────────────────────────")
    cmd("hide everything, ligand")
    cmd("show sticks, ligand")
    cmd("show spheres, ligand")
    cmd("set stick_radius, 0.22, ligand")
    cmd("set sphere_scale, 0.32, ligand")
    cmd("util.cbag ligand")
    cmd("set stick_transparency, 0.0, ligand")
    cmd()

    # ── Surface around binding site ───────────────────────
    cmd("# ── Binding site surface ────────────────────────")
    cmd("select binding_pocket, byres (protein within 8 of ligand)")
    cmd("show surface, binding_pocket")
    cmd("color grey80, binding_pocket")
    cmd("set transparency, 0.70, binding_pocket")
    cmd("set surface_quality, 1")
    cmd()

    # ── Highlight interacting residues ────────────────────
    cmd("# ── Interacting residues ────────────────────────")

    all_interaction_sele = []

    for residue_str, itypes in residue_map.items():
        resname, resi, chain = parse_residue(residue_str)

        if not resname:
            continue

        primary = itypes[0]

        color_name, color_hex = INTERACTION_COLORS.get(
            primary, ("other", "0x4CAF50")
        )

        sele_name = f"res_{resname}{resi}"
        color_var = f"col_{color_name}"

        cmd(f"# {residue_str}: {' + '.join(itypes)}")
        cmd(f"set_color {color_var}, {hex_to_rgb(color_hex)}")

        cmd(
            f"select {sele_name}, "
            f"(protein and resn {resname} and resi {resi} "
            f"and chain {chain})"
        )

        cmd(f"show sticks, {sele_name}")
        cmd(f"show spheres, {sele_name}")
        cmd(f"set stick_radius, 0.20, {sele_name}")
        cmd(f"set sphere_scale, 0.30, {sele_name}")
        cmd(f"color {color_var}, {sele_name}")

        label_text = f"{resname}{resi}\\n{'+'.join(itypes)}"
        cmd(f'label ({sele_name} and name CA), "{label_text}"')

        all_interaction_sele.append(sele_name)
        cmd()

    # ── Distance measurements for polar/interpretable interactions ──────
    cmd("# ── Distance measurements ───────────────────────")
    dist_counter = 1

    for ix in interactions:
        if ix.get("interaction_type") not in SHOW_DISTANCE:
            continue

        resname, resi, chain = parse_residue(ix.get("residue", ""))

        if not resname:
            continue

        itype = ix["interaction_type"]

        _, color_hex = INTERACTION_COLORS.get(
            itype, ("other", "0x4CAF50")
        )

        dist_name = f"dist_{dist_counter:02d}_{resname}{resi}"
        color_var = f"dist_col_{dist_counter}"

        if "HB" in itype:
            cmd(
                f"distance {dist_name}, "
                f"(ligand and not elem H), "
                f"(protein and resn {resname} and resi {resi} "
                f"and chain {chain} and not elem H and "
                f"(name N or name O or name S))"
            )
        else:
            cmd(
                f"distance {dist_name}, "
                f"(ligand and not elem H), "
                f"(protein and resn {resname} and resi {resi} "
                f"and chain {chain} and not elem H)"
            )

        # IMPORTANT FIX:
        # PyMOL set_color cannot accept 0x1565C0 directly.
        # It needs [r, g, b].
        cmd(f"set_color {color_var}, {hex_to_rgb(color_hex)}")
        cmd(f"color {color_var}, {dist_name}")
        cmd(f"set dash_radius, 0.06, {dist_name}")
        cmd(f"show dashes, {dist_name}")
        cmd(f"hide labels, {dist_name}")
        cmd()

        dist_counter += 1

    # ── Overall interaction selection group ───────────────
    if all_interaction_sele:
        cmd("# ── Group all interacting residues ─────────────")
        combined = " or ".join(all_interaction_sele)
        cmd(f"select all_interactions, ({combined})")
        cmd()

    # ── Final camera and zoom ─────────────────────────────
    cmd("# ── Camera and rendering ─────────────────────────")
    cmd("zoom ligand, 12")
    cmd("orient ligand")
    cmd("turn y, 15")
    cmd("turn x, -10")
    cmd()
    cmd("deselect")
    cmd()

    # ── Legend via pseudoatoms ────────────────────────────
    cmd("# ── Legend (pseudoatom labels) ────────────────────")

    seen_types = {}

    for ix in interactions:
        t = ix.get("interaction_type")
        if t and t not in seen_types:
            seen_types[t] = INTERACTION_COLORS.get(
                t, ("other", "0x4CAF50")
            )

    legend_y = 5.0

    for itype, (color_name, color_hex) in seen_types.items():
        leg_name  = f"legend_{color_name}"
        color_var = f"leg_col_{color_name}"

        # IMPORTANT FIX:
        # Convert hex to PyMOL RGB list format.
        cmd(f"set_color {color_var}, {hex_to_rgb(color_hex)}")

        cmd(
            f"pseudoatom {leg_name}, "
            f"pos=[30, {legend_y:.1f}, 0]"
        )

        cmd(f"color {color_var}, {leg_name}")
        cmd(f"show sphere, {leg_name}")
        cmd(f"set sphere_scale, 0.5, {leg_name}")
        cmd(f'label {leg_name}, "{itype}"')

        legend_y -= 2.5

    cmd()

    # ── Save session and image ────────────────────────────
    cmd("# ── Save outputs ─────────────────────────────────")
    cmd(f"save {session_abs}")
    cmd()

    cmd("# Render PNG")
    if interactive:
        # Full-quality render for desktop/GUI use.
        cmd("set ray_trace_mode, 1")
        cmd("set ray_shadows, 1")
        cmd("ray 2400, 1800")
        cmd(f"png {image_abs}, dpi=300")
    else:
        # Fast headless render for web/server use.
        # This avoids the 2-5 minute ray-tracing delay.
        cmd("set ray_trace_mode, 0")
        cmd("set ray_shadows, 0")
        cmd(f"png {image_abs}, width=1400, height=1000, dpi=150, ray=0")

    cmd()
    cmd(f"# Session saved: {session_abs}")
    cmd(f"# Image saved  : {image_abs}")
    cmd()
    cmd("# Type 'quit' to exit after saving.")

    # In headless mode, close PyMOL automatically after saving.
    if not interactive:
        cmd("quit")

    # ── Write .pml file ───────────────────────────────────
    with open(pml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✓ PyMOL script written: {pml_path}")
    return pml_path, out_dir


def _sdf_to_pdb(sdf_path: str, pdb_path: str):
    """Convert SDF to PDB using RDKit for PyMOL compatibility."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next((m for m in supplier if m is not None), None)

    if mol is None:
        raise RuntimeError(f"Could not read SDF: {sdf_path}")

    writer = Chem.PDBWriter(pdb_path)
    writer.write(mol)
    writer.close()


def run_pymol(pml_path: str, interactive: bool = True):
    """
    Launch PyMOL with the generated script.

    Args:
        pml_path    : path to the .pml script
        interactive : True = open GUI
                      False = headless render only
    """
    pymol_exe = None

    candidates = [
        "pymol",
        "/usr/local/bin/pymol",
        "/opt/homebrew/bin/pymol",
        "/Applications/PyMOL.app/Contents/MacOS/PyMOL",
        shutil.which("pymol"),
    ]

    for candidate in candidates:
        if not candidate:
            continue

        # If candidate is a name, check PATH
        found = shutil.which(candidate)
        if found:
            pymol_exe = found
            break

        # If candidate is an absolute path, check existence
        if os.path.exists(str(candidate)):
            pymol_exe = str(candidate)
            break

    if not pymol_exe:
        print("\n⚠ PyMOL not found in PATH.")
        print("  Install: brew install pymol")
        print("  Or:      conda install -c conda-forge pymol-open-source")
        print("\n  To run manually:")
        print(f"    pymol {pml_path}")
        return False

    print(f"\n  PyMOL found: {pymol_exe}")

    if interactive:
        print("  Launching PyMOL GUI (interactive mode)...")
        print("  The visualization will open — explore freely.")
        print("  Session and PNG will be saved automatically.")
        subprocess.Popen([pymol_exe, pml_path])
    else:
        print("  Rendering headless (fast web/server mode)...")

        # -c = command line / no GUI
        # -r = run script
        # stdout/stderr are inherited, so Flask subprocess logging can capture them.
        subprocess.run(
            [pymol_exe, "-c", "-r", pml_path],
            check=True
        )

        print("  ✓ Headless render complete")

    return True


# ════════════════════════════════════════════════════════════
# MAIN VISUALIZATION ENTRYPOINT
# ════════════════════════════════════════════════════════════

def visualize(result_json_path: str,
              interactive: bool = True):
    """
    Full visualization pipeline:
      1. Generate .pml script from results JSON
      2. Launch PyMOL with the script
      3. Save .pse session + PNG

    Args:
        result_json_path : path to *_results.json from pipeline
        interactive      : True = open GUI and high-quality render
                           False = headless fast render
    """
    with open(result_json_path, encoding="utf-8") as f:
        result = json.load(f)

    drug    = result["drug"]
    protein = result["protein"]
    n_ix    = result.get("n_interactions", len(result.get("interactions", [])))

    print("=" * 58)
    print("DockExplain — PyMOL Visualization")
    print(f"  Drug        : {drug}")
    print(f"  Protein     : {protein}")
    print(f"  Interactions: {n_ix}")
    print("=" * 58)

    print("\n[1/2] Generating PyMOL script...")
    pml_path, out_dir = generate_pymol_script(
        result_json_path,
        interactive=interactive
    )

    interactions = result.get("interactions", [])
    residue_map = defaultdict(list)

    for ix in interactions:
        residue = ix.get("residue")
        itype   = ix.get("interaction_type")

        if residue and itype:
            residue_map[residue].append(itype)

    hbond_res = [
        r for r, t in residue_map.items()
        if any(x in {"HBDonor", "HBAcceptor"} for x in t)
    ]

    print("\n  Visualization contents:")
    print("    ✓ Protein cartoon + transparent surface")
    print(f"    ✓ {drug} as coloured sticks + spheres")
    print(f"    ✓ {len(residue_map)} interacting residues highlighted")
    print(f"    ✓ Distance dashes for {len(hbond_res)} H-bond residues")
    print("    ✓ Labels on every interacting residue")
    print("    ✓ Colour-coded by interaction type")
    if interactive:
        print("    ✓ Saves .pse session + 2400×1800 PNG at 300 dpi")
    else:
        print("    ✓ Saves .pse session + fast 1400×1000 PNG")

    print("\n[2/2] Launching PyMOL...")
    success = run_pymol(pml_path, interactive=interactive)

    if success:
        safe_drug    = drug.lower().replace(" ", "_")
        session_path = os.path.join(out_dir, f"{safe_drug}_session.pse")
        image_path   = os.path.join(out_dir, f"{safe_drug}_visualization.png")

        print("\n  Outputs:")
        print(f"    Session : {session_path}")
        print(f"    Image   : {image_path}")
        print(f"    Script  : {pml_path}")
    else:
        print("\n  Script ready to run manually:")
        print(f"    pymol {pml_path}")

    return pml_path


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DockExplain PyMOL visualization"
    )

    parser.add_argument(
        "json",
        nargs="?",
        default="../data/results/mode_a/erlotinib_results.json",
        help="Path to *_results.json from dockexplain_pipeline"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render without GUI using fast web/server mode"
    )

    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"Error: {args.json} not found")
        print("Run dockexplain_pipeline.py first")
        sys.exit(1)

    visualize(args.json, interactive=not args.headless)
