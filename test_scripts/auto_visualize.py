# auto_visualize.py
# Automated 3D visualization of docked pose + interactions
# Generates a self-contained HTML file — open in any browser
# Run: python auto_visualize.py \
#          ../data/results/mode_a/erlotinib_results.json

import json
import os
import sys

def generate_visualization(result_json_path: str) -> str:
    """
    Generate an interactive 3D HTML visualization showing:
      - Protein surface and cartoon
      - Docked ligand as 3D sticks
      - Interacting residues highlighted by interaction type
      - Clickable interaction labels
      - Color legend

    Returns path to the saved HTML file.
    """
    # ── Load result JSON ──────────────────────────────────
    with open(result_json_path) as f:
        result = json.load(f)

    drug         = result["drug"]
    protein      = result["protein"]
    interactions = result.get("interactions", [])
    score        = result.get("docking_score")
    score_str    = f"{score:.2f} kcal/mol" if score else "crystal pose"
    out_dir      = os.path.dirname(result_json_path)
    safe_drug    = drug.lower().replace(" ","_")

    # ── Locate structure files ────────────────────────────
    # Find cleaned protein PDB
    protein_pdb = None
    for fname in os.listdir(out_dir):
        if fname.endswith("_clean.pdb"):
            protein_pdb = os.path.join(out_dir, fname)
            break
    if not protein_pdb:
        # Check parent data directory
        parent = os.path.dirname(out_dir)
        for fname in os.listdir(parent):
            if "clean" in fname and fname.endswith(".pdb"):
                protein_pdb = os.path.join(parent, fname)
                break

    # Find ligand SDF
    ligand_sdf = os.path.join(out_dir, f"{safe_drug}_best_pose.sdf")

    if not protein_pdb or not os.path.exists(protein_pdb):
        raise FileNotFoundError(
            f"Could not find cleaned protein PDB in {out_dir}\n"
            f"Run dockexplain_pipeline.py first."
        )
    if not os.path.exists(ligand_sdf):
        raise FileNotFoundError(
            f"Could not find ligand SDF: {ligand_sdf}\n"
            f"Run dockexplain_pipeline.py first."
        )

    # ── Read structure files ──────────────────────────────
    with open(protein_pdb) as f:
        protein_pdbstr = f.read()
    with open(ligand_sdf) as f:
        ligand_sdfstr = f.read()

    # ── Build interaction colour map ──────────────────────
    # Group residues by interaction type for colouring
    COLORS = {
        "HBDonor"    : "#2196F3",   # blue
        "HBAcceptor" : "#1565C0",   # dark blue
        "Hydrophobic": "#FF9800",   # orange
        "PiStacking" : "#9C27B0",   # purple
        "EdgeToFace" : "#7B1FA2",   # dark purple
        "Cationic"   : "#F44336",   # red
        "Anionic"    : "#E91E63",   # pink
        "CationPi"   : "#00BCD4",   # cyan
        "PiCation"   : "#009688",   # teal
    }

    # Build per-residue data for JavaScript
    residue_data = {}
    for ix in interactions:
        res = ix["residue"]           # e.g. "MET769.A"
        itype = ix["interaction_type"]
        if res not in residue_data:
            residue_data[res] = []
        residue_data[res].append(itype)

    # Build JavaScript arrays for highlighting
    highlight_js_parts = []
    for res, itypes in residue_data.items():
        # Parse residue name and number
        # Format is like "MET769.A" → resn=MET, resi=769, chain=A
        parts = res.split(".")
        chain = parts[1] if len(parts) > 1 else "A"
        resid_part = parts[0]   # e.g. MET769

        # Extract residue number (digits at end)
        import re
        m = re.match(r"([A-Z]+)(\d+)", resid_part)
        if not m:
            continue
        resname = m.group(1)
        resnum  = m.group(2)

        # Primary interaction type determines colour
        primary = itypes[0]
        color   = COLORS.get(primary, "#4CAF50")
        label   = " + ".join(itypes)

        highlight_js_parts.append(
            f'  highligthResidue(viewer, "{resname}", '
            f'"{resnum}", "{chain}", "{color}", "{label}");'
        )

    highlight_js = "\n".join(highlight_js_parts)

    # ── Build legend HTML ─────────────────────────────────
    seen_types = set()
    for ix in interactions:
        seen_types.add(ix["interaction_type"])

    legend_items = []
    for itype in [
        "HBDonor","HBAcceptor","Hydrophobic",
        "PiStacking","EdgeToFace",
        "Cationic","Anionic","CationPi","PiCation"
    ]:
        if itype in seen_types:
            color = COLORS.get(itype, "#4CAF50")
            legend_items.append(
                f'<div class="legend-item">'
                f'<div class="legend-dot" '
                f'style="background:{color}"></div>'
                f'{itype}</div>'
            )
    legend_html = "\n".join(legend_items)

    # ── Build interaction table HTML ──────────────────────
    table_rows = []
    for ix in interactions:
        color = COLORS.get(ix["interaction_type"], "#4CAF50")
        table_rows.append(
            f'<tr>'
            f'<td class="res-cell">{ix["residue"]}</td>'
            f'<td><span class="badge" style="background:{color}">'
            f'{ix["interaction_type"]}</span></td>'
            f'<td class="desc-cell">{ix["description"]}</td>'
            f'</tr>'
        )
    table_html = "\n".join(table_rows)

    # ── Build the full HTML page ──────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DockExplain — {drug} ↔ {protein}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    min-height: 100vh;
  }}

  .header {{
    background: #1a1d27;
    border-bottom: 1px solid #2a2d3a;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
  }}

  .header-title {{
    font-size: 20px;
    font-weight: 600;
    color: #fff;
  }}

  .header-sub {{
    font-size: 13px;
    color: #888;
    margin-top: 3px;
  }}

  .score-badge {{
    background: #1e3a5f;
    border: 1px solid #2d6db4;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 14px;
    color: #64b5f6;
    font-weight: 500;
  }}

  .layout {{
    display: grid;
    grid-template-columns: 1fr 380px;
    height: calc(100vh - 65px);
  }}

  /* ── Viewer ── */
  #viewer-container {{
    position: relative;
    background: #070a0f;
  }}

  #viewer {{
    width: 100%;
    height: 100%;
  }}

  .viewer-controls {{
    position: absolute;
    top: 12px;
    left: 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}

  .ctrl-btn {{
    background: rgba(26,29,39,0.92);
    border: 1px solid #2a2d3a;
    border-radius: 6px;
    color: #ccc;
    font-size: 12px;
    padding: 6px 12px;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }}

  .ctrl-btn:hover {{ background: rgba(40,44,60,0.95); color: #fff; }}
  .ctrl-btn.active {{ border-color: #2d6db4; color: #64b5f6; }}

  /* ── Sidebar ── */
  .sidebar {{
    background: #1a1d27;
    border-left: 1px solid #2a2d3a;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }}

  .sidebar-section {{
    padding: 16px;
    border-bottom: 1px solid #2a2d3a;
  }}

  .section-title {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #666;
    margin-bottom: 12px;
  }}

  /* Legend */
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #aaa;
    margin-bottom: 6px;
  }}

  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }}

  /* Interaction table */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}

  th {{
    text-align: left;
    padding: 6px 8px;
    color: #666;
    font-weight: 500;
    border-bottom: 1px solid #2a2d3a;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  td {{
    padding: 7px 8px;
    border-bottom: 1px solid #1e2130;
    vertical-align: top;
  }}

  tr:hover td {{ background: #1e2234; }}

  .res-cell {{
    font-family: monospace;
    font-size: 12px;
    color: #e0e0e0;
    white-space: nowrap;
  }}

  .badge {{
    display: inline-block;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 99px;
    color: #fff;
    font-weight: 500;
    white-space: nowrap;
  }}

  .desc-cell {{
    color: #777;
    font-size: 11px;
    line-height: 1.4;
  }}

  /* Stats */
  .stats-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }}

  .stat-card {{
    background: #12151e;
    border-radius: 8px;
    padding: 10px 12px;
    border: 1px solid #2a2d3a;
  }}

  .stat-label {{
    font-size: 10px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
  }}

  .stat-value {{
    font-size: 16px;
    font-weight: 600;
    color: #e0e0e0;
  }}

  .stat-value.score-val {{ color: #64b5f6; }}

  /* Tooltip */
  #tooltip {{
    position: absolute;
    background: rgba(20,23,33,0.95);
    border: 1px solid #2d6db4;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    color: #ccc;
    pointer-events: none;
    display: none;
    max-width: 200px;
    z-index: 100;
  }}

  @media (max-width: 768px) {{
    .layout {{ grid-template-columns: 1fr; grid-template-rows: 60vh 1fr; }}
    .sidebar {{ border-left: none; border-top: 1px solid #2a2d3a; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="header-title">{drug} ↔ {protein}</div>
    <div class="header-sub">
      Automated 3D interaction verification — generated by DockExplain
    </div>
  </div>
  <div class="score-badge">
    Binding score: {score_str}
  </div>
</div>

<div class="layout">

  <!-- 3D Viewer -->
  <div id="viewer-container">
    <div id="viewer"></div>

    <div class="viewer-controls">
      <button class="ctrl-btn active" id="btn-surface"
              onclick="toggleSurface()">
        ◼ Surface
      </button>
      <button class="ctrl-btn active" id="btn-cartoon"
              onclick="toggleCartoon()">
        〜 Cartoon
      </button>
      <button class="ctrl-btn" id="btn-sticks"
              onclick="toggleAllSticks()">
        ⊕ All sticks
      </button>
      <button class="ctrl-btn" onclick="resetView()">
        ⟳ Reset view
      </button>
      <button class="ctrl-btn" onclick="zoomLigand()">
        🔍 Zoom ligand
      </button>
    </div>

    <div id="tooltip"></div>
  </div>

  <!-- Sidebar -->
  <div class="sidebar">

    <!-- Stats -->
    <div class="sidebar-section">
      <div class="section-title">Summary</div>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-label">Drug</div>
          <div class="stat-value" style="font-size:13px">{drug}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Interactions</div>
          <div class="stat-value">{len(interactions)}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Score</div>
          <div class="stat-value score-val">
            {score_str.replace(" kcal/mol","")}
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Unit</div>
          <div class="stat-value" style="font-size:12px;color:#64b5f6">
            kcal/mol
          </div>
        </div>
      </div>
    </div>

    <!-- Legend -->
    <div class="sidebar-section">
      <div class="section-title">Interaction colours</div>
      {legend_html}
    </div>

    <!-- Interaction table -->
    <div class="sidebar-section" style="flex:1">
      <div class="section-title">
        Detected interactions ({len(interactions)})
      </div>
      <table>
        <thead>
          <tr>
            <th>Residue</th>
            <th>Type</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          {table_html}
        </tbody>
      </table>
    </div>

  </div>
</div>

<script>
// ── Embed structure data ──────────────────────────────────
const PROTEIN_PDB = `{protein_pdbstr.replace("`", "'")}`;
const LIGAND_SDF  = `{ligand_sdfstr.replace("`", "'")}`;

// ── Initialise 3Dmol viewer ───────────────────────────────
const element   = document.getElementById("viewer");
const config    = {{ backgroundColor: "0x070a0f" }};
const viewer    = $3Dmol.createViewer(element, config);

// State flags
let surfaceOn  = true;
let cartoonOn  = true;
let sticksOn   = false;
let surfaceObj = null;

// ── Add protein ───────────────────────────────────────────
viewer.addModel(PROTEIN_PDB, "pdb");
const proteinModel = viewer.getModel(0);

// Cartoon backbone
proteinModel.setStyle(
  {{}},
  {{ cartoon: {{ color: "spectrum", opacity: 0.85 }} }}
);

// ── Add ligand ────────────────────────────────────────────
viewer.addModel(LIGAND_SDF, "sdf");
const ligandModel = viewer.getModel(1);

ligandModel.setStyle(
  {{}},
  {{
    stick: {{
      radius: 0.25,
      colorscheme: "Jmol",   // C=grey, O=red, N=blue, S=yellow
    }},
    sphere: {{ radius: 0.35, colorscheme: "Jmol" }}
  }}
);

// ── Highlight interacting residues ────────────────────────
function highligthResidue(v, resname, resnum, chain,
                           color, label) {{
  // Sticks for the residue — thicker than background
  proteinModel.setStyle(
    {{ resn: resname, resi: resnum, chain: chain }},
    {{
      stick : {{ radius: 0.2, color: color }},
      sphere: {{ radius: 0.4, color: color }},
    }}
  );

  // Transparent sphere around the residue to show reach
  v.addSphere({{
    center : {{ resi: resnum, chain: chain }},
    radius : 2.8,
    color  : color,
    opacity: 0.08,
  }});

  // Label on the residue
  v.addLabel(
    resname + resnum + " (" + label + ")",
    {{
      position   : {{ resi: resnum, chain: chain }},
      fontSize   : 11,
      fontColor  : color,
      backgroundColor: "black",
      backgroundOpacity: 0.5,
      borderThickness: 0,
      showBackground : true,
    }}
  );
}}

// Apply all interaction highlights
{highlight_js}

// ── Surface (transparent) ─────────────────────────────────
function addSurface() {{
  surfaceObj = viewer.addSurface(
    $3Dmol.SurfaceType.VDW,
    {{ opacity: 0.12, color: "white" }},
    {{ model: proteinModel }}
  );
}}
addSurface();

// ── Zoom to ligand binding site ───────────────────────────
viewer.zoomTo({{ model: ligandModel }});
viewer.zoom(0.8);
viewer.render();

// ── Controls ──────────────────────────────────────────────
function toggleSurface() {{
  surfaceOn = !surfaceOn;
  if (surfaceOn) {{
    addSurface();
  }} else {{
    viewer.removeSurface(surfaceObj);
    surfaceObj = null;
  }}
  viewer.render();
  document.getElementById("btn-surface")
          .classList.toggle("active", surfaceOn);
}}

function toggleCartoon() {{
  cartoonOn = !cartoonOn;
  if (cartoonOn) {{
    proteinModel.setStyle(
      {{}},
      {{ cartoon: {{ color: "spectrum", opacity: 0.85 }} }}
    );
    // Re-apply highlights on top
    {highlight_js}
  }} else {{
    proteinModel.setStyle({{}}, {{}});
    {highlight_js}
  }}
  viewer.render();
  document.getElementById("btn-cartoon")
          .classList.toggle("active", cartoonOn);
}}

function toggleAllSticks() {{
  sticksOn = !sticksOn;
  if (sticksOn) {{
    proteinModel.setStyle(
      {{}},
      {{ stick: {{ radius: 0.1, colorscheme: "Jmol" }} }}
    );
    {highlight_js}
  }} else {{
    proteinModel.setStyle(
      {{}},
      {{ cartoon: {{ color: "spectrum", opacity: 0.85 }} }}
    );
    {highlight_js}
  }}
  viewer.render();
  document.getElementById("btn-sticks")
          .classList.toggle("active", sticksOn);
}}

function resetView() {{
  viewer.zoomTo({{ model: ligandModel }});
  viewer.zoom(0.8);
  viewer.render();
}}

function zoomLigand() {{
  viewer.zoomTo({{ model: ligandModel }});
  viewer.zoom(1.4);
  viewer.render();
}}
</script>
</body>
</html>"""

    # ── Save HTML file ────────────────────────────────────
    out_path = os.path.join(out_dir, f"{safe_drug}_3D_viewer.html")
    with open(out_path, "w") as f:
        f.write(html)

    print(f"✓ 3D viewer saved: {out_path}")
    print(f"  Open in any browser — no installation needed")

    # Auto-open in browser
    try:
        import subprocess
        subprocess.Popen(["open", out_path])   # macOS
        print(f"  Opening in browser...")
    except Exception:
        print(f"  → Manually open: {out_path}")

    return out_path


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    json_path = (
        sys.argv[1] if len(sys.argv) > 1
        else "../data/results/mode_a/erlotinib_results.json"
    )
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        sys.exit(1)

    generate_visualization(json_path)
