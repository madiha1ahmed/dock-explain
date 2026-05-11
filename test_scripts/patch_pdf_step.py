# patch_pdf_step.py
# Run once from your python_scripts/ directory:
#   python patch_pdf_step.py
#
# What it does:
#   1. Checks explanation_pdf_generator.py exists here
#   2. Updates run_dockexplain.py to import it instead of report_generator

import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Check explanation_pdf_generator.py is present ──────────────────────────
gen_path = os.path.join(HERE, "explanation_pdf_generator.py")
if not os.path.exists(gen_path):
    print("✗  explanation_pdf_generator.py not found in", HERE)
    print("   Copy it here first, then re-run this script.")
    sys.exit(1)
print(f"✓  explanation_pdf_generator.py found")

# ── Patch run_dockexplain.py ────────────────────────────────────────────────
run_path = os.path.join(HERE, "run_dockexplain.py")
if not os.path.exists(run_path):
    print("✗  run_dockexplain.py not found in", HERE)
    sys.exit(1)

with open(run_path) as f:
    src = f.read()

original = src  # keep for comparison

# 1. Replace the PDF generator import block (handles several variants)
for old_import in [
    # variant A — report_generator
    ("    # ── PDF report generator ───────────────────────────\n"
     "    try:\n"
     "        from report_generator import generate_report\n"
     "        components[\"generate_report\"] = generate_report\n"
     "    except ImportError as e:\n"
     "        errors.append(f\"report_generator: {e}\")"),
    # variant B — already partially patched
    ("    # ── Explanation PDF generator ──────────────────────\n"
     "    try:\n"
     "        from report_generator import generate_report\n"
     "        components[\"generate_report\"] = generate_report\n"
     "    except ImportError as e:\n"
     "        errors.append(f\"report_generator: {e}\")"),
]:
    if old_import in src:
        src = src.replace(old_import,
            "    # ── Explanation PDF generator ──────────────────────\n"
            "    try:\n"
            "        from explanation_pdf_generator import generate_explanation_pdf\n"
            "        components[\"generate_report\"] = generate_explanation_pdf\n"
            "    except ImportError as e:\n"
            "        errors.append(f\"explanation_pdf_generator: {e}\")"
        )
        print("✓  Import block updated")
        break
else:
    # Check if already correct
    if "explanation_pdf_generator" in src and "generate_explanation_pdf" in src:
        print("✓  Import block already correct")
    else:
        print("⚠  Could not find import block to patch — manual edit needed:")
        print("   In import_components(), replace:")
        print("     from report_generator import generate_report")
        print("   with:")
        print("     from explanation_pdf_generator import generate_explanation_pdf")
        print("     components[\"generate_report\"] = generate_explanation_pdf")

# 2. Replace the Step 6 PDF call to pass explanation JSON, not results JSON
for old_step in [
    # old report_generator call pattern
    ("    else:\n"
     "        result = run_step(\n"
     "            \"pdf\",\n"
     "            components[\"generate_report\"],\n"
     "            json_path,\n"
     "        )"),
    # partially patched variant
    ("        result = run_step(\n"
     "            \"pdf\",\n"
     "            components[\"generate_report\"],\n"
     "            json_path,\n"
     "        )"),
]:
    if old_step in src:
        src = src.replace(old_step,
            "    else:\n"
            "        explanation_json = os.path.join(\n"
            "            out_dir, f\"{safe_drug}_explanation.json\"\n"
            "        )\n"
            "        pdf_input = (explanation_json\n"
            "                     if os.path.exists(explanation_json)\n"
            "                     else json_path)\n"
            "        result = run_step(\n"
            "            \"pdf\",\n"
            "            components[\"generate_report\"],\n"
            "            pdf_input,\n"
            "        )"
        )
        print("✓  Step 6 call updated to use explanation JSON")
        break
else:
    if "explanation_json" in src:
        print("✓  Step 6 call already correct")
    else:
        print("⚠  Could not patch Step 6 call — it will use results JSON as fallback")

# 3. Write back only if changed
if src != original:
    backup = run_path + ".bak"
    with open(backup, "w") as f:
        f.write(original)
    with open(run_path, "w") as f:
        f.write(src)
    print(f"✓  run_dockexplain.py updated  (backup: {backup})")
else:
    print("✓  run_dockexplain.py already up to date — no changes needed")

print("\nDone. Run your pipeline again and the PDF step should work.")