# fix_extract_pose.py — properly extract best pose
# Run: python fix_extract_pose.py

INPUT  = "/Users/imad/Desktop/Gemma4Good/data/docking_results.pdbqt"
OUTPUT = "/Users/imad/Desktop/Gemma4Good/data/docking_best.pdbqt"

with open(INPUT) as f:
    content = f.read()

# Find MODEL 1 block
start = content.find("MODEL        1")
end   = content.find("ENDMDL", start) + len("ENDMDL")

if start == -1 or end == -1:
    raise RuntimeError("MODEL 1 not found in docking_results.pdbqt")

best_pose = content[start:end] + "\nEND\n"  # END line is required

with open(OUTPUT, "w") as f:
    f.write(best_pose)

# Verify
with open(OUTPUT) as f:
    lines = f.readlines()

print(f"Extracted {len(lines)} lines to {OUTPUT}")
print(f"First line : {lines[0].rstrip()}")
print(f"Last line  : {lines[-1].rstrip()}")
print("✓ Done — now run: python run_prolif.py")