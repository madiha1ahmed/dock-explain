#!/usr/bin/env python3
"""
Debug script to test pocket_predictor.py's fpocket call
Shows exactly what arguments are being passed and what fpocket returns
"""

import subprocess
import tempfile
import os
import sys
from pathlib import Path

def debug_fpocket_call(pdb_file):
    """
    Replicate exactly what pocket_predictor.py does
    """
    print(f"\n📋 Testing fpocket call from pocket_predictor.py")
    print(f"   Input PDB: {pdb_file}")
    
    if not os.path.exists(pdb_file):
        print(f"   ✗ File not found: {pdb_file}")
        return False
    
    # Create temp directory like pocket_predictor does
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_stem = Path(pdb_file).stem  # e.g., "8RZX"
        
        # This is what pocket_predictor.py does:
        cmd = ["fpocket", "-f", pdb_file, "-o", tmpdir]
        
        print(f"\n   Command: {' '.join(cmd)}")
        print(f"   Working directory: {tmpdir}")
        
        try:
            # Run fpocket with verbose output
            print(f"\n   Running fpocket...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=tmpdir  # Run in the temp directory
            )
            
            print(f"\n   Return code: {result.returncode}")
            
            if result.stdout:
                print(f"\n   STDOUT:")
                for line in result.stdout.split('\n')[:30]:
                    if line.strip():
                        print(f"     {line}")
            
            if result.stderr:
                print(f"\n   STDERR:")
                for line in result.stderr.split('\n')[:30]:
                    if line.strip():
                        print(f"     {line}")
            
            # Check what was created
            print(f"\n   Files in {tmpdir}:")
            for item in Path(tmpdir).rglob("*"):
                rel_path = item.relative_to(tmpdir)
                print(f"     {rel_path}")
            
            # Check for expected output directory
            expected_dir = Path(tmpdir) / f"{pdb_stem}_out"
            if expected_dir.exists():
                print(f"\n   ✓ SUCCESS: {expected_dir} was created")
                # List contents
                contents = list(expected_dir.glob("*"))[:10]
                print(f"     Contents: {[c.name for c in contents]}")
                return True
            else:
                print(f"\n   ✗ FAIL: Expected directory not created: {expected_dir}")
                
                # Try alternate patterns that fpocket might use
                print(f"\n   Checking for alternate output patterns...")
                patterns = [
                    f"{pdb_stem.lower()}_out",
                    f"{pdb_stem}_pockets",
                    "*_out",
                    "*_pockets",
                ]
                found_any = False
                for pattern in patterns:
                    matches = list(Path(tmpdir).glob(pattern))
                    if matches:
                        print(f"     Found: {[m.name for m in matches]}")
                        found_any = True
                
                if not found_any:
                    print(f"     No alternate output directories found")
                
                return False
                
        except subprocess.TimeoutExpired:
            print(f"   ✗ fpocket timed out (>60s)")
            return False
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False

def show_fpocket_info():
    """Show fpocket version and location"""
    print("\n📋 fpocket Information")
    
    try:
        result = subprocess.run(["which", "fpocket"], capture_output=True, text=True)
        print(f"   Location: {result.stdout.strip()}")
    except:
        pass
    
    try:
        result = subprocess.run(["fpocket", "-v"], capture_output=True, text=True, timeout=5)
        version = result.stdout.strip() or result.stderr.strip()
        if version:
            print(f"   Version: {version}")
    except:
        pass

def test_simple_call():
    """Test a very simple fpocket call"""
    print("\n📋 Testing simple fpocket call...")
    
    # Create a minimal test PDB
    minimal_pdb = "/tmp/minimal_test.pdb"
    with open(minimal_pdb, 'w') as f:
        f.write("""ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.440   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.221   2.440   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.988  -0.760  -1.220  1.00  0.00           C
END
""")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["fpocket", "-f", minimal_pdb, "-o", tmpdir]
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            print(f"   Return code: {result.returncode}")
            
            # Check output
            expected = Path(tmpdir) / "minimal_test_out"
            if expected.exists():
                print(f"   ✓ SUCCESS: Output directory created")
                return True
            else:
                print(f"   ✗ FAIL: Output directory not created")
                print(f"   Files: {list(Path(tmpdir).glob('*'))}")
                return False
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False

def main():
    print("="*70)
    print("fpocket Call Debugger")
    print("="*70)
    
    show_fpocket_info()
    
    # Test 1: Simple call
    print("\n" + "="*70)
    print("TEST 1: Simple PDB")
    print("="*70)
    test_simple_call()
    
    # Test 2: User's PDB
    print("\n" + "="*70)
    print("TEST 2: Your 8RZX.pdb")
    print("="*70)
    
    # Try to find 8RZX.pdb
    possible_paths = [
        "./8RZX.pdb",
        "../data/8RZX.pdb",
        "~/Downloads/8RZX.pdb",
        "/tmp/8RZX.pdb",
        os.path.expanduser("~/Downloads/8RZX.pdb"),
    ]
    
    pdb_path = None
    for path in possible_paths:
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            pdb_path = expanded
            break
    
    if pdb_path:
        debug_fpocket_call(pdb_path)
    else:
        print("\n   ⊘ 8RZX.pdb not found in standard locations")
        print("     To test with your file, run manually:")
        print("     python FPOCKET_CALL_DEBUG.py /path/to/8RZX.pdb")
    
    # Summary
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
If fpocket works but DockExplain still fails:
1. The issue is in pocket_predictor.py's code
2. Check: Does the temp directory persist after fpocket runs?
3. Check: Are file permissions correct?
4. Check: Is there a race condition?

If fpocket doesn't work at all:
1. Try: conda install -c bioconda fpocket=4.1 --force-reinstall
2. Try: conda update fpocket
3. Last resort: Use OpenBabel's pocket prediction instead

For now, simplest fix:
- Disable fpocket in pocket_predictor.py
- Always use geometric center fallback
- This still works for docking (just less optimized)
""")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdb_file = sys.argv[1]
        debug_fpocket_call(pdb_file)
    else:
        main()