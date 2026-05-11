#!/usr/bin/env python3
"""
fpocket Diagnostic Tool
Runs fpocket independently to identify the issue
"""

import subprocess
import tempfile
import os
import sys
from pathlib import Path

def test_fpocket_installed():
    """Test if fpocket is in PATH"""
    print("\n[TEST 1] Is fpocket installed?")
    try:
        result = subprocess.run(["fpocket", "--help"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("  ✓ fpocket is installed and accessible")
            return True
        else:
            print(f"  ✗ fpocket exists but errored: {result.stderr[:100]}")
            return False
    except FileNotFoundError:
        print("  ✗ fpocket not found in PATH")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_fpocket_version():
    """Check fpocket version"""
    print("\n[TEST 2] fpocket version?")
    try:
        result = subprocess.run(["fpocket", "--version"], capture_output=True, text=True, timeout=5)
        version = result.stdout.strip() or result.stderr.strip()
        print(f"  ✓ Version: {version}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_fpocket_with_known_pdb():
    """Test fpocket with a known working PDB"""
    print("\n[TEST 3] Run fpocket on a test PDB (1M17 - known to work)?")
    
    # Download a known working PDB
    test_pdb = "/tmp/1M17_test.pdb"
    
    # Check if it exists, if not create a minimal one
    if not os.path.exists(test_pdb):
        print("  ℹ Downloading test PDB 1M17...")
        try:
            os.system(f"curl -s https://files.rcsb.org/download/1M17.pdb -o {test_pdb}")
            if not os.path.exists(test_pdb) or os.path.getsize(test_pdb) < 1000:
                print("  ⚠ Download failed, using minimal test PDB instead")
                # Use 8RZX as fallback
                test_pdb = None
        except:
            print("  ⚠ Could not download, will test with your 8RZX instead")
            test_pdb = None
    
    if test_pdb and os.path.exists(test_pdb):
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"  ℹ Running: fpocket -f {test_pdb} -o {tmpdir}")
            try:
                result = subprocess.run(
                    ["fpocket", "-f", test_pdb, "-o", tmpdir],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                print(f"  Return code: {result.returncode}")
                if result.stdout:
                    print(f"  STDOUT: {result.stdout[:300]}")
                if result.stderr:
                    print(f"  STDERR: {result.stderr[:300]}")
                
                # Check for output
                output_dir = Path(tmpdir) / "1M17_out"
                if output_dir.exists():
                    print(f"  ✓ Output directory created: {output_dir}")
                    return True
                else:
                    print(f"  ✗ Output directory NOT created (expected: {output_dir})")
                    # List what was created
                    created = list(Path(tmpdir).glob("*"))
                    print(f"  Files created: {created}")
                    return False
            except Exception as e:
                print(f"  ✗ Error running fpocket: {e}")
                return False
    else:
        print("  ⊘ Skipping (no test PDB available)")
        return None

def test_fpocket_with_your_pdb():
    """Test fpocket with the actual 8RZX.pdb"""
    print("\n[TEST 4] Run fpocket on YOUR 8RZX.pdb?")
    
    # Try to find 8RZX.pdb
    possible_paths = [
        "./8RZX.pdb",
        "../data/8RZX.pdb",
        "~/Downloads/8RZX.pdb",
        "/tmp/8RZX.pdb",
    ]
    
    pdb_path = None
    for path in possible_paths:
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            pdb_path = expanded
            break
    
    if not pdb_path:
        print("  ⊘ 8RZX.pdb not found in common locations")
        print("     Try manually: fpocket -f /path/to/8RZX.pdb -o /tmp/fpocket_test")
        return None
    
    print(f"  ℹ Found: {pdb_path}")
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  ℹ Running: fpocket -f {pdb_path} -o {tmpdir}")
        try:
            result = subprocess.run(
                ["fpocket", "-f", pdb_path, "-o", tmpdir],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            print(f"  Return code: {result.returncode}")
            if result.stdout:
                print(f"  STDOUT:\n{result.stdout[:500]}")
            if result.stderr:
                print(f"  STDERR:\n{result.stderr[:500]}")
            
            # Check for output
            output_base = Path(tmpdir) / "8RZX_out"
            if output_base.exists():
                print(f"  ✓ Output directory created: {output_base}")
                print(f"  Files: {list(output_base.glob('*'))[:5]}")
                return True
            else:
                print(f"  ✗ Output directory NOT created (expected: {output_base})")
                created = list(Path(tmpdir).glob("*"))
                print(f"  What WAS created: {created}")
                return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

def show_fpocket_help():
    """Show fpocket command-line options"""
    print("\n[INFO] fpocket command-line options:")
    try:
        result = subprocess.run(["fpocket", "-h"], capture_output=True, text=True, timeout=5)
        print(result.stdout[:1000])
    except:
        print("  (Unable to retrieve)")

def main():
    print("="*70)
    print("fpocket Diagnostic Tool")
    print("="*70)
    
    # Run tests
    test1 = test_fpocket_installed()
    if not test1:
        print("\n❌ fpocket is not properly installed!")
        print("   Fix: conda install -c bioconda fpocket")
        sys.exit(1)
    
    test2 = test_fpocket_version()
    test3 = test_fpocket_with_known_pdb()
    test4 = test_fpocket_with_your_pdb()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if test3 is False:
        print("\n❌ fpocket fails even on test PDFs!")
        print("   This suggests a conda/macOS issue")
        print("   Try:")
        print("   1. conda remove fpocket -y")
        print("   2. conda install -c bioconda fpocket=4.1")
        print("   3. Run this diagnostic again")
        
    elif test4 is False:
        print("\n⚠️  fpocket works on test PDFs but fails on 8RZX.pdb")
        print("   The PDB file might be:")
        print("   - Malformed or missing atoms")
        print("   - An unusual structure type")
        print("   - Missing hydrogens")
        print("\n   Try:")
        print("   1. Check if 8RZX.pdb is valid:")
        print("      python -c \"from Bio import PDB; p = PDB.PDBParser(); p.get_structure('', '8RZX.pdb')\"")
        print("   2. Run fpocket manually:")
        print("      fpocket -f 8RZX.pdb -o /tmp/fpocket_test")
        print("      ls -la /tmp/fpocket_test/")
        
    elif test3 is None:
        print("\n⊘ Could not test with known PDB")
        print("  Please manually run:")
        print("    fpocket -f /path/to/your/pdb.pdb -o /tmp/test_fpocket")
        print("    ls -la /tmp/test_fpocket/")
    
    else:
        print("\n✓ fpocket seems to be working correctly!")
        print("  The issue might be in how DockExplain is calling fpocket")
        print("  See: FPOCKET_CALL_DEBUG.py")
    
    # Show help
    show_fpocket_help()

if __name__ == "__main__":
    main()