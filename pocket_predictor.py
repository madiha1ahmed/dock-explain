# pocket_predictor.py
# DockExplain — Unified binding pocket prediction
#
# Supports:
#   • fpocket — fast, installed, decent accuracy
#   • P2Rank — slower, better accuracy, requires separate installation
#
# Usage:
#   from pocket_predictor import predict_pockets
#   pockets = predict_pockets(
#       protein_pdb='../data/1ABC.pdb',
#       method='p2rank',  # or 'fpocket'
#       verbose=True
#   )
#
# Install P2Rank:
#   wget https://github.com/rdk/p2rank/releases/download/2.4.1/p2rank_2.4.1.tar.gz
#   tar xzf p2rank_2.4.1.tar.gz
#   export PATH=$PATH:/path/to/p2rank/distro/bin

import os
import json
import subprocess
import tempfile
import shutil
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Pocket:
    """Unified representation of a binding pocket."""
    pocket_id: int
    center: tuple[float, float, float]
    score: float  # 0-1 scale (higher = more druggable)
    volume: float  # Ų
    residues: list[str]
    method: str  # 'fpocket' or 'p2rank'
    confidence: float  # 0-1, method-specific
    
    def to_dict(self) -> dict:
        return {
            'pocket_id': self.pocket_id,
            'center': list(self.center),
            'score': self.score,
            'volume': self.volume,
            'residues': self.residues,
            'method': self.method,
            'confidence': self.confidence,
        }


# ════════════════════════════════════════════════════════════════════════════
# FPOCKET BACKEND
# ════════════════════════════════════════════════════════════════════════════

def _check_fpocket_installed() -> bool:
    """Check if fpocket is available in PATH."""
    try:
        result = subprocess.run(
            ['fpocket', '--help'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_fpocket(pdb_file: str, tmpdir: str, verbose: bool = True) -> dict:
    """
    Run fpocket with workaround for the -o output flag bug.
    
    BUG: fpocket ignores the -o output flag and creates output in the 
    same directory as the input PDB. 
    
    WORKAROUND: Copy the PDB to the temp directory, run fpocket there.
    """
    import subprocess
    import shutil
    from pathlib import Path
    
    if verbose:
        print(f"  [fpocket] Running fpocket...")
    
    # Get the PDB filename
    pdb_name = Path(pdb_file).name  # e.g., "8RZX.pdb"
    pdb_stem = Path(pdb_file).stem  # e.g., "8RZX"
    
    # Copy PDB to temp directory
    pdb_in_tmpdir = Path(tmpdir) / pdb_name
    try:
        shutil.copy(pdb_file, pdb_in_tmpdir)
    except Exception as e:
        if verbose:
            print(f"  [fpocket] ✗ Failed to copy PDB: {e}")
        return None
    
    # Run fpocket IN the temp directory
    try:
        cmd = ["fpocket", "-f", str(pdb_in_tmpdir), "-o", tmpdir]
        
        if verbose:
            print(f"  [fpocket] Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmpdir  # Run fpocket FROM the temp directory
        )
        
        if result.returncode != 0 and verbose:
            print(f"  [fpocket] Warning: return code {result.returncode}")
        
        # Check for output directory
        output_dir = Path(tmpdir) / f"{pdb_stem}_out"
        
        if not output_dir.exists():
            if verbose:
                print(f"  [fpocket] ✗ Output directory not found: {output_dir}")
            return None
        
        if verbose:
            print(f"  [fpocket] ✓ Output directory created: {output_dir}")
        
        # Parse fpocket results
        try:
            pockets = _parse_fpocket_output(output_dir, verbose=verbose)
            return pockets
        except Exception as e:
            if verbose:
                print(f"  [fpocket] ✗ Failed to parse output: {e}")
            return None
            
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"  [fpocket] ✗ fpocket timed out (>120s)")
        return None
    except Exception as e:
        if verbose:
            print(f"  [fpocket] ✗ Error running fpocket: {e}")
        return None

# CORRECT _parse_fpocket_output
# Creates Pocket dataclass instances with all required fields

def _parse_fpocket_output(output_dir: str, verbose: bool = True):
    """
    Parse fpocket pockets and create Pocket dataclass instances.
    
    Extracts:
    - pocket_id: sequential number (0, 1, 2, ...)
    - center: center of mass from PDB atoms
    - score: based on number of atoms
    - volume: estimated from atom count
    - residues: extracted from PDB atom records
    - method: 'fpocket'
    - confidence: based on pocket rank (higher rank = lower confidence)
    """
    from pathlib import Path
    
    try:
        output_dir = Path(output_dir)
        pockets_dir = output_dir / "pockets"
        
        if not pockets_dir.exists():
            if verbose:
                print(f"  [fpocket] ✗ pockets/ directory not found")
            return None
        
        pocket_files = sorted(pockets_dir.glob("pocket*_atm.pdb"))
        
        if not pocket_files:
            if verbose:
                print(f"  [fpocket] ✗ No pocket files found")
            return None
        
        pockets = []
        
        for pocket_id, pocket_file in enumerate(pocket_files):
            try:
                # Parse PDB file
                atoms = []
                residues = set()
                coordinates = []
                
                with open(pocket_file) as f:
                    for line in f:
                        if line.startswith("ATOM"):
                            atoms.append(line.strip())
                            
                            # Extract coordinates (columns 30-38, 38-46, 46-54)
                            try:
                                x = float(line[30:38])
                                y = float(line[38:46])
                                z = float(line[46:54])
                                coordinates.append((x, y, z))
                            except ValueError:
                                pass
                            
                            # Extract residue info (columns 17-26)
                            try:
                                res_info = line[17:26].strip()  # e.g., "ALA A 123"
                                residues.add(res_info)
                            except:
                                pass
                
                if len(atoms) == 0:
                    if verbose:
                        print(f"  [fpocket] ⚠ Pocket {pocket_id} has no atoms")
                    continue
                
                # Calculate center of mass
                if len(coordinates) > 0:
                    center_x = sum(c[0] for c in coordinates) / len(coordinates)
                    center_y = sum(c[1] for c in coordinates) / len(coordinates)
                    center_z = sum(c[2] for c in coordinates) / len(coordinates)
                    center = (center_x, center_y, center_z)
                else:
                    center = (0.0, 0.0, 0.0)
                
                # Estimate volume (rough approximation)
                # Assume ~50 Ų per atom (typical for protein pockets)
                volume = len(atoms) * 50.0
                
                # Calculate score based on atom count
                # More atoms = higher score (more druggable pocket)
                atom_count = len(atoms)
                score = min(1.0, atom_count / 150.0)  # Normalize to 0-1
                
                # Confidence decreases with rank (first pocket is most confident)
                confidence = max(0.1, 1.0 - (pocket_id * 0.15))
                
                # Create Pocket object
                pocket = Pocket(
                    pocket_id=pocket_id,
                    center=center,
                    score=score,
                    volume=volume,
                    residues=list(residues),
                    method='fpocket',
                    confidence=confidence
                )
                
                pockets.append(pocket)
                
                if verbose:
                    print(f"  [fpocket] ✓ Pocket {pocket_id}: {len(atoms)} atoms, "
                          f"score={score:.2f}, vol={volume:.1f}Ų, conf={confidence:.2f}")
                
            except Exception as e:
                if verbose:
                    print(f"  [fpocket] ⚠ Error parsing {pocket_file.name}: {e}")
                continue
        
        if not pockets:
            if verbose:
                print(f"  [fpocket] ✗ No valid pockets created")
            return None
        
        if verbose:
            print(f"  [fpocket] ✓ Successfully created {len(pockets)} Pocket objects")
        
        return pockets
        
    except Exception as e:
        if verbose:
            print(f"  [fpocket] ✗ Exception: {e}")
            import traceback
            traceback.print_exc()
        return None

def _parse_fpocket_pockets(fpocket_dir: str, verbose: bool = True) -> list[Pocket]:
    """
    Parse fpocket output into Pocket objects.
    
    fpocket creates:
        pockets/pocket1/
        pockets/pocket2/
        ...
        
    Each pocket has:
        pocket_info.txt — score, residues, volume
        pocket.py — center coordinates
    """
    def log(msg):
        if verbose:
            print(f"  [fpocket] {msg}")
    
    pockets_dir = os.path.join(fpocket_dir, 'pockets')
    if not os.path.exists(pockets_dir):
        log(f"✗ No pockets directory found")
        return []
    
    pockets = []
    pocket_dirs = sorted([
        d for d in os.listdir(pockets_dir)
        if d.startswith('pocket') and os.path.isdir(os.path.join(pockets_dir, d))
    ], key=lambda x: int(x.replace('pocket', '')) if x.replace('pocket', '').isdigit() else 999)
    
    for pocket_dir in pocket_dirs:
        pocket_path = os.path.join(pockets_dir, pocket_dir)
        pocket_num = int(pocket_dir.replace('pocket', ''))
        
        # Parse pocket_info.txt
        info_path = os.path.join(pocket_path, 'pocket_info.txt')
        score = 0.5
        volume = 0.0
        residues = []
        
        if os.path.exists(info_path):
            try:
                with open(info_path) as f:
                    for line in f:
                        if 'Score' in line or 'score' in line:
                            parts = line.split('=')
                            if len(parts) > 1:
                                try:
                                    score = float(parts[1].split()[0])
                                except ValueError:
                                    pass
                        if 'Volume' in line or 'volume' in line:
                            parts = line.split('=')
                            if len(parts) > 1:
                                try:
                                    volume = float(parts[1].split()[0])
                                except ValueError:
                                    pass
            except Exception:
                pass
        
        # Parse pocket.py for center coordinates
        pocket_py_path = os.path.join(pocket_path, 'pocket.py')
        center = None
        
        if os.path.exists(pocket_py_path):
            try:
                with open(pocket_py_path) as f:
                    for line in f:
                        if 'pocket_center' in line:
                            # pocket_center = [x, y, z]
                            match = line.split('[')
                            if len(match) > 1:
                                coords_str = match[1].split(']')[0]
                                coords = [float(x.strip()) for x in coords_str.split(',')]
                                if len(coords) == 3:
                                    center = tuple(coords)
                                    break
            except Exception:
                pass
        
        # Fallback: calculate center from residues in pocket_info.txt
        if center is None:
            center = (0.0, 0.0, 0.0)
        
        # Normalize score to 0-1 if needed
        if score > 1.0:
            score = min(score / 100.0, 1.0)
        
        pockets.append(Pocket(
            pocket_id=pocket_num,
            center=center,
            score=min(score, 1.0),  # Ensure 0-1
            volume=volume,
            residues=residues,
            method='fpocket',
            confidence=score,
        ))
    
    if pockets:
        log(f"✓ Parsed {len(pockets)} pockets")
    
    return pockets


# ════════════════════════════════════════════════════════════════════════════
# P2RANK BACKEND
# ════════════════════════════════════════════════════════════════════════════

def _check_p2rank_installed() -> bool:
    """Check if P2Rank is available in PATH."""
    try:
        result = subprocess.run(
            ['p2rank', '-version'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_p2rank(protein_pdb: str, tmpdir: str, verbose: bool = True) -> str:
    """
    Run P2Rank on protein and return output directory.
    
    Args:
        protein_pdb : path to cleaned PDB file
        tmpdir      : temporary directory for outputs
        verbose     : print progress
    
    Returns:
        Path to P2Rank output directory
    """
    def log(msg):
        if verbose:
            print(f"  [p2rank] {msg}")
    
    log(f"Running P2Rank...")
    
    try:
        output_dir = os.path.join(tmpdir, 'p2rank_out')
        os.makedirs(output_dir, exist_ok=True)
        
        result = subprocess.run(
            [
                'p2rank',
                'predict',
                '-f', protein_pdb,
                '-o', output_dir
            ],
            capture_output=True,
            text=True,
            timeout=600  # P2Rank can be slow
        )
        
        if result.returncode != 0:
            log(f"✗ P2Rank failed: {result.stderr[:200]}")
            return None
        
        log(f"✓ P2Rank output: {output_dir}")
        return output_dir
        
    except subprocess.TimeoutExpired:
        log(f"✗ P2Rank timeout (>600s)")
        return None
    except Exception as e:
        log(f"✗ P2Rank error: {str(e)[:200]}")
        return None


def _parse_p2rank_pockets(output_dir: str, verbose: bool = True) -> list[Pocket]:
    """
    Parse P2Rank output into Pocket objects.
    
    P2Rank creates:
        predictions.txt or {protein}_predictions.txt
        {protein}_predictions_pymol.py
    """
    def log(msg):
        if verbose:
            print(f"  [p2rank] {msg}")
    
    # Find predictions file
    pred_files = [
        f for f in os.listdir(output_dir)
        if 'predictions' in f and f.endswith('.txt')
    ]
    
    if not pred_files:
        log(f"✗ No predictions file found")
        return []
    
    pred_file = os.path.join(output_dir, pred_files[0])
    pockets = []
    
    try:
        with open(pred_file) as f:
            lines = f.readlines()
        
        # Skip header line(s)
        header_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('rank') or line.startswith('#'):
                header_idx = i + 1
                break
        
        pocket_idx = 1
        for line in lines[header_idx:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # P2Rank format:
            # rank  name  score  center_x  center_y  center_z  size  ...
            parts = line.split()
            if len(parts) < 7:
                continue
            
            try:
                score = float(parts[2])  # P2Rank score (higher = better)
                center_x = float(parts[3])
                center_y = float(parts[4])
                center_z = float(parts[5])
                size = int(parts[6])  # Number of atoms
                
                # Normalize P2Rank score (typically 0-100) to 0-1
                if score > 1.0:
                    score_norm = min(score / 100.0, 1.0)
                else:
                    score_norm = score
                
                pockets.append(Pocket(
                    pocket_id=pocket_idx,
                    center=(center_x, center_y, center_z),
                    score=score_norm,
                    volume=float(size),  # Use atom count as proxy for volume
                    residues=[],
                    method='p2rank',
                    confidence=score_norm,
                ))
                pocket_idx += 1
            except (ValueError, IndexError):
                continue
        
        if pockets:
            log(f"✓ Parsed {len(pockets)} pockets")
        
        return pockets
        
    except Exception as e:
        log(f"✗ Error parsing predictions: {str(e)[:200]}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# UNIFIED INTERFACE
# ════════════════════════════════════════════════════════════════════════════

def predict_pockets(
    protein_pdb: str,
    method: str = 'auto',
    n_pockets: int = None,
    verbose: bool = True,
) -> list[Pocket]:
    """
    Predict binding pockets in an apo protein using fpocket or P2Rank.
    
    Args:
        protein_pdb : path to cleaned PDB file
        method      : 'fpocket', 'p2rank', or 'auto' (best available)
        n_pockets   : limit results to top N pockets (None = all)
        verbose     : print progress
    
    Returns:
        List of Pocket objects sorted by score (best first)
    
    Raises:
        RuntimeError if no pocket prediction method is available
    """
    def log(msg):
        if verbose:
            print(f"  [pocket_predictor] {msg}")
    
    if not os.path.exists(protein_pdb):
        raise FileNotFoundError(f"PDB file not found: {protein_pdb}")
    
    # Determine method
    if method == 'auto':
        p2rank_ok = _check_p2rank_installed()
        fpocket_ok = _check_fpocket_installed()
        
        if p2rank_ok:
            log("Auto-selected P2Rank (better accuracy)")
            method = 'p2rank'
        elif fpocket_ok:
            log("Auto-selected fpocket (P2Rank not available)")
            method = 'fpocket'
        else:
            raise RuntimeError(
                "No pocket prediction method available. "
                "Install fpocket or P2Rank:\n"
                "  fpocket: conda install -c bioconda fpocket\n"
                "  p2rank: https://github.com/rdk/p2rank/releases"
            )
    
    # Create temporary directory for outputs
    tmpdir = tempfile.mkdtemp(prefix='pockets_')
    
    try:
        if method == 'fpocket':
            if not _check_fpocket_installed():
                raise RuntimeError(
                    "fpocket not found in PATH. Install with:\n"
                    "  conda install -c bioconda fpocket"
                )
            
            pockets = _run_fpocket(protein_pdb, tmpdir, verbose)  # ← Returns list directly
            if not pockets:
                raise RuntimeError("fpocket failed to run")
            # Don't parse again - pockets are already parsed!
        
        elif method == 'p2rank':
            if not _check_p2rank_installed():
                raise RuntimeError(
                    "P2Rank not found in PATH. Install from:\n"
                    "  https://github.com/rdk/p2rank/releases"
                )
            
            p2rank_dir = _run_p2rank(protein_pdb, tmpdir, verbose)
            if not p2rank_dir:
                raise RuntimeError("P2Rank failed to run")
            
            pockets = _parse_p2rank_pockets(p2rank_dir, verbose)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Sort by score (best first)
        pockets = sorted(pockets, key=lambda p: p.score, reverse=True)
        
        # Limit to top N if requested
        if n_pockets is not None:
            pockets = pockets[:n_pockets]
        
        if not pockets:
            log(f"✗ No pockets detected")
        else:
            log(f"✓ {len(pockets)} pockets detected")
            for p in pockets[:3]:
                log(f"  Pocket {p.pocket_id}: score={p.score:.2f}, "
                    f"center=({p.center[0]:.1f}, {p.center[1]:.1f}, {p.center[2]:.1f})")
        
        return pockets
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


def compare_methods(
    protein_pdb: str,
    verbose: bool = True,
) -> dict:
    """
    Run both fpocket and P2Rank and compare results.
    
    Returns:
        {
            'fpocket': [Pocket, ...],
            'p2rank': [Pocket, ...],
            'comparison': analysis of differences
        }
    """
    def log(msg):
        if verbose:
            print(f"  [compare] {msg}")
    
    log(f"Comparing pocket prediction methods...")
    
    results = {'fpocket': [], 'p2rank': [], 'comparison': {}}
    
    # Try fpocket
    if _check_fpocket_installed():
        try:
            results['fpocket'] = predict_pockets(
                protein_pdb, method='fpocket', verbose=verbose
            )
            log(f"✓ fpocket: {len(results['fpocket'])} pockets")
        except Exception as e:
            log(f"✗ fpocket failed: {str(e)[:100]}")
    else:
        log(f"⊘ fpocket not installed")
    
    # Try P2Rank
    if _check_p2rank_installed():
        try:
            results['p2rank'] = predict_pockets(
                protein_pdb, method='p2rank', verbose=verbose
            )
            log(f"✓ P2Rank: {len(results['p2rank'])} pockets")
        except Exception as e:
            log(f"✗ P2Rank failed: {str(e)[:100]}")
    else:
        log(f"⊘ P2Rank not installed")
    
    # Compare if both available
    if results['fpocket'] and results['p2rank']:
        fp_best = results['fpocket'][0]
        p2_best = results['p2rank'][0]
        
        dist = np.sqrt(
            (fp_best.center[0] - p2_best.center[0])**2 +
            (fp_best.center[1] - p2_best.center[1])**2 +
            (fp_best.center[2] - p2_best.center[2])**2
        )
        
        results['comparison'] = {
            'fpocket_best_score': fp_best.score,
            'p2rank_best_score': p2_best.score,
            'center_distance_angstrom': dist,
            'same_pocket': dist < 5.0,  # < 5Å = likely same pocket
        }
        
        log(f"Comparison:")
        log(f"  fpocket best score: {fp_best.score:.3f}")
        log(f"  P2Rank best score: {p2_best.score:.3f}")
        log(f"  Center distance: {dist:.1f} Å")
        log(f"  Agree on pocket: {'Yes' if dist < 5.0 else 'No'}")
    
    return results


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified pocket prediction (fpocket or P2Rank)"
    )
    parser.add_argument("pdb", help="Path to PDB file")
    parser.add_argument(
        "--method", default="auto",
        choices=['fpocket', 'p2rank', 'auto'],
        help="Pocket prediction method"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare fpocket and P2Rank results"
    )
    parser.add_argument(
        "--output", help="Save results to JSON"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.pdb):
        print(f"Error: {args.pdb} not found")
        exit(1)
    
    print(f"\n{'═'*60}")
    print(f"  DockExplain — Pocket Prediction")
    print(f"  PDB: {args.pdb}")
    if args.compare:
        print(f"  Mode: Compare fpocket vs P2Rank")
    else:
        print(f"  Method: {args.method}")
    print(f"{'═'*60}\n")
    
    if args.compare:
        results = compare_methods(args.pdb, verbose=True)
    else:
        pockets = predict_pockets(args.pdb, method=args.method, verbose=True)
        results = {args.method: [p.to_dict() for p in pockets]}
    
    if args.output:
        output = {}
        for method, pockets_list in results.items():
            if isinstance(pockets_list, list) and pockets_list:
                if isinstance(pockets_list[0], Pocket):
                    output[method] = [p.to_dict() for p in pockets_list]
                else:
                    output[method] = pockets_list
            elif isinstance(pockets_list, dict):
                output[method] = pockets_list
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved: {args.output}")