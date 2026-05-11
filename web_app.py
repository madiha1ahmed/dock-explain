"""
DockExplain Flask Frontend — live logs + ZIP download

Place this file inside your python_scripts/ folder.

Run:
    python web_app.py

Open:
    http://127.0.0.1:5000
"""

import os
import sys
import json
import uuid
import time
import shutil
import zipfile
import threading
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory


# ═════════════════════════════════════════════════════════════
# PATH SETUP
# ═════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "../data").resolve()
RESULTS_DIR = DATA_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))


# ═════════════════════════════════════════════════════════════
# DOCKEXPLAIN IMPORTS
# ═════════════════════════════════════════════════════════════

from input_handler import (
    download_pdb,
    detect_ligands,
    fetch_ligand_info,
    fetch_protein_metadata,
    resolve_drug_input,
    get_preferred_name,
    _get_pubchem_cid,
)


# ═════════════════════════════════════════════════════════════
# FLASK APP
# ═════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["SECRET_KEY"] = "dockexplain-dev"

JOBS = {}


# ═════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════

def safe_name(name: str) -> str:
    cleaned = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in str(name).strip()
    )
    return cleaned[:80] or "job"


def add_log(job_id: str, text: str):
    if job_id not in JOBS:
        return

    JOBS[job_id]["log"] += text

    # Prevent browser payload from becoming massive
    max_chars = 250_000
    if len(JOBS[job_id]["log"]) > max_chars:
        JOBS[job_id]["log"] = (
            "... earlier log truncated ...\n"
            + JOBS[job_id]["log"][-max_chars:]
        )


def enrich_ligands(pdb_path: str):
    ligands = detect_ligands(pdb_path)
    enriched = []

    for lig in ligands:
        info = fetch_ligand_info(lig["code"])
        info["n_atoms"] = lig["n_atoms"]

        if info.get("smiles"):
            cid = _get_pubchem_cid(info["smiles"])
            info["preferred_name"] = get_preferred_name(
                cid,
                fallback=info.get("name") or lig["code"],
            )
            info["cid"] = cid
        else:
            info["preferred_name"] = info.get("name") or lig["code"]
            info["cid"] = None

        enriched.append(info)

    return enriched


def read_result_summary(out_dir: Path) -> dict:
    result_json = next(out_dir.glob("*_results.json"), None)

    if not result_json:
        fallback = out_dir / "results.json"
        result_json = fallback if fallback.exists() else None

    if not result_json:
        return {}

    with open(result_json, "r", encoding="utf-8") as f:
        result = json.load(f)

    return {
        "drug": result.get("drug"),
        "protein": result.get("protein"),
        "mode": result.get("mode"),
        "docking_score": result.get("docking_score"),
        "n_interactions": result.get("n_interactions"),
        "binding_site": result.get("binding_site"),
        "interactions": result.get("interactions", [])[:30],
    }


def collect_output_files(out_dir: Path) -> list[dict]:
    files = []

    if not out_dir.exists():
        return files

    preferred_order = {
        ".pdf": 1,
        ".png": 2,
        ".json": 3,
        ".csv": 4,
        ".txt": 5,
        ".pml": 6,
        ".pse": 7,
        ".sdf": 8,
        ".pdb": 9,
        ".pdbqt": 10,
    }

    paths = sorted(
    [
        p for p in out_dir.glob("*")
        if p.is_file()
        and not p.name.startswith("_")
        and p.suffix.lower() != ".zip"
    ],
    key=lambda p: (preferred_order.get(p.suffix.lower(), 99), p.name),
)

    for path in paths:
        files.append({
            "name": path.name,
            "url": f"/outputs/{out_dir.name}/{path.name}",
            "ext": path.suffix.lower().replace(".", ""),
        })

    return files


def make_zip(out_dir: Path) -> Path:
    zip_path = out_dir / f"{out_dir.name}_all_results.zip"

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in out_dir.glob("*"):
            if (
                path.is_file()
                and path.name != zip_path.name
                and not path.name.startswith("_")
            ):
                z.write(path, arcname=path.name)

            return zip_path


def write_runner_script(job_id: str, payload: dict, out_dir: Path) -> Path:
    """
    Creates a temporary runner script that runs the real DockExplain pipeline.

    Why this exists:
    - Running DockExplain inside a subprocess allows Flask to capture live output.
    - It captures Python prints and PyMOL subprocess output.
    - python -u gives unbuffered line-by-line logs.
    """

    runner_path = out_dir / "_dockexplain_runner.py"
    payload_path = out_dir / "_payload.json"

    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    code = f"""
import json
import sys
from pathlib import Path

BASE_DIR = Path({str(BASE_DIR)!r})
sys.path.insert(0, str(BASE_DIR))

from run_dockexplain import run_dockexplain

payload_path = Path({str(payload_path)!r})

with open(payload_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

print("\\n════════════════════════════════════════════════════════════")
print(" DockExplain Web Job Started")
print("════════════════════════════════════════════════════════════")
print(f"PDB ID      : {{payload.get('pdb_id')}}")
print(f"Drug        : {{payload.get('drug_name')}}")
print(f"Out dir     : {{payload.get('out_dir')}}")
print(f"Binding site: {{payload.get('selected_ligand_code')}}")
print("════════════════════════════════════════════════════════════\\n")

run_dockexplain(
    protein_pdb=payload["protein_pdb"],
    drug_name=payload.get("drug_name"),
    drug_smiles=payload.get("drug_smiles"),
    pdb_id=payload.get("pdb_id"),
    out_dir=payload["out_dir"],
    exhaustiveness=payload.get("exhaustiveness", 8),
    ollama_model=payload.get("model", "gemma4:e2b"),
    n_papers=payload.get("papers", 10),
    skip_pymol=payload.get("skip_pymol", False),
    skip_web_search=payload.get("skip_web_search", False),
    skip_explanation=payload.get("skip_explanation", False),
    pymol_headless=True,
    selected_ligand_code=payload.get("selected_ligand_code"),
)

print("\\n════════════════════════════════════════════════════════════")
print(" DockExplain Web Job Finished")
print("════════════════════════════════════════════════════════════")
"""

    with open(runner_path, "w", encoding="utf-8") as f:
        f.write(code)

    return runner_path


# ═════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/pdb")
def api_pdb():
    data = request.get_json(force=True)
    pdb_id = (data.get("pdb_id") or "").strip().upper()

    if not pdb_id or not pdb_id.isalnum():
        return jsonify({
            "ok": False,
            "error": "Enter a valid RCSB PDB ID, for example 4COX, 1M17, or 6LU7.",
        }), 400

    try:
        pdb_path = download_pdb(pdb_id, out_dir=str(DATA_DIR))
        metadata = fetch_protein_metadata(pdb_id)
        ligands = enrich_ligands(pdb_path)

        return jsonify({
            "ok": True,
            "pdb_id": pdb_id,
            "pdb_path": str(pdb_path),
            "metadata": metadata,
            "ligands": ligands,
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


@app.post("/api/resolve-drug")
def api_resolve_drug():
    data = request.get_json(force=True)
    drug_input = (data.get("drug_input") or "").strip()

    if not drug_input:
        return jsonify({
            "ok": False,
            "error": "Drug name, PDB ligand code, or SMILES is required.",
        }), 400

    try:
        resolved = resolve_drug_input(drug_input)

        if not resolved or not resolved.get("smiles"):
            return jsonify({
                "ok": False,
                "error": "Could not resolve this drug. Try a generic drug name, PDB ligand code, or direct SMILES.",
            }), 404

        return jsonify({
            "ok": True,
            "resolved": resolved,
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


def run_job(job_id: str, incoming_payload: dict):
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["started_at"] = datetime.now().isoformat(timespec="seconds")

    try:
        pdb_id = incoming_payload["pdb_id"].strip().upper()
        pdb_path = download_pdb(pdb_id, out_dir=str(DATA_DIR))

        use_crystal_ligand = bool(incoming_payload.get("use_crystal_ligand", False))
        selected_ligand_code = incoming_payload.get("selected_ligand_code") or None

        drug_name = (
            incoming_payload.get("drug_name")
            or incoming_payload.get("drug_input")
            or selected_ligand_code
            or "DockExplainDrug"
        )

        drug_smiles = None if use_crystal_ligand else incoming_payload.get("drug_smiles")

        out_dir = RESULTS_DIR / f"{safe_name(drug_name)}_{job_id[:8]}"
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
        "protein_pdb": str(pdb_path),
        "pdb_id": pdb_id,
        "drug_name": drug_name,
        "drug_smiles": drug_smiles,
        "selected_ligand_code": selected_ligand_code,
        "out_dir": str(out_dir),

        "exhaustiveness": 8,
        "model": "gemma4:e2b",
        "papers": 10,
        "skip_pymol": False,
        "skip_web_search": False,
        "skip_explanation": False,
    }

        JOBS[job_id]["out_dir"] = out_dir.name

        runner_path = write_runner_script(job_id, payload, out_dir)

        add_log(job_id, "Starting DockExplain...\n")
        add_log(job_id, f"Results folder: {out_dir}\n\n")

        process = subprocess.Popen(
            [sys.executable, "-u", str(runner_path)],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        JOBS[job_id]["pid"] = process.pid

        for line in process.stdout:
            add_log(job_id, line)

        return_code = process.wait()

        if return_code != 0:
            add_log(job_id, f"\nDockExplain failed with return code {return_code}\n")
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["finished_at"] = datetime.now().isoformat(timespec="seconds")
            JOBS[job_id]["files"] = collect_output_files(out_dir)
            JOBS[job_id]["summary"] = read_result_summary(out_dir)
            return

        zip_path = make_zip(out_dir)

        files = collect_output_files(out_dir)
        summary = read_result_summary(out_dir)

        JOBS[job_id].update({
            "status": "done",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "files": files,
            "summary": summary,
            "zip_url": f"/outputs/{out_dir.name}/{zip_path.name}",
        })

        add_log(job_id, f"\nZIP package created: {zip_path.name}\n")

    except Exception:
        add_log(job_id, "\n" + traceback.format_exc() + "\n")
        JOBS[job_id].update({
            "status": "failed",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "files": [],
            "summary": {},
        })


@app.post("/api/run")
def api_run():
    payload = request.get_json(force=True)

    if not payload.get("pdb_id"):
        return jsonify({
            "ok": False,
            "error": "PDB ID is required.",
        }), 400

    use_crystal_ligand = bool(payload.get("use_crystal_ligand", False))

    if use_crystal_ligand and not payload.get("selected_ligand_code"):
        return jsonify({
            "ok": False,
            "error": "Select a co-crystal ligand first.",
        }), 400

    if not use_crystal_ligand and not payload.get("drug_smiles"):
        return jsonify({
            "ok": False,
            "error": "Resolve or enter a drug before running.",
        }), 400

    job_id = uuid.uuid4().hex

    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": None,
        "finished_at": None,
        "log": "",
        "files": [],
        "summary": {},
        "zip_url": None,
        "out_dir": None,
        "pid": None,
    }

    thread = threading.Thread(
        target=run_job,
        args=(job_id, payload),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "ok": True,
        "job_id": job_id,
    })


@app.get("/api/job/<job_id>")
def api_job(job_id):
    job = JOBS.get(job_id)

    if not job:
        return jsonify({
            "ok": False,
            "error": "Job not found.",
        }), 404

    return jsonify({
        "ok": True,
        "job": job,
    })


@app.route("/outputs/<job_folder>/<path:filename>")
def outputs(job_folder, filename):
    return send_from_directory(
        RESULTS_DIR / job_folder,
        filename,
        as_attachment=True if filename.endswith(".zip") else False,
    )

@app.route("/health")
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
