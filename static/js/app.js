let state = {
  pdb: null,
  selectedLigand: null,
  resolvedDrug: null,
  jobId: null,
  poller: null,
};

const $ = (id) => document.getElementById(id);

function setHtml(id, html) {
  $(id).innerHTML = html;
}

function setText(id, text) {
  $(id).textContent = text;
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"]/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
  }[m]));
}

function setButtonLoading(button, isLoading, textWhenLoading, textDefault) {
  button.disabled = isLoading;
  button.textContent = isLoading ? textWhenLoading : textDefault;
}

function setStatus(status) {
  const el = $("jobStatus");
  el.textContent = status;
  el.className = "status";

  const lower = status.toLowerCase();

  if (lower.includes("running") || lower.includes("queued")) {
    el.classList.add("running");
  } else if (lower.includes("done")) {
    el.classList.add("done");
  } else if (lower.includes("failed")) {
    el.classList.add("failed");
  }
}

function scrollLogToBottom() {
  const log = $("log");
  log.scrollTop = log.scrollHeight;
}


// ═════════════════════════════════════════════════════════════
// STEP 1: FETCH PDB
// ═════════════════════════════════════════════════════════════

$("fetchPdb").onclick = async () => {
  const button = $("fetchPdb");
  const pdb_id = $("pdbId").value.trim();

  if (!pdb_id) {
    setHtml("proteinMeta", `<span class="danger">Please enter a PDB ID.</span>`);
    return;
  }

  setButtonLoading(button, true, "Fetching...", "Fetch Structure");
  setHtml("proteinMeta", "Fetching structure, metadata, and co-crystal ligands...");
  setHtml("ligandList", "Detecting ligands...");
  state.pdb = null;
  state.selectedLigand = null;

  try {
    const response = await fetch("/api/pdb", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({pdb_id}),
    });

    const data = await response.json();

    if (!data.ok) {
      setHtml("proteinMeta", `<span class="danger">${esc(data.error)}</span>`);
      setHtml("ligandList", `<span class="danger">Could not load ligands.</span>`);
      return;
    }

    state.pdb = data;

    const m = data.metadata || {};

    setHtml("proteinMeta", `
      <div class="protein-name">${esc(m.name || "Unknown protein")}</div>
      <div class="meta-grid">
        <span><b>PDB ID</b>${esc(data.pdb_id)}</span>
        <span><b>Organism</b>${esc(m.organism || "Unknown")}</span>
        <span><b>Method</b>${esc(m.method || "Unknown")}</span>
        <span><b>Resolution</b>${esc(m.resolution || "N/A")} Å</span>
        <span><b>Deposited</b>${esc(m.deposited || "Unknown")}</span>
      </div>
    `);

    renderLigands(data.ligands || []);

  } catch (err) {
    setHtml("proteinMeta", `<span class="danger">${esc(err.message)}</span>`);
  } finally {
    setButtonLoading(button, false, "Fetching...", "Fetch Structure");
  }
};


// ═════════════════════════════════════════════════════════════
// STEP 2: RENDER AND SELECT LIGANDS
// ═════════════════════════════════════════════════════════════

function renderLigands(ligands) {
  if (!ligands.length) {
    setHtml("ligandList", `
      <span class="warning">No co-crystal ligand detected.</span><br>
      Provide a drug and DockExplain will use pocket prediction if available.
    `);
    return;
  }

  const html = ligands.map((ligand, index) => {
    const code = esc(ligand.code || "");
    const name = esc(ligand.preferred_name || ligand.name || "Unknown ligand");
    const formula = esc(ligand.formula || "Formula unavailable");
    const atoms = esc(ligand.n_atoms || "N/A");
    const smiles = ligand.smiles ? "SMILES available" : "No SMILES found";

    return `
      <div class="ligand-card" data-index="${index}">
        <div>
          <div class="ligand-code">${code}</div>
          <strong>${name}</strong>
          <small>${formula} · ${atoms} atoms · ${smiles}</small>
        </div>
        <button type="button" onclick="selectLigand(${index})">Select</button>
      </div>
    `;
  }).join("");

  setHtml("ligandList", html);
  selectLigand(0);
}

window.selectLigand = (index) => {
  if (!state.pdb || !state.pdb.ligands || !state.pdb.ligands[index]) {
    return;
  }

  state.selectedLigand = state.pdb.ligands[index];

  document.querySelectorAll(".ligand-card").forEach((el, idx) => {
    el.classList.toggle("selected", idx === index);
  });

  if ($("useCrystal").checked) {
    const ligand = state.selectedLigand;
    setHtml("drugResolved", `
      <div class="resolved-title">Using selected crystal ligand</div>
      <b>${esc(ligand.code)} — ${esc(ligand.preferred_name || ligand.name || "")}</b><br>
      <small>The existing ligand will be analysed in its co-crystal site.</small>
    `);
  }
};


// ═════════════════════════════════════════════════════════════
// STEP 3: DRUG RESOLUTION
// ═════════════════════════════════════════════════════════════

$("useCrystal").onchange = () => {
  const useCrystal = $("useCrystal").checked;

  if (useCrystal) {
    if (!state.selectedLigand) {
      setHtml("drugResolved", `<span class="danger">Select a co-crystal ligand first.</span>`);
      return;
    }

    const ligand = state.selectedLigand;

    setHtml("drugResolved", `
      <div class="resolved-title">Using selected crystal ligand</div>
      <b>${esc(ligand.code)} — ${esc(ligand.preferred_name || ligand.name || "")}</b><br>
      <small>The existing ligand will be analysed in its co-crystal binding site.</small>
    `);
  } else {
    if (state.resolvedDrug) {
      renderResolvedDrug(state.resolvedDrug);
    } else {
      setHtml("drugResolved", "No drug resolved yet.");
    }
  }
};

$("resolveDrug").onclick = async () => {
  const button = $("resolveDrug");
  const drug_input = $("drugInput").value.trim();

  if (!drug_input) {
    setHtml("drugResolved", `<span class="danger">Please enter a drug name, PDB code, or SMILES.</span>`);
    return;
  }

  $("useCrystal").checked = false;

  setButtonLoading(button, true, "Resolving...", "Resolve");
  setHtml("drugResolved", "Resolving molecule...");

  try {
    const response = await fetch("/api/resolve-drug", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({drug_input}),
    });

    const data = await response.json();

    if (!data.ok) {
      state.resolvedDrug = null;
      setHtml("drugResolved", `<span class="danger">${esc(data.error)}</span>`);
      return;
    }

    state.resolvedDrug = data.resolved;
    renderResolvedDrug(data.resolved);

  } catch (err) {
    state.resolvedDrug = null;
    setHtml("drugResolved", `<span class="danger">${esc(err.message)}</span>`);
  } finally {
    setButtonLoading(button, false, "Resolving...", "Resolve");
  }
};

function renderResolvedDrug(drug) {
  setHtml("drugResolved", `
    <div class="resolved-title">Resolved molecule</div>
    <b>${esc(drug.name || "Resolved drug")}</b><br>
    <span>Source: ${esc(drug.source || "Unknown")}</span><br>
    <small>${esc(drug.smiles || "")}</small>
  `);
}


// ═════════════════════════════════════════════════════════════
// STEP 4: RUN JOB
// ═════════════════════════════════════════════════════════════

$("runBtn").onclick = async () => {
  if (!state.pdb) {
    alert("Fetch a PDB structure first.");
    return;
  }

  const useCrystal = $("useCrystal").checked;

  if (useCrystal && !state.selectedLigand) {
    alert("Select a co-crystal ligand first.");
    return;
  }

  if (!useCrystal && !state.resolvedDrug) {
    alert("Resolve a drug first, or choose the co-crystal ligand option.");
    return;
  }

  const selectedLigand = state.selectedLigand;
  const resolvedDrug = state.resolvedDrug;

  const payload = {
    pdb_id: state.pdb.pdb_id,
    selected_ligand_code: selectedLigand?.code || null,
    use_crystal_ligand: useCrystal,

    drug_input: $("drugInput").value.trim(),

    drug_name: useCrystal
      ? (selectedLigand?.preferred_name || selectedLigand?.name || selectedLigand?.code)
      : (resolvedDrug?.name || $("drugInput").value.trim()),

    drug_smiles: useCrystal ? null : resolvedDrug?.smiles,
  };

  const button = $("runBtn");
  setButtonLoading(button, true, "DockExplain is running...", "Run DockExplain");

  setStatus("Queued");
  setHtml("summary", "DockExplain is starting...");
  setHtml("files", "");
  $("zipDownload").classList.add("hidden");
  $("zipDownload").href = "#";
  setText("log", "Starting DockExplain...\n");
  scrollLogToBottom();

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!data.ok) {
      alert(data.error);
      setStatus("Failed");
      return;
    }

    state.jobId = data.job_id;

    if (state.poller) {
      clearInterval(state.poller);
    }

    state.poller = setInterval(pollJob, 1000);
    pollJob();

  } catch (err) {
    alert(err.message);
    setStatus("Failed");
  }
};


// ═════════════════════════════════════════════════════════════
// POLLING
// ═════════════════════════════════════════════════════════════

async function pollJob() {
  if (!state.jobId) return;

  try {
    const response = await fetch(`/api/job/${state.jobId}`);
    const data = await response.json();

    if (!data.ok) return;

    const job = data.job;

    setStatus(capitalize(job.status || "unknown"));

    if (job.log !== undefined) {
      setText("log", job.log || "");
      scrollLogToBottom();
    }

    if (job.summary && Object.keys(job.summary).length) {
      renderSummary(job.summary);
    }

    if (job.files && job.files.length) {
      renderFiles(job.files);
    }

    if (job.zip_url) {
      $("zipDownload").href = job.zip_url;
      $("zipDownload").classList.remove("hidden");
    }

    if (job.status === "done" || job.status === "failed") {
      clearInterval(state.poller);
      $("runBtn").disabled = false;
      $("runBtn").textContent = "Run DockExplain";
    }

  } catch (err) {
    console.error(err);
  }
}

function renderSummary(summary) {
  const score = summary.docking_score !== null && summary.docking_score !== undefined
    ? `${Number(summary.docking_score).toFixed(3)} kcal/mol`
    : "Crystal pose";

  setHtml("summary", `
    <div class="metric">
      <b>Drug</b>
      <span>${esc(summary.drug || "N/A")}</span>
    </div>

    <div class="metric">
      <b>Protein</b>
      <span>${esc(summary.protein || "N/A")}</span>
    </div>

    <div class="metric">
      <b>Mode</b>
      <span>${esc(summary.mode || "N/A")}</span>
    </div>

    <div class="metric">
      <b>Docking score</b>
      <span>${esc(score)}</span>
    </div>

    <div class="metric">
      <b>Interactions</b>
      <span>${esc(summary.n_interactions ?? "N/A")}</span>
    </div>
  `);
}

function renderFiles(files) {
  const html = files.map((file) => {
    const label = fileLabel(file.name, file.ext);
    return `<a href="${file.url}" target="_blank">${label}</a>`;
  }).join("");

  setHtml("files", html);
}

function fileLabel(name, ext) {
  const lower = String(ext || "").toLowerCase();

  if (lower === "pdf") return `PDF Report · ${esc(name)}`;
  if (lower === "png") return `Visualization · ${esc(name)}`;
  if (lower === "json") return `JSON · ${esc(name)}`;
  if (lower === "csv") return `CSV · ${esc(name)}`;
  if (lower === "txt") return `Explanation · ${esc(name)}`;
  if (lower === "pml") return `PyMOL Script · ${esc(name)}`;
  if (lower === "pse") return `PyMOL Session · ${esc(name)}`;
  if (lower === "sdf") return `Ligand Pose · ${esc(name)}`;
  if (lower === "zip") return `ZIP Package · ${esc(name)}`;

  return esc(name);
}

function capitalize(text) {
  if (!text) return "";
  return text.charAt(0).toUpperCase() + text.slice(1);
}


// ═════════════════════════════════════════════════════════════
// COPY LOG
// ═════════════════════════════════════════════════════════════

$("copyLog").onclick = async () => {
  const text = $("log").textContent || "";

  try {
    await navigator.clipboard.writeText(text);
    $("copyLog").textContent = "Copied";
    setTimeout(() => $("copyLog").textContent = "Copy Log", 1200);
  } catch {
    alert("Could not copy log.");
  }
};
