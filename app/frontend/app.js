/* ==========================================================================
   PneumoNet — frontend logic

   Sends uploads to the Flask backend, which runs the real trained CNN.
   Nothing here is simulated.
   ========================================================================== */

// 127.0.0.1, not "localhost": Safari resolves "localhost" to the IPv6
// address ::1, where Flask is not listening, and the request hangs.
const API = "http://127.0.0.1:5001";
const TIMEOUT_MS = 15000;

const el = (id) => document.getElementById(id);

const dropzone = el("dropzone");
const fileInput = el("fileInput");
const xray = el("xray");
const cta = el("cta");

let currentFile = null;   // chosen, not yet analysed
let scanCount = 0;

/* ---------------- state machine -------------------------------------------
   One function owns every visual state, so no element can be left behind
   showing the wrong thing.
--------------------------------------------------------------------------- */

function setState(state, data) {
  document.body.dataset.state = state;

  const spinner = el("ctaSpinner");
  const ctaIcon = el("ctaIcon");
  const meter = el("meter");
  const scanline = el("scanline");

  // reset the pieces that vary
  spinner.hidden = true;
  ctaIcon.hidden = true;
  scanline.hidden = true;

  if (state === "empty") {
    dropzone.hidden = false;
    xray.hidden = true;
    xray.removeAttribute("src");
    meter.hidden = true;

    cta.disabled = true;
    el("ctaLabel").textContent = "Analyze scan";
    el("addScan").firstChild.textContent = "Add scan ";

    el("headline").innerHTML = "Preliminary<br>diagnosis";
    el("kvConfidence").textContent = "—";
    el("kvInference").textContent = "—";
    el("kvRef").textContent = "—";
    el("thumbBadge").textContent = "—";
    el("thumbBadge").className = "thumb-badge";
    el("thumbImg").style.backgroundImage = "";

    el("vcAvatar").textContent = "AI";
    el("vcTitle").textContent = "PneumoNet model";
    el("vcLead").textContent = "Awaiting scan.";
    el("vcLead").className = "vc-lead";
    el("vcText").textContent = " Upload a chest X-ray to run the model.";
    el("vcBadge").className = "vc-badge";
  }

  if (state === "ready") {
    dropzone.hidden = true;
    xray.hidden = false;
    cta.disabled = false;
    meter.hidden = true;                       // clear the previous result's bar
    el("meterFill").style.width = "0";
    el("ctaLabel").textContent = "Analyze scan";
    el("headline").innerHTML = "Preliminary<br>diagnosis";
    el("kvConfidence").textContent = "—";
    el("kvInference").textContent = "—";
    el("kvRef").textContent = "—";
    el("thumbBadge").textContent = "—";
    el("thumbBadge").className = "thumb-badge";
    el("vcAvatar").textContent = "AI";
    el("vcTitle").textContent = "PneumoNet model";
    el("vcBadge").className = "vc-badge";
    el("vcLead").textContent = "Scan loaded.";
    el("vcLead").className = "vc-lead";
    el("vcText").textContent = ' Press "Analyze scan" to run the model.';
  }

  if (state === "processing") {
    cta.disabled = true;
    meter.hidden = true;
    spinner.hidden = false;
    scanline.hidden = false;
    el("ctaLabel").textContent = "Processing";
    el("vcLead").textContent = "Analysing…";
    el("vcLead").className = "vc-lead";
    el("vcText").textContent = " Running the convolutional network on this scan.";
  }

  if (state === "result") {
    cta.disabled = false;
    ctaIcon.hidden = false;
    el("ctaLabel").textContent = "Explore results";
    el("addScan").firstChild.textContent = "New scan ";
    meter.hidden = false;
    renderResult(data);
  }

  if (state === "error") {
    cta.disabled = false;
    el("ctaLabel").textContent = "Retry";
    meter.hidden = true;
    el("headline").innerHTML = "Analysis<br>failed";
    el("vcAvatar").textContent = "!";
    el("vcTitle").textContent = "Analysis failed";
    el("vcLead").textContent = "Error.";
    el("vcLead").className = "vc-lead";
    el("vcText").textContent = " " + data;
    el("vcBadge").className = "vc-badge";
  }
}

/* ---------------- rendering ---------------- */

function renderResult(d) {
  const pct = d.probability * 100;
  const positive = d.probability > 0.5;

  el("headline").innerHTML = positive
    ? "Pneumonia<br>detected"
    : "No pneumonia<br>detected";

  el("kvConfidence").textContent = `${pct.toFixed(1)}%`;
  el("kvInference").textContent = `${d.inference_ms} ms`;
  el("kvRef").textContent = d.scan_ref;
  // A symbol, not a number — the precise figure is already shown beside it,
  // and two roundings of the same value read as two different values.
  el("thumbBadge").textContent = positive ? "!" : "✓";
  el("thumbBadge").className = positive ? "thumb-badge" : "thumb-badge is-good";

  el("vcAvatar").textContent = positive ? "P" : "N";
  el("vcTitle").textContent = positive ? "Pneumonia detected" : "No pneumonia detected";
  el("vcSub").textContent = `${d.image_width}×${d.image_height} ${d.image_format} · resized to 28×28`;

  el("vcLead").textContent = positive ? "Positive." : "Negative.";
  el("vcLead").className = positive ? "vc-lead" : "vc-lead is-good";
  el("vcText").textContent = " " + interpret(d.probability);

  el("vcBadge").className = positive ? "vc-badge" : "vc-badge is-good";

  const fill = el("meterFill");
  fill.className = positive ? "meter-fill" : "meter-fill is-good";
  requestAnimationFrame(() => { fill.style.width = `${pct}%`; });
}

/*
  Wording stays inside what a binary pneumonia/normal classifier supports.
  The model cannot identify a cause, localise findings, or exclude other
  conditions — so it does not claim to.
*/
function interpret(p) {
  if (p > 0.9)
    return "Strong indication of pneumonia. This is a screening signal only — the model cannot identify a cause or rule out other conditions.";
  if (p > 0.5)
    return "Findings lean towards pneumonia with moderate confidence. Results near the 0.5 threshold are the least reliable.";
  if (p > 0.2)
    return "No pneumonia indicated, though confidence is not high. This model favours sensitivity, so it rarely misses pneumonia but does flag many normal scans.";
  return "Lungs appear clear to the model. No indication of pneumonia.";
}

/* ---------------- history ---------------- */

function addToHistory(dataUrl, d) {
  el("historyEmpty").hidden = true;
  scanCount += 1;
  el("histCount").textContent = scanCount;

  const positive = d.probability > 0.5;
  const item = document.createElement("div");
  item.className = `hist-item ${positive ? "is-pos" : "is-neg"}`;
  item.title = `${d.label} · ${(d.probability * 100).toFixed(1)}%`;

  const thumb = document.createElement("div");
  thumb.className = "hist-thumb";
  thumb.style.backgroundImage = `url(${dataUrl})`;

  const label = document.createElement("span");
  label.className = "hist-label";
  label.textContent = `${(d.probability * 100).toFixed(0)}%`;

  item.append(thumb, label);
  item.addEventListener("click", () => {
    xray.src = dataUrl;
    el("thumbImg").style.backgroundImage = `url(${dataUrl})`;
    setState("result", d);
  });

  el("history").prepend(item);
}

/* ---------------- upload ---------------- */

dropzone.addEventListener("click", () => fileInput.click());
el("addScan").addEventListener("click", () => {
  if (document.body.dataset.state === "result") setState("empty");
  fileInput.click();
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) loadFile(e.target.files[0]);
});

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) loadFile(f);
});
// stop the browser navigating away if a file misses the dropzone
["dragover", "drop"].forEach((ev) => document.addEventListener(ev, (e) => e.preventDefault()));


function loadFile(file) {
  if (!file.type.startsWith("image/")) {
    setState("error", "That file is not an image. Upload a PNG or JPG chest X-ray.");
    return;
  }
  currentFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    xray.src = e.target.result;
    el("thumbImg").style.backgroundImage = `url(${e.target.result})`;
    setState("ready");
  };
  reader.readAsDataURL(file);
}

/* ---------------- analyse ---------------- */

cta.addEventListener("click", () => {
  const state = document.body.dataset.state;
  if (state === "ready" || state === "error") analyse();
  else if (state === "result") el("history").scrollIntoView({ behavior: "smooth" });
});

async function analyse() {
  if (!currentFile) return;
  setState("processing");

  const form = new FormData();
  form.append("image", currentFile);

  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST", body: form, signal: abort.signal,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed.");

    setState("result", data);
    addToHistory(xray.src, data);
  } catch (err) {
    console.error("prediction failed:", err);
    if (err.name === "AbortError") {
      setState("error", `No response from the model after ${TIMEOUT_MS / 1000}s. Check the backend terminal.`);
    } else if (err instanceof TypeError) {
      setState("error", "Cannot reach the backend. Is 'python app.py' running?");
    } else {
      setState("error", err.message);
    }
  } finally {
    clearTimeout(timer);
  }
}

/* ---------------- boot ---------------- */

el("backBtn").addEventListener("click", () => setState("empty"));

el("todayDate").textContent = new Date().toLocaleDateString("en-GB", {
  day: "numeric", month: "short", year: "numeric",
});

(async function checkBackend() {
  const pip = el("statusPip");
  try {
    const r = await fetch(`${API}/health`);
    if (!r.ok) throw new Error();
    const d = await r.json();
    pip.className = "pip ok";
    el("statusBtn").title = `Model online · ${d.params.toLocaleString()} parameters`;
    el("vcSub").textContent = `CNN · ${d.params.toLocaleString()} parameters`;
    el("perfSens").textContent = `${d.sensitivity}%`;
    el("perfSpec").textContent = `${d.specificity}%`;
    el("perfAuc").textContent = d.auc.toFixed(3);
    el("perfPill").textContent = `test set n=${d.test_size}`;
  } catch {
    pip.className = "pip down";
    el("statusBtn").title = "Backend offline — run: python app.py";
    el("vcSub").textContent = "Backend offline";
  }
})();

setState("empty");
