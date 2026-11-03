/**
 * Main Application Controller for the MNIST 3D Loss Landscape Live Demo.
 *
 * Updates:
 * - Loads Int16-quantized weights (`weights_q16.bin`, 50.9 KB) and 50 MNIST images (`data.bin`, 39 KB).
 * - Simplified 4-slider manual interface:
 *     1. Achse α Dimension-Scroller (1..50)
 *     2. Achse α Stärke (-2.0..+2.0)
 *     3. Achse β Dimension-Scroller (1..50)
 *     4. Achse β Stärke (-2.0..+2.0)
 * - Initialized to Dim #1 & Dim #2 where the audience sees an inclined ramp / ridge instead of
 *   the minimum!
 * - Phase 2 (~2.5 min): Unlocks Gaussian random directions -> sharp funnel on rescaled network.
 * - Phase 3 (Secret 3s long-press on Loss L(θ*) or `#presenter`): Filter Normalization -> flat valley.
 */

import {
  describeDimension,
  buildManualDirections4Sliders,
  buildGaussianDirections,
} from "./directions.js";
import { LandscapePlotter } from "./plot.js";

const state = {
  meta: null,
  W1: null,
  b1: null,
  W2: null,
  b2: null,

  mode: "manual", // 'manual' | 'random'
  alphaDim: 1,
  alphaStr: 1.0,
  betaDim: 2,
  betaStr: 1.0,
  span: 1.0,

  randomSeed: 2026,
  randomUnlocked: false,
  secretUnlocked: false,
  useFilterNorm: false,

  jobCounter: 0,
  debounceTimer: null,
  unlockTimerStarted: false,
};

// DOM Elements
const elPlot3D = document.getElementById("plot3D");
const elPlot2D = document.getElementById("plot2D");
const elValCenterLoss = document.getElementById("valCenterLoss");
const elValTestAcc = document.getElementById("valTestAcc");
const elValTimeMs = document.getElementById("valTimeMs");
const elComputeBadge = document.getElementById("computeBadge");

const elBtnView3D = document.getElementById("btnView3D");
const elBtnView2D = document.getElementById("btnView2D");
const elBtnLogScale = document.getElementById("btnLogScale");

const elRandomBanner = document.getElementById("randomPhaseBanner");
const elBtnRollGaussian = document.getElementById("btnRollGaussian");
const elSecretContainer = document.getElementById("secretContainer");
const elSecretTrigger = document.getElementById("secretTrigger");

const elTabManual = document.getElementById("tabManual");
const elTabRandom = document.getElementById("tabRandom");
const elModeStatusBadge = document.getElementById("modeStatusBadge");
const elPanelManual = document.getElementById("panelManual");
const elPanelRandom = document.getElementById("panelRandom");

// The 4 Sliders
const elRngAlphaDim = document.getElementById("rngAlphaDim");
const elLblAlphaDim = document.getElementById("lblAlphaDim");
const elDescAlphaDim = document.getElementById("descAlphaDim");
const elRngAlphaStr = document.getElementById("rngAlphaStr");
const elLblAlphaStr = document.getElementById("lblAlphaStr");

const elRngBetaDim = document.getElementById("rngBetaDim");
const elLblBetaDim = document.getElementById("lblBetaDim");
const elDescBetaDim = document.getElementById("descBetaDim");
const elRngBetaStr = document.getElementById("rngBetaStr");
const elLblBetaStr = document.getElementById("lblBetaStr");

const elBtnRerollSeed = document.getElementById("btnRerollSeed");
const elLblSeedInfo = document.getElementById("lblSeedInfo");

const elBtnInfo = document.getElementById("btnInfo");
const elModalInfo = document.getElementById("modalInfo");
const elBtnCloseModal = document.getElementById("btnCloseModal");

let plotter = null;
let worker = null;

function syncSliderLabels() {
  elLblAlphaDim.textContent = `Dim #${state.alphaDim}`;
  elDescAlphaDim.textContent = describeDimension(state.alphaDim);
  elLblAlphaStr.textContent = (state.alphaStr >= 0 ? "+" : "") + state.alphaStr.toFixed(2);

  elLblBetaDim.textContent = `Dim #${state.betaDim}`;
  elDescBetaDim.textContent = describeDimension(state.betaDim);
  elLblBetaStr.textContent = (state.betaStr >= 0 ? "+" : "") + state.betaStr.toFixed(2);
}

function updateStatusBadge() {
  if (state.mode === "manual") {
    elModeStatusBadge.textContent = state.useFilterNorm
      ? "Manuell (normiert)"
      : "Manueller Schnitt";
    elModeStatusBadge.style.color = "#38bdf8";
  } else {
    if (state.useFilterNorm) {
      elModeStatusBadge.textContent = "Gauß + Filternorm (Li et al.)";
      elModeStatusBadge.style.color = "#10b981";
    } else {
      elModeStatusBadge.textContent = "Gauß (unnormiert)";
      elModeStatusBadge.style.color = "#c084fc";
    }
  }
}

function switchMode(newMode) {
  state.mode = newMode;
  if (newMode === "manual") {
    elTabManual.classList.add("active");
    elTabRandom.classList.remove("active");
    elPanelManual.classList.remove("hidden");
    elPanelRandom.classList.add("hidden");
  } else {
    elTabManual.classList.remove("active");
    elTabRandom.classList.add("active");
    elPanelManual.classList.add("hidden");
    elPanelRandom.classList.remove("hidden");
    elRandomBanner.classList.add("hidden");
  }
  updateStatusBadge();
  scheduleCompute(5);
}

function unlockRandomPhase() {
  if (state.randomUnlocked) return;
  state.randomUnlocked = true;
  elTabRandom.classList.remove("hidden");
  if (state.mode === "manual") {
    elRandomBanner.classList.remove("hidden");
  }
}

/**
 * Dynamically builds the secret Phase 3 control panel at runtime so the term
 * is not sitting visibly in the static DOM before the presenter triggers it.
 */
function unlockSecretPresenterControls(autoEnable = false) {
  if (!state.secretUnlocked) {
    state.secretUnlocked = true;
    unlockRandomPhase();

    const row = document.createElement("div");
    row.className = "secret-row";

    const info = document.createElement("div");
    info.className = "secret-info";
    const title = document.createElement("strong");
    title.textContent = "✨ " + ["Filter", "normierung"].join("") + " (Li et al., 2018)";
    const sub = document.createElement("span");
    sub.textContent = "Skalierungsinvariante Normierung pro Neuron: d ← (d / ‖d‖) · ‖θ‖";
    info.appendChild(title);
    info.appendChild(sub);

    const btn = document.createElement("button");
    btn.className = "btn-toggle-fn";
    const syncBtnText = () => {
      if (state.useFilterNorm) {
        btn.classList.add("active");
        btn.textContent = "Aktiviert ✓";
      } else {
        btn.classList.remove("active");
        btn.textContent = "Jetzt aktivieren";
      }
    };
    syncBtnText();

    btn.addEventListener("click", () => {
      state.useFilterNorm = !state.useFilterNorm;
      syncBtnText();
      updateStatusBadge();
      scheduleCompute(5);
    });

    row.appendChild(info);
    row.appendChild(btn);
    elSecretContainer.innerHTML = "";
    elSecretContainer.appendChild(row);
    elSecretContainer.classList.remove("hidden");
    state._syncSecretBtn = syncBtnText;
  }

  if (autoEnable && !state.useFilterNorm) {
    state.useFilterNorm = true;
    if (state._syncSecretBtn) state._syncSecretBtn();
    updateStatusBadge();
    scheduleCompute(5);
  }
}

function setupSecretGesture() {
  const checkHash = () => {
    if (window.location.hash.toLowerCase().includes("presenter")) {
      unlockRandomPhase();
      unlockSecretPresenterControls(false);
    }
  };
  checkHash();
  window.addEventListener("hashchange", checkHash);

  let pressTimer = null;
  const startPress = () => {
    clearTimeout(pressTimer);
    pressTimer = setTimeout(() => {
      unlockSecretPresenterControls(false);
    }, 3000);
  };
  const cancelPress = () => {
    clearTimeout(pressTimer);
  };

  elSecretTrigger.addEventListener("touchstart", startPress, { passive: true });
  elSecretTrigger.addEventListener("touchend", cancelPress);
  elSecretTrigger.addEventListener("touchcancel", cancelPress);
  elSecretTrigger.addEventListener("mousedown", startPress);
  elSecretTrigger.addEventListener("mouseup", cancelPress);
  elSecretTrigger.addEventListener("mouseleave", cancelPress);
}

function scheduleCompute(delayMs = 12) {
  clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(() => {
    runCompute();
  }, delayMs);
}

function runCompute() {
  if (!worker || !state.W1) return;

  if (!state.unlockTimerStarted) {
    state.unlockTimerStarted = true;
    const waitSec = state.meta?.unlock_random_seconds ?? 150;
    setTimeout(() => {
      unlockRandomPhase();
    }, waitSec * 1000);
  }

  const jobId = ++state.jobCounter;
  elComputeBadge.classList.remove("hidden");

  let dirs;
  if (state.mode === "manual") {
    dirs = buildManualDirections4Sliders(
      state.alphaDim,
      state.alphaStr,
      state.betaDim,
      state.betaStr,
      state.W1,
      state.b1,
      state.W2,
      state.b2,
      state.useFilterNorm
    );
  } else {
    dirs = buildGaussianDirections(
      state.randomSeed,
      state.meta.sigma_unnorm,
      state.W1,
      state.W2,
      state.useFilterNorm
    );
  }

  worker.postMessage({
    type: "compute",
    jobId,
    gridSize: 41,
    span: state.span,
    ...dirs,
  });
}

function bindUIEvents() {
  elBtnView3D.addEventListener("click", () => {
    elBtnView3D.classList.add("active");
    elBtnView2D.classList.remove("active");
    plotter.setViewMode("3d");
  });

  elBtnView2D.addEventListener("click", () => {
    elBtnView2D.classList.add("active");
    elBtnView3D.classList.remove("active");
    plotter.setViewMode("2d");
  });

  elBtnLogScale.addEventListener("click", () => {
    const next = !plotter.useLogScale;
    elBtnLogScale.classList.toggle("active", next);
    plotter.setLogScale(next);
  });

  elTabManual.addEventListener("click", () => switchMode("manual"));
  elTabRandom.addEventListener("click", () => switchMode("random"));

  elBtnRollGaussian.addEventListener("click", () => {
    switchMode("random");
  });

  // The 4 Sliders
  elRngAlphaDim.addEventListener("input", () => {
    state.alphaDim = parseInt(elRngAlphaDim.value, 10);
    syncSliderLabels();
    scheduleCompute(10);
  });

  elRngAlphaStr.addEventListener("input", () => {
    state.alphaStr = parseFloat(elRngAlphaStr.value);
    syncSliderLabels();
    scheduleCompute(10);
  });

  elRngBetaDim.addEventListener("input", () => {
    state.betaDim = parseInt(elRngBetaDim.value, 10);
    syncSliderLabels();
    scheduleCompute(10);
  });

  elRngBetaStr.addEventListener("input", () => {
    state.betaStr = parseFloat(elRngBetaStr.value);
    syncSliderLabels();
    scheduleCompute(10);
  });

  elBtnRerollSeed.addEventListener("click", () => {
    state.randomSeed = Math.floor(1000 + Math.random() * 9000);
    elLblSeedInfo.textContent = `Seed #${state.randomSeed}`;
    scheduleCompute(5);
  });

  elBtnInfo.addEventListener("click", () => elModalInfo.classList.remove("hidden"));
  elBtnCloseModal.addEventListener("click", () => elModalInfo.classList.add("hidden"));
  elModalInfo.addEventListener("click", (e) => {
    if (e.target === elModalInfo) elModalInfo.classList.add("hidden");
  });
}

function dequantizeSlice(q16Array, start, end, scale) {
  const len = end - start;
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    out[i] = q16Array[start + i] * scale;
  }
  return out;
}

async function boot() {
  plotter = new LandscapePlotter(elPlot3D, elPlot2D);
  syncSliderLabels();
  bindUIEvents();
  setupSecretGesture();

  elComputeBadge.textContent = "Lade…";
  elComputeBadge.classList.remove("hidden");

  // Load compact Int16 quantized weights (50.9 KB) + 50-image dataset (39.2 KB)
  const [metaRes, weightsQ16Res, dataRes] = await Promise.all([
    fetch("assets/meta.json"),
    fetch("assets/weights_q16.bin"),
    fetch("assets/data.bin"),
  ]);

  const meta = await metaRes.json();
  const weightsQ16Buffer = await weightsQ16Res.arrayBuffer();
  const dataBuffer = await dataRes.arrayBuffer();

  state.meta = meta;
  elValTestAcc.textContent = `${(meta.metrics.full_test_acc * 100).toFixed(1)}%`;

  const q16 = new Int16Array(weightsQ16Buffer);
  const o = meta.offsets;
  const sc = meta.quantization.scales;

  state.W1 = dequantizeSlice(q16, o.W1[0], o.W1[1], sc.W1);
  state.b1 = dequantizeSlice(q16, o.b1[0], o.b1[1], sc.b1);
  state.W2 = dequantizeSlice(q16, o.W2[0], o.W2[1], sc.W2);
  state.b2 = dequantizeSlice(q16, o.b2[0], o.b2[1], sc.b2);

  worker = new Worker("js/worker.js");
  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === "ready") {
      elValCenterLoss.textContent = msg.centerLoss.toFixed(3);
      runCompute();
    } else if (msg.type === "result") {
      if (msg.jobId === state.jobCounter) {
        plotter.updateSurface(msg);
        elValCenterLoss.textContent = msg.centerLoss.toFixed(3);
        elValTimeMs.textContent = `${msg.elapsedMs} ms`;
        elComputeBadge.classList.add("hidden");
      }
    }
  };

  worker.postMessage(
    {
      type: "init",
      W1: state.W1,
      b1: state.b1,
      W2: state.W2,
      b2: state.b2,
      dataBuffer,
      numSamples: meta.num_samples,
    },
    [dataBuffer]
  );
}

boot().catch((err) => {
  console.error("Failed to initialize Loss Landscape Explorer:", err);
  elComputeBadge.textContent = "Fehler beim Laden";
});
