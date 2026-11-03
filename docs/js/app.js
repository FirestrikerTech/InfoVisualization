/**
 * Main Application Controller for the MNIST 3D Loss Landscape Live Demo.
 *
 * Orchestrates:
 * - Loading binary weights (`weights.bin`), 500-image MNIST subset (`data.bin`), and `meta.json`.
 * - Initializing the Web Worker (`worker.js`) and Three.js / 2D Contour Plotter (`plot.js`).
 * - Phase 1: Manual Direction Mixer (`BASIS_COMPONENTS`, plane angle, span).
 * - Phase 2: Time-based unlock (~2.5 minutes, configurable via `meta.json` or instant via `#presenter`)
 *            of isotropic Gaussian random directions -> sharp funnel on the rescaled network.
 * - Phase 3: Hidden Presenter Unlock (3-second long-press on the `Loss L(θ*)` pill or `#presenter` hash)
 *            activating neuron-wise Filter Normalization (Li et al., 2018) -> wide, flat valley.
 */

import {
  BASIS_COMPONENTS,
  buildBasisTemplates,
  buildManualDirections,
  buildGaussianDirections,
} from "./directions.js";
import { LandscapePlotter } from "./plot.js";

const state = {
  meta: null,
  W1: null,
  b1: null,
  W2: null,
  b2: null,
  templates: null,

  mode: "manual", // 'manual' | 'random'
  editingAxis: "alpha", // 'alpha' | 'beta'
  alphaMix: {},
  betaMix: {},
  angleDeg: 0,
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

const elBtnEditAlpha = document.getElementById("btnEditAlpha");
const elBtnEditBeta = document.getElementById("btnEditBeta");
const elSlidersContainer = document.getElementById("slidersContainer");

const elRngAngle = document.getElementById("rngAngle");
const elLblAngle = document.getElementById("lblAngle");
const elRngSpan = document.getElementById("rngSpan");
const elLblSpan = document.getElementById("lblSpan");
const elRngSpanRandom = document.getElementById("rngSpanRandom");
const elLblSpanRandom = document.getElementById("lblSpanRandom");

const elBtnResetManual = document.getElementById("btnResetManual");
const elBtnHintValley = document.getElementById("btnHintValley");
const elBtnRerollSeed = document.getElementById("btnRerollSeed");
const elLblSeedInfo = document.getElementById("lblSeedInfo");

const elBtnInfo = document.getElementById("btnInfo");
const elModalInfo = document.getElementById("modalInfo");
const elBtnCloseModal = document.getElementById("btnCloseModal");

let plotter = null;
let worker = null;

function initDefaultMix() {
  for (const c of BASIS_COMPONENTS) {
    state.alphaMix[c.id] = c.defaultAlpha;
    state.betaMix[c.id] = c.defaultBeta;
  }
  state.angleDeg = 0;
  state.span = 1.0;
  elRngAngle.value = "0";
  elLblAngle.textContent = "0°";
  elRngSpan.value = "1.0";
  elLblSpan.textContent = "±1.0";
  elRngSpanRandom.value = "1.0";
  elLblSpanRandom.textContent = "±1.0";
}

function renderManualSliders() {
  elSlidersContainer.innerHTML = "";
  const activeMix = state.editingAxis === "alpha" ? state.alphaMix : state.betaMix;
  const accentColor = state.editingAxis === "alpha" ? "#38bdf8" : "#f59e0b";

  for (const comp of BASIS_COMPONENTS) {
    const val = activeMix[comp.id] ?? 0.0;
    const block = document.createElement("div");
    block.className = "slider-block";

    const head = document.createElement("div");
    head.className = "slider-head";
    const titleSpan = document.createElement("span");
    titleSpan.textContent = comp.label;
    const valStrong = document.createElement("strong");
    valStrong.style.color = accentColor;
    valStrong.textContent = (val >= 0 ? "+" : "") + val.toFixed(2);
    head.appendChild(titleSpan);
    head.appendChild(valStrong);

    const sub = document.createElement("div");
    sub.className = "slider-sub";
    sub.textContent = comp.shortDesc;

    const input = document.createElement("input");
    input.type = "range";
    input.min = "-1.5";
    input.max = "1.5";
    input.step = "0.05";
    input.value = String(val);
    input.style.accentColor = accentColor;

    input.addEventListener("input", () => {
      const v = parseFloat(input.value);
      activeMix[comp.id] = v;
      valStrong.textContent = (v >= 0 ? "+" : "") + v.toFixed(2);
      scheduleCompute(45);
    });

    block.appendChild(head);
    block.appendChild(sub);
    block.appendChild(input);
    elSlidersContainer.appendChild(block);
  }
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
  scheduleCompute(10);
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
    // Constructed dynamically at runtime
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
      scheduleCompute(10);
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
    scheduleCompute(10);
  }
}

function setupSecretGesture() {
  // 1. URL Hash shortcut (#presenter) for presenter's laptop / QR code
  const checkHash = () => {
    if (window.location.hash.toLowerCase().includes("presenter")) {
      unlockRandomPhase();
      unlockSecretPresenterControls(false);
    }
  };
  checkHash();
  window.addEventListener("hashchange", checkHash);

  // 2. 3-second long-press on the "Loss L(θ*)" metric pill
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

function scheduleCompute(delayMs = 40) {
  clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(() => {
    runCompute();
  }, delayMs);
}

function runCompute() {
  if (!worker || !state.templates) return;

  // Start the 2.5-minute countdown for Phase 2 on first landscape generation
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
    dirs = buildManualDirections(
      state.templates,
      state.alphaMix,
      state.betaMix,
      state.angleDeg,
      state.W1,
      state.W2,
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
  // 3D vs 2D toggle
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

  // Log scale toggle
  elBtnLogScale.addEventListener("click", () => {
    const next = !plotter.useLogScale;
    elBtnLogScale.classList.toggle("active", next);
    plotter.setLogScale(next);
  });

  // Mode tabs
  elTabManual.addEventListener("click", () => switchMode("manual"));
  elTabRandom.addEventListener("click", () => switchMode("random"));

  // Banner action -> jump straight into Phase 2 (Gaussian Random Directions)
  elBtnRollGaussian.addEventListener("click", () => {
    switchMode("random");
  });

  // Axis editing toggle in Manual Mode
  elBtnEditAlpha.addEventListener("click", () => {
    state.editingAxis = "alpha";
    elBtnEditAlpha.className = "axis-tab active-alpha";
    elBtnEditBeta.className = "axis-tab";
    renderManualSliders();
  });

  elBtnEditBeta.addEventListener("click", () => {
    state.editingAxis = "beta";
    elBtnEditAlpha.className = "axis-tab";
    elBtnEditBeta.className = "axis-tab active-beta";
    renderManualSliders();
  });

  // Angle & Span sliders
  elRngAngle.addEventListener("input", () => {
    state.angleDeg = parseFloat(elRngAngle.value);
    elLblAngle.textContent = `${state.angleDeg}°`;
    scheduleCompute(40);
  });

  const onSpanChange = (val) => {
    state.span = val;
    elRngSpan.value = String(val);
    elRngSpanRandom.value = String(val);
    elLblSpan.textContent = `±${val.toFixed(1)}`;
    elLblSpanRandom.textContent = `±${val.toFixed(1)}`;
    scheduleCompute(40);
  };

  elRngSpan.addEventListener("input", () => onSpanChange(parseFloat(elRngSpan.value)));
  elRngSpanRandom.addEventListener("input", () => onSpanChange(parseFloat(elRngSpanRandom.value)));

  // Manual presets
  elBtnResetManual.addEventListener("click", () => {
    initDefaultMix();
    renderManualSliders();
    scheduleCompute(15);
  });

  // "Tipp: Schichten koppeln" shows that by carefully mixing hidden + output + radial components
  // and rotating the plane, one CAN manually glimpse a local basin, though still distorted!
  elBtnHintValley.addEventListener("click", () => {
    state.alphaMix = {
      pixTopBottom: 0.65,
      pixLeftRight: -0.35,
      pixCenterEdge: 0.9,
      hidGroupAB: 1.15,
      outDigits: 0.55,
      radialScale: 0.85,
    };
    state.betaMix = {
      pixTopBottom: -0.4,
      pixLeftRight: 0.75,
      pixCenterEdge: -0.6,
      hidGroupAB: -0.8,
      outDigits: 0.65,
      radialScale: -0.75,
    };
    state.angleDeg = 35;
    elRngAngle.value = "35";
    elLblAngle.textContent = "35°";
    renderManualSliders();
    scheduleCompute(15);
  });

  // Random seed re-roll
  elBtnRerollSeed.addEventListener("click", () => {
    state.randomSeed = Math.floor(1000 + Math.random() * 9000);
    elLblSeedInfo.textContent = `Seed #${state.randomSeed}`;
    scheduleCompute(10);
  });

  // Info modal
  elBtnInfo.addEventListener("click", () => elModalInfo.classList.remove("hidden"));
  elBtnCloseModal.addEventListener("click", () => elModalInfo.classList.add("hidden"));
  elModalInfo.addEventListener("click", (e) => {
    if (e.target === elModalInfo) elModalInfo.classList.add("hidden");
  });
}

async function boot() {
  plotter = new LandscapePlotter(elPlot3D, elPlot2D);
  initDefaultMix();
  renderManualSliders();
  bindUIEvents();
  setupSecretGesture();

  elComputeBadge.textContent = "Lade Modell…";
  elComputeBadge.classList.remove("hidden");

  const [metaRes, weightsRes, dataRes] = await Promise.all([
    fetch("assets/meta.json"),
    fetch("assets/weights.bin"),
    fetch("assets/data.bin"),
  ]);

  const meta = await metaRes.json();
  const weightsBuffer = await weightsRes.arrayBuffer();
  const dataBuffer = await dataRes.arrayBuffer();

  state.meta = meta;
  elValTestAcc.textContent = `${(meta.metrics.full_test_acc * 100).toFixed(1)}%`;

  const weightsFloats = new Float32Array(weightsBuffer);
  const o = meta.offsets;
  state.W1 = weightsFloats.slice(o.W1[0], o.W1[1]);
  state.b1 = weightsFloats.slice(o.b1[0], o.b1[1]);
  state.W2 = weightsFloats.slice(o.W2[0], o.W2[1]);
  state.b2 = weightsFloats.slice(o.b2[0], o.b2[1]);

  state.templates = buildBasisTemplates(state.W1, state.b1, state.W2, state.b2);

  worker = new Worker("js/worker.js");
  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === "ready") {
      elValCenterLoss.textContent = msg.centerLoss.toFixed(3);
      runCompute();
    } else if (msg.type === "preview") {
      if (msg.jobId === state.jobCounter) {
        plotter.updateSurface(msg);
      }
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
      weightsBuffer,
      dataBuffer,
      numSamples: meta.num_samples,
      offsets: meta.offsets,
    },
    [weightsBuffer, dataBuffer]
  );
}

boot().catch((err) => {
  console.error("Failed to initialize Loss Landscape Explorer:", err);
  elComputeBadge.textContent = "Fehler beim Laden";
});
