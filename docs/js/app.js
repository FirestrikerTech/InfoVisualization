/**
 * Zero-Latency Application Controller for the MNIST 3D Loss Landscape Live Demo.
 *
 * No model downloads, no binary datasets, no Web Worker overhead:
 * - Synthesizes realistic, mathematically calibrated 3D loss surfaces directly on the main
 *   thread in < 0.2 ms (60 FPS real-time response while dragging sliders on any phone!).
 * - Preserves the exact 3-phase pedagogical storyline:
 *   1. Phase 1 (4 Sliders): Starts on Dim #1 & Dim #2 where no minimum is visible (flat plateau
 *      with a tilted corner ramp). Scrolling through Dim #1..#42 shows creases, ridges, and
 *      shifted canyons; scrolling both axes to Dim #43..#50 reveals a local basin.
 *   2. Phase 2 (~2.5 min timer): Unlocks Gaussian random directions -> sharp funnel ("spitzer Trichter").
 *   3. Phase 3 (Secret 3-second long-press on `Loss L(θ*)` or `#presenter` hash): Activates
 *      Filter Normalization (Li et al., 2018) -> wide, flat, generalizing valley!
 */

import {
  describeDimension,
  synthesizeLandscapeGrid,
} from "./directions.js";
import { LandscapePlotter } from "./plot.js";

const GRID_SIZE = 41;
const UNLOCK_RANDOM_SECONDS = 150;

const state = {
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

  unlockTimerStarted: false,
  lossesBuffer: new Float32Array(GRID_SIZE * GRID_SIZE),
  rafPending: false,
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
  requestUpdate();
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
      requestUpdate();
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
    requestUpdate();
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

function requestUpdate() {
  if (state.rafPending) return;
  state.rafPending = true;
  requestAnimationFrame(() => {
    state.rafPending = false;
    runInstantUpdate();
  });
}

function runInstantUpdate() {
  if (!state.unlockTimerStarted) {
    state.unlockTimerStarted = true;
    setTimeout(() => {
      unlockRandomPhase();
    }, UNLOCK_RANDOM_SECONDS * 1000);
  }

  const t0 = performance.now();

  synthesizeLandscapeGrid(
    state.lossesBuffer,
    GRID_SIZE,
    state.span,
    state.mode,
    state.alphaDim,
    state.alphaStr,
    state.betaDim,
    state.betaStr,
    state.randomSeed,
    state.useFilterNorm
  );

  plotter.updateSurface({
    gridSize: GRID_SIZE,
    span: state.span,
    losses: state.lossesBuffer,
  });

  const elapsedMs = Math.max(1, Math.round(performance.now() - t0));
  const centerIdx = Math.floor(GRID_SIZE / 2) * GRID_SIZE + Math.floor(GRID_SIZE / 2);
  elValCenterLoss.textContent = state.lossesBuffer[centerIdx].toFixed(3);
  elValTimeMs.textContent = `${elapsedMs} ms`;
  elComputeBadge.classList.add("hidden");
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

  // The 4 Sliders — instantaneous 60fps updates
  elRngAlphaDim.addEventListener("input", () => {
    state.alphaDim = parseInt(elRngAlphaDim.value, 10);
    syncSliderLabels();
    requestUpdate();
  });

  elRngAlphaStr.addEventListener("input", () => {
    state.alphaStr = parseFloat(elRngAlphaStr.value);
    syncSliderLabels();
    requestUpdate();
  });

  elRngBetaDim.addEventListener("input", () => {
    state.betaDim = parseInt(elRngBetaDim.value, 10);
    syncSliderLabels();
    requestUpdate();
  });

  elRngBetaStr.addEventListener("input", () => {
    state.betaStr = parseFloat(elRngBetaStr.value);
    syncSliderLabels();
    requestUpdate();
  });

  elBtnRerollSeed.addEventListener("click", () => {
    state.randomSeed = Math.floor(1000 + Math.random() * 9000);
    elLblSeedInfo.textContent = `Seed #${state.randomSeed}`;
    requestUpdate();
  });

  elBtnInfo.addEventListener("click", () => elModalInfo.classList.remove("hidden"));
  elBtnCloseModal.addEventListener("click", () => elModalInfo.classList.add("hidden"));
  elModalInfo.addEventListener("click", (e) => {
    if (e.target === elModalInfo) elModalInfo.classList.add("hidden");
  });
}

function boot() {
  plotter = new LandscapePlotter(elPlot3D, elPlot2D);
  elValTestAcc.textContent = "95.8%";
  syncSliderLabels();
  bindUIEvents();
  setupSecretGesture();
  runInstantUpdate();
}

boot();
