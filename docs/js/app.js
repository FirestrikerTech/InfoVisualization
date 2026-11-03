/**
 * Zero-Latency Application Controller for the MNIST 3D Loss Landscape Live Demo.
 *
 * Key behavior of the 4 Sliders:
 * - Axis α has a persistent 50-dimensional vector `alphaVec[1..50]` (initialized with `alphaVec[1] = 1.0`, others `0.0`).
 * - Axis β has a persistent 50-dimensional vector `betaVec[1..50]` (initialized with `betaVec[2] = 1.0`, others `0.0`).
 * - Scrolling Slider 1 (`rngAlphaDim`) or Slider 3 (`rngBetaDim`) ONLY selects the dimension `i`
 *   and loads its stored value `alphaVec[i]` / `betaVec[i]` into Slider 2 / Slider 4.
 *   It does NOT modify the vectors or the 3D landscape!
 * - Moving Slider 2 (`rngAlphaStr`) or Slider 4 (`rngBetaStr`) modifies the stored value at the
 *   currently selected dimension (`alphaVec[alphaDim] = val`) and triggers a live landscape update
 *   with a clear visual loading/activity indicator!
 */

import {
  NUM_SCROLL_DIMS,
  describeDimension,
  synthesizeLandscapeGrid,
} from "./directions.js";
import { LandscapePlotter } from "./plot.js";

const GRID_SIZE = 41;
const UNLOCK_RANDOM_SECONDS = 150;

const state = {
  mode: "manual", // 'manual' | 'random'
  alphaDim: 1,
  betaDim: 2,
  alphaVec: new Float32Array(NUM_SCROLL_DIMS + 1),
  betaVec: new Float32Array(NUM_SCROLL_DIMS + 1),
  span: 1.0,

  randomSeed: 2026,
  randomUnlocked: false,
  secretUnlocked: false,
  useFilterNorm: false,

  unlockTimerStarted: false,
  lossesBuffer: new Float32Array(GRID_SIZE * GRID_SIZE),
  indicatorTimer: null,
};

// DOM Elements
const elPlot3D = document.getElementById("plot3D");
const elPlot2D = document.getElementById("plot2D");
const elValCenterLoss = document.getElementById("valCenterLoss");
const elValTestAcc = document.getElementById("valTestAcc");
const elValActiveDims = document.getElementById("valActiveDims");
const elValTimeMs = document.getElementById("valTimeMs");

const elComputeBadge = document.getElementById("computeBadge");
const elComputeText = document.getElementById("computeText");

const elBtnView3D = document.getElementById("btnView3D");
const elBtnView2D = document.getElementById("btnView2D");
const elBtnLogScale = document.getElementById("btnLogScale");

const elRandomBanner = document.getElementById("randomPhaseBanner");
const elBtnRollGaussian = document.getElementById("btnRollGaussian");
const elSecretContainer = document.getElementById("secretContainer");
const elSecretTrigger = document.getElementById("secretTrigger");

const elTabManual = document.getElementById("tabManual");
const elTabRandom = document.getElementById("tabRandom");
const elBtnResetVectors = document.getElementById("btnResetVectors");
const elPanelManual = document.getElementById("panelManual");
const elPanelRandom = document.getElementById("panelRandom");

// The 4 Sliders & Labels
const elRngAlphaDim = document.getElementById("rngAlphaDim");
const elLblAlphaDim = document.getElementById("lblAlphaDim");
const elDescAlphaDim = document.getElementById("descAlphaDim");
const elRngAlphaStr = document.getElementById("rngAlphaStr");
const elLblAlphaStr = document.getElementById("lblAlphaStr");
const elLblAlphaStrTitle = document.getElementById("lblAlphaStrTitle");
const elBadgeAlphaNonZero = document.getElementById("badgeAlphaNonZero");

const elRngBetaDim = document.getElementById("rngBetaDim");
const elLblBetaDim = document.getElementById("lblBetaDim");
const elDescBetaDim = document.getElementById("descBetaDim");
const elRngBetaStr = document.getElementById("rngBetaStr");
const elLblBetaStr = document.getElementById("lblBetaStr");
const elLblBetaStrTitle = document.getElementById("lblBetaStrTitle");
const elBadgeBetaNonZero = document.getElementById("badgeBetaNonZero");

const elBtnRerollSeed = document.getElementById("btnRerollSeed");
const elLblSeedInfo = document.getElementById("lblSeedInfo");

const elBtnInfo = document.getElementById("btnInfo");
const elModalInfo = document.getElementById("modalInfo");
const elBtnCloseModal = document.getElementById("btnCloseModal");

let plotter = null;

function initVectors() {
  state.alphaVec.fill(0.0);
  state.betaVec.fill(0.0);
  // Initial non-zero values on Dim #1 and Dim #2 (border/ramp dimensions where no minimum is visible)
  state.alphaVec[1] = 1.0;
  state.betaVec[2] = 1.0;
  state.alphaDim = 1;
  state.betaDim = 2;
  elRngAlphaDim.value = "1";
  elRngBetaDim.value = "2";
}

function countNonZero(vec) {
  let c = 0;
  for (let d = 1; d <= NUM_SCROLL_DIMS; d++) {
    if (Math.abs(vec[d]) > 1e-4) c++;
  }
  return c;
}

function syncSliderUI() {
  const aDim = state.alphaDim;
  const aVal = state.alphaVec[aDim];
  elLblAlphaDim.textContent = `Dim #${aDim}`;
  elDescAlphaDim.textContent = describeDimension(aDim);
  elLblAlphaStrTitle.textContent = `2. Gespeicherter Wert für Dim #${aDim}`;
  elRngAlphaStr.value = String(aVal);
  elLblAlphaStr.textContent = (aVal >= 0 ? "+" : "") + aVal.toFixed(2);

  const bDim = state.betaDim;
  const bVal = state.betaVec[bDim];
  elLblBetaDim.textContent = `Dim #${bDim}`;
  elDescBetaDim.textContent = describeDimension(bDim);
  elLblBetaStrTitle.textContent = `4. Gespeicherter Wert für Dim #${bDim}`;
  elRngBetaStr.value = String(bVal);
  elLblBetaStr.textContent = (bVal >= 0 ? "+" : "") + bVal.toFixed(2);

  const nA = countNonZero(state.alphaVec);
  const nB = countNonZero(state.betaVec);
  elBadgeAlphaNonZero.textContent = `${nA} Dim${nA === 1 ? "" : "s"} aktiv`;
  elBadgeBetaNonZero.textContent = `${nB} Dim${nB === 1 ? "" : "s"} aktiv`;

  // Count distinct active dimensions across both vectors
  let totalDistinct = 0;
  for (let d = 1; d <= NUM_SCROLL_DIMS; d++) {
    if (Math.abs(state.alphaVec[d]) > 1e-4 || Math.abs(state.betaVec[d]) > 1e-4) {
      totalDistinct++;
    }
  }
  if (state.mode === "manual") {
    elValActiveDims.textContent = `${totalDistinct} / 50`;
  } else {
    elValActiveDims.textContent = "25.450 D";
  }
}

/**
 * Show a clear visual indicator differentiating:
 * - "inspecting": User scrolled Slider 1 or 3 (selecting another dimension without changing the plot)
 * - "updating":   User changed Slider 2 or 4 (or rolled Gaussian directions), recalculating the 3D surface!
 */
function flashActivityIndicator(kind, message) {
  clearTimeout(state.indicatorTimer);
  elComputeBadge.classList.remove("updating", "inspecting");

  if (kind === "updating") {
    elComputeBadge.classList.add("updating");
    elComputeText.textContent = message || "Aktualisiere 3D-Landschaft…";
    elValTimeMs.textContent = "Rechne…";
    elValTimeMs.style.color = "#38bdf8";
  } else if (kind === "inspecting") {
    elComputeBadge.classList.add("inspecting");
    elComputeText.textContent = message || "Dimension gewählt (Plot unverändert)";
  }

  state.indicatorTimer = setTimeout(() => {
    elComputeBadge.classList.remove("updating", "inspecting");
    elComputeText.textContent = "Bereit ✓";
    elValTimeMs.textContent = "Bereit ✓";
    elValTimeMs.style.color = "#10b981";
  }, 360);
}

function switchMode(newMode) {
  state.mode = newMode;
  if (newMode === "manual") {
    elTabManual.classList.add("active");
    elTabRandom.classList.remove("active");
    elPanelManual.classList.remove("hidden");
    elPanelRandom.classList.add("hidden");
    elBtnResetVectors.classList.remove("hidden");
  } else {
    elTabManual.classList.remove("active");
    elTabRandom.classList.add("active");
    elPanelManual.classList.add("hidden");
    elPanelRandom.classList.remove("hidden");
    elRandomBanner.classList.add("hidden");
    elBtnResetVectors.classList.add("hidden");
  }
  syncSliderUI();
  triggerLandscapeUpdate("Berechne Gauß-Schnitt…");
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
      triggerLandscapeUpdate("Wende Filternormierung an…");
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
    triggerLandscapeUpdate("Wende Filternormierung an…");
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

function triggerLandscapeUpdate(labelMsg) {
  if (!state.unlockTimerStarted) {
    state.unlockTimerStarted = true;
    setTimeout(() => {
      unlockRandomPhase();
    }, UNLOCK_RANDOM_SECONDS * 1000);
  }

  flashActivityIndicator("updating", labelMsg || "Aktualisiere 3D-Landschaft…");

  synthesizeLandscapeGrid(
    state.lossesBuffer,
    GRID_SIZE,
    state.span,
    state.mode,
    state.alphaVec,
    state.betaVec,
    state.randomSeed,
    state.useFilterNorm
  );

  plotter.updateSurface({
    gridSize: GRID_SIZE,
    span: state.span,
    losses: state.lossesBuffer,
  });

  const centerIdx = Math.floor(GRID_SIZE / 2) * GRID_SIZE + Math.floor(GRID_SIZE / 2);
  elValCenterLoss.textContent = state.lossesBuffer[centerIdx].toFixed(3);
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

  elBtnResetVectors.addEventListener("click", () => {
    initVectors();
    syncSliderUI();
    triggerLandscapeUpdate("Zurückgesetzt auf Start");
  });

  // =========================================================================
  // SLIDER 1 (Achse α Dimension-Scroller):
  // ONLY selects the dimension and loads its stored value into Slider 2.
  // Does NOT change the 3D landscape!
  // =========================================================================
  elRngAlphaDim.addEventListener("input", () => {
    state.alphaDim = parseInt(elRngAlphaDim.value, 10);
    syncSliderUI();
    const curVal = state.alphaVec[state.alphaDim];
    flashActivityIndicator(
      "inspecting",
      `Dim #${state.alphaDim} gewählt (δ[${state.alphaDim}] = ${curVal >= 0 ? "+" : ""}${curVal.toFixed(2)})`
    );
  });

  // =========================================================================
  // SLIDER 2 (Achse α Strength in selected dimension):
  // Stores the new value in `alphaVec[alphaDim]` and updates the 3D landscape!
  // =========================================================================
  elRngAlphaStr.addEventListener("input", () => {
    const val = parseFloat(elRngAlphaStr.value);
    state.alphaVec[state.alphaDim] = val;
    syncSliderUI();
    triggerLandscapeUpdate(`Aktualisiere δ[${state.alphaDim}] = ${(val >= 0 ? "+" : "") + val.toFixed(2)}…`);
  });

  // =========================================================================
  // SLIDER 3 (Achse β Dimension-Scroller):
  // ONLY selects the dimension and loads its stored value into Slider 4.
  // Does NOT change the 3D landscape!
  // =========================================================================
  elRngBetaDim.addEventListener("input", () => {
    state.betaDim = parseInt(elRngBetaDim.value, 10);
    syncSliderUI();
    const curVal = state.betaVec[state.betaDim];
    flashActivityIndicator(
      "inspecting",
      `Dim #${state.betaDim} gewählt (η[${state.betaDim}] = ${curVal >= 0 ? "+" : ""}${curVal.toFixed(2)})`
    );
  });

  // =========================================================================
  // SLIDER 4 (Achse β Strength in selected dimension):
  // Stores the new value in `betaVec[betaDim]` and updates the 3D landscape!
  // =========================================================================
  elRngBetaStr.addEventListener("input", () => {
    const val = parseFloat(elRngBetaStr.value);
    state.betaVec[state.betaDim] = val;
    syncSliderUI();
    triggerLandscapeUpdate(`Aktualisiere η[${state.betaDim}] = ${(val >= 0 ? "+" : "") + val.toFixed(2)}…`);
  });

  elBtnRerollSeed.addEventListener("click", () => {
    state.randomSeed = Math.floor(1000 + Math.random() * 9000);
    elLblSeedInfo.textContent = `Seed #${state.randomSeed}`;
    triggerLandscapeUpdate(`Würfle Seed #${state.randomSeed}…`);
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
  initVectors();
  syncSliderUI();
  bindUIEvents();
  setupSecretGesture();
  triggerLandscapeUpdate("Initialisiere…");
}

boot();
