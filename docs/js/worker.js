/**
 * Web Worker for fast, progressive 3D Loss Landscape evaluation on mobile devices.
 *
 * Mathematical model (Li et al., 2018 - "Visualizing the Loss Landscape of Neural Nets"):
 *   f(alpha, beta) = L(theta* + alpha * delta + beta * eta)
 *
 * Network architecture: 784 -> 32 (ReLU) -> 10 (Cross-Entropy Loss)
 * Key speedup:
 *   Since layer 1 is linear before ReLU:
 *     X * (W1 + alpha * dW1 + beta * eW1)^T = A + alpha * B + beta * C
 *   where:
 *     A = X * W1^T   (precomputed ONCE at load time, 500 x 32)
 *     B = X * dW1^T  (precomputed ONCE per direction pair, 500 x 32)
 *     C = X * eW1^T  (precomputed ONCE per direction pair, 500 x 32)
 *   Each grid point (alpha, beta) then only requires adding 32-dim vectors, ReLU,
 *   and a tiny 32 -> 10 matrix multiplication!
 */

let N = 0;
const D_IN = 784;
const D_HID = 32;
const D_OUT = 10;

let X = null;       // Float32Array (N * 784)
let y = null;       // Uint8Array (N)
let W1 = null;      // Float32Array (32 * 784)
let b1 = null;      // Float32Array (32)
let W2 = null;      // Float32Array (10 * 32)
let b2 = null;      // Float32Array (10)
let A = null;       // Float32Array (N * 32) = X @ W1^T

/**
 * Compute projections P = X @ M^T where X is (N, 784) and M is (32, 784).
 * Uses cache-friendly loop order over samples and hidden units.
 */
function projectInput(M, out) {
  for (let i = 0; i < N; i++) {
    const xOff = i * D_IN;
    const outOff = i * D_HID;
    for (let j = 0; j < D_HID; j++) {
      const mOff = j * D_IN;
      let sum = 0.0;
      for (let k = 0; k < D_IN; k += 4) {
        sum +=
          X[xOff + k] * M[mOff + k] +
          X[xOff + k + 1] * M[mOff + k + 1] +
          X[xOff + k + 2] * M[mOff + k + 2] +
          X[xOff + k + 3] * M[mOff + k + 3];
      }
      out[outOff + j] = sum;
    }
  }
}

/**
 * Evaluate loss and accuracy at a single (alpha, beta) point using precomputed A, B, C.
 */
function evalPoint(alpha, beta, B, C, dW2, db1, db2, eW2, eb1, eb2, W2Cur, b1Cur, b2Cur, hBuf) {
  for (let j = 0; j < D_HID; j++) {
    b1Cur[j] = b1[j] + alpha * db1[j] + beta * eb1[j];
  }
  const nW2 = D_OUT * D_HID;
  for (let idx = 0; idx < nW2; idx++) {
    W2Cur[idx] = W2[idx] + alpha * dW2[idx] + beta * eW2[idx];
  }
  for (let c = 0; c < D_OUT; c++) {
    b2Cur[c] = b2[c] + alpha * db2[c] + beta * eb2[c];
  }

  let totalLoss = 0.0;
  let correct = 0;

  for (let i = 0; i < N; i++) {
    const off = i * D_HID;
    for (let j = 0; j < D_HID; j++) {
      const v = A[off + j] + alpha * B[off + j] + beta * C[off + j] + b1Cur[j];
      hBuf[j] = v > 0.0 ? v : 0.0;
    }

    let maxLogit = -1e30;
    let bestClass = 0;
    const logits = new Float32Array(D_OUT);

    for (let c = 0; c < D_OUT; c++) {
      const wOff = c * D_HID;
      let s = b2Cur[c];
      for (let j = 0; j < D_HID; j += 4) {
        s +=
          hBuf[j] * W2Cur[wOff + j] +
          hBuf[j + 1] * W2Cur[wOff + j + 1] +
          hBuf[j + 2] * W2Cur[wOff + j + 2] +
          hBuf[j + 3] * W2Cur[wOff + j + 3];
      }
      logits[c] = s;
      if (s > maxLogit) {
        maxLogit = s;
        bestClass = c;
      }
    }

    let sumExp = 0.0;
    for (let c = 0; c < D_OUT; c++) {
      sumExp += Math.exp(logits[c] - maxLogit);
    }
    const target = y[i];
    totalLoss += -(logits[target] - maxLogit) + Math.log(sumExp);
    if (bestClass === target) correct++;
  }

  return {
    loss: totalLoss / N,
    acc: correct / N,
  };
}

self.onmessage = function (e) {
  const msg = e.data;

  if (msg.type === "init") {
    const weights = new Float32Array(msg.weightsBuffer);
    const dataBytes = new Uint8Array(msg.dataBuffer);
    N = msg.numSamples;

    const o = msg.offsets;
    W1 = weights.slice(o.W1[0], o.W1[1]);
    b1 = weights.slice(o.b1[0], o.b1[1]);
    W2 = weights.slice(o.W2[0], o.W2[1]);
    b2 = weights.slice(o.b2[0], o.b2[1]);

    const imgBytes = dataBytes.subarray(0, N * D_IN);
    y = dataBytes.slice(N * D_IN, N * D_IN + N);

    X = new Float32Array(N * D_IN);
    const inv255 = 1.0 / 255.0;
    for (let i = 0; i < imgBytes.length; i++) {
      X[i] = imgBytes[i] * inv255;
    }

    A = new Float32Array(N * D_HID);
    projectInput(W1, A);

    // Evaluate center (0, 0) as immediate sanity check
    const zeros32 = new Float32Array(D_HID);
    const zeros10 = new Float32Array(D_OUT);
    const zerosW2 = new Float32Array(D_OUT * D_HID);
    const W2Cur = new Float32Array(D_OUT * D_HID);
    const b1Cur = new Float32Array(D_HID);
    const b2Cur = new Float32Array(D_OUT);
    const hBuf = new Float32Array(D_HID);
    const center = evalPoint(
      0,
      0,
      A,
      A,
      zerosW2,
      zeros32,
      zeros10,
      zerosW2,
      zeros32,
      zeros10,
      W2Cur,
      b1Cur,
      b2Cur,
      hBuf
    );

    self.postMessage({
      type: "ready",
      centerLoss: center.loss,
      centerAcc: center.acc,
    });
    return;
  }

  if (msg.type === "compute") {
    const t0 = performance.now();
    const {
      jobId,
      gridSize,
      span,
      dW1,
      db1,
      dW2,
      db2,
      eW1,
      eb1,
      eW2,
      eb2,
    } = msg;

    const B = new Float32Array(N * D_HID);
    const C = new Float32Array(N * D_HID);
    projectInput(dW1, B);
    projectInput(eW1, C);

    const W2Cur = new Float32Array(D_OUT * D_HID);
    const b1Cur = new Float32Array(D_HID);
    const b2Cur = new Float32Array(D_OUT);
    const hBuf = new Float32Array(D_HID);

    // Progressive Stage 1: coarse 21x21 preview if gridSize >= 35
    if (gridSize >= 35) {
      const coarseSize = 21;
      const coarseLosses = new Float32Array(coarseSize * coarseSize);
      const coarseAccs = new Float32Array(coarseSize * coarseSize);
      for (let ib = 0; ib < coarseSize; ib++) {
        const beta = -span + (2.0 * span * ib) / (coarseSize - 1);
        for (let ia = 0; ia < coarseSize; ia++) {
          const alpha = -span + (2.0 * span * ia) / (coarseSize - 1);
          const res = evalPoint(
            alpha,
            beta,
            B,
            C,
            dW2,
            db1,
            db2,
            eW2,
            eb1,
            eb2,
            W2Cur,
            b1Cur,
            b2Cur,
            hBuf
          );
          const idx = ib * coarseSize + ia;
          coarseLosses[idx] = res.loss;
          coarseAccs[idx] = res.acc;
        }
      }
      self.postMessage({
        type: "preview",
        jobId,
        gridSize: coarseSize,
        span,
        losses: coarseLosses,
        accs: coarseAccs,
      });
    }

    // Full resolution grid
    const losses = new Float32Array(gridSize * gridSize);
    const accs = new Float32Array(gridSize * gridSize);

    for (let ib = 0; ib < gridSize; ib++) {
      const beta = -span + (2.0 * span * ib) / (gridSize - 1);
      for (let ia = 0; ia < gridSize; ia++) {
        const alpha = -span + (2.0 * span * ia) / (gridSize - 1);
        const res = evalPoint(
          alpha,
          beta,
          B,
          C,
          dW2,
          db1,
          db2,
          eW2,
          eb1,
          eb2,
          W2Cur,
          b1Cur,
          b2Cur,
          hBuf
        );
        const idx = ib * gridSize + ia;
        losses[idx] = res.loss;
        accs[idx] = res.acc;
      }
    }

    const elapsedMs = Math.round(performance.now() - t0);
    const centerIdx = Math.floor(gridSize / 2) * gridSize + Math.floor(gridSize / 2);

    self.postMessage({
      type: "result",
      jobId,
      gridSize,
      span,
      losses,
      accs,
      centerLoss: losses[centerIdx],
      centerAcc: accs[centerIdx],
      elapsedMs,
    });
  }
};
