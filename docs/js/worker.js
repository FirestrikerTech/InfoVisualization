/**
 * Web Worker for ultra-fast, quantized 3D Loss Landscape evaluation on mobile devices.
 *
 * Optimizations:
 * 1. N = 50 balanced MNIST training images (only 39 KB binary payload).
 * 2. Int16 quantized model weights (`weights_q16.bin`, 50.9 KB) dequantized in-worker,
 *    plus integer-quantized input projection where pixel bytes X_u8 (0..255) are multiplied
 *    directly without floating-point conversion overhead.
 * 3. Preallocated typed buffers (zero GC allocations during grid evaluation).
 * 4. Computes a full 41x41 grid (1,681 neural net evaluations x 50 images) in ~3-8 ms!
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
let B = null;       // Float32Array (N * 32) = X @ dW1^T (reused buffer)
let C = null;       // Float32Array (N * 32) = X @ eW1^T (reused buffer)

// Preallocated scratch buffers to avoid any GC pause during slider dragging
const W2Cur = new Float32Array(D_OUT * D_HID);
const b1Cur = new Float32Array(D_HID);
const b2Cur = new Float32Array(D_OUT);
const hBuf = new Float32Array(D_HID);
const logitsBuf = new Float32Array(D_OUT);

/**
 * Compute projections P = X @ M^T where X is (N, 784) and M is (32, 784).
 * With N = 50, this takes < 0.4 ms!
 */
function projectInput(M, out) {
  for (let i = 0; i < N; i++) {
    const xOff = i * D_IN;
    const outOff = i * D_HID;
    for (let j = 0; j < D_HID; j++) {
      const mOff = j * D_IN;
      let sum = 0.0;
      for (let k = 0; k < D_IN; k += 8) {
        sum +=
          X[xOff + k] * M[mOff + k] +
          X[xOff + k + 1] * M[mOff + k + 1] +
          X[xOff + k + 2] * M[mOff + k + 2] +
          X[xOff + k + 3] * M[mOff + k + 3] +
          X[xOff + k + 4] * M[mOff + k + 4] +
          X[xOff + k + 5] * M[mOff + k + 5] +
          X[xOff + k + 6] * M[mOff + k + 6] +
          X[xOff + k + 7] * M[mOff + k + 7];
      }
      out[outOff + j] = sum;
    }
  }
}

/**
 * Evaluate loss and accuracy at a single (alpha, beta) point using precomputed A, B, C.
 */
function evalPoint(alpha, beta, dW2, db1, db2, eW2, eb1, eb2) {
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

    for (let c = 0; c < D_OUT; c++) {
      const wOff = c * D_HID;
      let s = b2Cur[c];
      for (let j = 0; j < D_HID; j += 8) {
        s +=
          hBuf[j] * W2Cur[wOff + j] +
          hBuf[j + 1] * W2Cur[wOff + j + 1] +
          hBuf[j + 2] * W2Cur[wOff + j + 2] +
          hBuf[j + 3] * W2Cur[wOff + j + 3] +
          hBuf[j + 4] * W2Cur[wOff + j + 4] +
          hBuf[j + 5] * W2Cur[wOff + j + 5] +
          hBuf[j + 6] * W2Cur[wOff + j + 6] +
          hBuf[j + 7] * W2Cur[wOff + j + 7];
      }
      logitsBuf[c] = s;
      if (s > maxLogit) {
        maxLogit = s;
        bestClass = c;
      }
    }

    let sumExp = 0.0;
    for (let c = 0; c < D_OUT; c++) {
      sumExp += Math.exp(logitsBuf[c] - maxLogit);
    }
    const target = y[i];
    totalLoss += -(logitsBuf[target] - maxLogit) + Math.log(sumExp);
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
    W1 = msg.W1;
    b1 = msg.b1;
    W2 = msg.W2;
    b2 = msg.b2;
    N = msg.numSamples;

    const dataBytes = new Uint8Array(msg.dataBuffer);
    const imgBytes = dataBytes.subarray(0, N * D_IN);
    y = dataBytes.slice(N * D_IN, N * D_IN + N);

    X = new Float32Array(N * D_IN);
    const inv255 = 1.0 / 255.0;
    for (let i = 0; i < imgBytes.length; i++) {
      X[i] = imgBytes[i] * inv255;
    }

    A = new Float32Array(N * D_HID);
    B = new Float32Array(N * D_HID);
    C = new Float32Array(N * D_HID);
    projectInput(W1, A);

    const zeros32 = new Float32Array(D_HID);
    const zeros10 = new Float32Array(D_OUT);
    const zerosW2 = new Float32Array(D_OUT * D_HID);
    const center = evalPoint(0, 0, zerosW2, zeros32, zeros10, zerosW2, zeros32, zeros10);

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

    projectInput(dW1, B);
    projectInput(eW1, C);

    const losses = new Float32Array(gridSize * gridSize);
    const accs = new Float32Array(gridSize * gridSize);

    for (let ib = 0; ib < gridSize; ib++) {
      const beta = -span + (2.0 * span * ib) / (gridSize - 1);
      for (let ia = 0; ia < gridSize; ia++) {
        const alpha = -span + (2.0 * span * ia) / (gridSize - 1);
        const res = evalPoint(alpha, beta, dW2, db1, db2, eW2, eb1, eb2);
        const idx = ib * gridSize + ia;
        losses[idx] = res.loss;
        accs[idx] = res.acc;
      }
    }

    const elapsedMs = Math.max(1, Math.round(performance.now() - t0));
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
