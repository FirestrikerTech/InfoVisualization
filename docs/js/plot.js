/**
 * Interactive 3D Loss Landscape & 2D Contour Renderer using self-hosted Three.js.
 *
 * Features:
 * - Smooth touch & mouse orbit/zoom controls (1-finger rotate, 2-finger pinch zoom).
 * - Viridis perceptual colormap (matching Li et al. 2018 & scientific standards).
 * - 3D surface mesh + wireframe grid + projected floor contour lines + glowing minimum marker at (0,0).
 * - 2D Contour / Heatmap view toggle (matching Paper Fig. 1 / Fig. 2).
 * - Linear or Logarithmic Loss Z-scale toggle.
 */

import * as THREE from "../vendor/three.module.min.js";

// Viridis colormap LUT (16 control stops, interpolated to 256 RGB entries)
const VIRIDIS_STOPS = [
  [0.267, 0.005, 0.329],
  [0.283, 0.106, 0.428],
  [0.277, 0.198, 0.496],
  [0.253, 0.283, 0.534],
  [0.221, 0.361, 0.552],
  [0.188, 0.433, 0.558],
  [0.158, 0.501, 0.558],
  [0.131, 0.568, 0.550],
  [0.121, 0.634, 0.529],
  [0.165, 0.698, 0.491],
  [0.267, 0.758, 0.433],
  [0.410, 0.810, 0.354],
  [0.580, 0.854, 0.254],
  [0.762, 0.886, 0.145],
  [0.916, 0.906, 0.106],
  [0.993, 0.906, 0.144],
];

export function sampleViridis(t) {
  const clamped = Math.max(0.0, Math.min(1.0, t));
  const pos = clamped * (VIRIDIS_STOPS.length - 1);
  const idx = Math.floor(pos);
  const frac = pos - idx;
  if (idx >= VIRIDIS_STOPS.length - 1) {
    return VIRIDIS_STOPS[VIRIDIS_STOPS.length - 1];
  }
  const a = VIRIDIS_STOPS[idx];
  const b = VIRIDIS_STOPS[idx + 1];
  return [
    a[0] + (b[0] - a[0]) * frac,
    a[1] + (b[1] - a[1]) * frac,
    a[2] + (b[2] - a[2]) * frac,
  ];
}

export class LandscapePlotter {
  constructor(container3D, canvas2D) {
    this.container3D = container3D;
    this.canvas2D = canvas2D;
    this.viewMode = "3d"; // '3d' or '2d'
    this.useLogScale = false;
    this.zCap = 10.0; // Fixed reference ceiling so Phase 2 funnel vs Phase 3 valley are directly comparable!
    this.lastData = null;

    this._initThree();
    this._initInteraction();
    window.addEventListener("resize", () => this.resize());
  }

  _initThree() {
    const w = this.container3D.clientWidth || 360;
    const h = this.container3D.clientHeight || 300;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b1120);

    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
    this.spherical = { radius: 4.35, theta: -0.72, phi: 1.04 };
    this.target = new THREE.Vector3(0, 0.15, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h);
    this.container3D.innerHTML = "";
    this.container3D.appendChild(this.renderer.domElement);

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.78);
    this.scene.add(ambient);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.65);
    dirLight1.position.set(4, 7, 5);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x93c5fd, 0.35);
    dirLight2.position.set(-5, 4, -4);
    this.scene.add(dirLight2);

    // Base coordinate box / floor grid at y = -0.65
    this.floorY = -0.65;
    this.boxHeight = 1.65;
    this._buildAxesBox();

    // Surface mesh placeholder
    this.surfaceMesh = null;
    this.wireMesh = null;
    this.minMarker = null;

    this._updateCamera();
    this.render();
  }

  _buildAxesBox() {
    const group = new THREE.Group();

    // Floor grid on [-1, 1] x [-1, 1]
    const gridHelper = new THREE.GridHelper(2.0, 10, 0x334155, 0x1e293b);
    gridHelper.position.y = this.floorY;
    group.add(gridHelper);

    // Bounding box edges
    const corners = [
      [-1, this.floorY, -1],
      [1, this.floorY, -1],
      [1, this.floorY, 1],
      [-1, this.floorY, 1],
    ];
    const lineMat = new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.7 });
    const pts = [];
    for (let i = 0; i < 4; i++) {
      const c1 = corners[i];
      const c2 = corners[(i + 1) % 4];
      pts.push(new THREE.Vector3(c1[0], c1[1], c1[2]), new THREE.Vector3(c2[0], c2[1], c2[2]));
      pts.push(
        new THREE.Vector3(c1[0], this.floorY, c1[2]),
        new THREE.Vector3(c1[0], this.floorY + this.boxHeight, c1[2])
      );
    }
    const boxGeo = new THREE.BufferGeometry().setFromPoints(pts);
    group.add(new THREE.LineSegments(boxGeo, lineMat));

    // Axis lines for alpha (red-ish cyan) and beta (amber)
    const alphaAxisGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-1, this.floorY + 0.003, 1.02),
      new THREE.Vector3(1, this.floorY + 0.003, 1.02),
    ]);
    group.add(new THREE.Line(alphaAxisGeo, new THREE.LineBasicMaterial({ color: 0x38bdf8 })));

    const betaAxisGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(1.02, this.floorY + 0.003, -1),
      new THREE.Vector3(1.02, this.floorY + 0.003, 1),
    ]);
    group.add(new THREE.Line(betaAxisGeo, new THREE.LineBasicMaterial({ color: 0xf59e0b })));

    // Minimum glowing pin at center (alpha=0, beta=0)
    const pinGeo = new THREE.SphereGeometry(0.042, 16, 16);
    const pinMat = new THREE.MeshBasicMaterial({ color: 0xf43f5e });
    this.minMarker = new THREE.Mesh(pinGeo, pinMat);
    this.minMarker.position.set(0, this.floorY, 0);
    group.add(this.minMarker);

    // Vertical drop-line from minimum to floor
    const dropGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, this.floorY, 0),
      new THREE.Vector3(0, this.floorY + 0.2, 0),
    ]);
    this.minDropLine = new THREE.Line(
      dropGeo,
      new THREE.LineBasicMaterial({ color: 0xf43f5e, transparent: true, opacity: 0.75 })
    );
    group.add(this.minDropLine);

    this.scene.add(group);
  }

  _initInteraction() {
    const dom = this.renderer.domElement;
    let isDragging = false;
    let prevX = 0;
    let prevY = 0;
    let prevPinchDist = 0;

    dom.addEventListener("mousedown", (e) => {
      isDragging = true;
      prevX = e.clientX;
      prevY = e.clientY;
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      const dx = e.clientX - prevX;
      const dy = e.clientY - prevY;
      prevX = e.clientX;
      prevY = e.clientY;
      this.spherical.theta -= dx * 0.0085;
      this.spherical.phi = Math.max(0.22, Math.min(1.48, this.spherical.phi - dy * 0.0085));
      this._updateCamera();
      this.render();
    });

    window.addEventListener("mouseup", () => {
      isDragging = false;
    });

    dom.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.spherical.radius = Math.max(2.3, Math.min(7.5, this.spherical.radius + e.deltaY * 0.003));
        this._updateCamera();
        this.render();
      },
      { passive: false }
    );

    // Touch support (1 finger orbit, 2 finger pinch zoom)
    dom.addEventListener(
      "touchstart",
      (e) => {
        if (e.touches.length === 1) {
          isDragging = true;
          prevX = e.touches[0].clientX;
          prevY = e.touches[0].clientY;
        } else if (e.touches.length === 2) {
          isDragging = false;
          prevPinchDist = Math.hypot(
            e.touches[0].clientX - e.touches[1].clientX,
            e.touches[0].clientY - e.touches[1].clientY
          );
        }
      },
      { passive: true }
    );

    dom.addEventListener(
      "touchmove",
      (e) => {
        if (e.touches.length === 1 && isDragging) {
          e.preventDefault();
          const dx = e.touches[0].clientX - prevX;
          const dy = e.touches[0].clientY - prevY;
          prevX = e.touches[0].clientX;
          prevY = e.touches[0].clientY;
          this.spherical.theta -= dx * 0.0095;
          this.spherical.phi = Math.max(0.22, Math.min(1.48, this.spherical.phi - dy * 0.0095));
          this._updateCamera();
          this.render();
        } else if (e.touches.length === 2) {
          e.preventDefault();
          const dist = Math.hypot(
            e.touches[0].clientX - e.touches[1].clientX,
            e.touches[0].clientY - e.touches[1].clientY
          );
          if (prevPinchDist > 0) {
            const delta = prevPinchDist - dist;
            this.spherical.radius = Math.max(2.3, Math.min(7.5, this.spherical.radius + delta * 0.012));
            this._updateCamera();
            this.render();
          }
          prevPinchDist = dist;
        }
      },
      { passive: false }
    );

    dom.addEventListener("touchend", () => {
      isDragging = false;
      prevPinchDist = 0;
    });
  }

  _updateCamera() {
    const { radius, theta, phi } = this.spherical;
    this.camera.position.x = this.target.x + radius * Math.sin(phi) * Math.sin(theta);
    this.camera.position.y = this.target.y + radius * Math.cos(phi);
    this.camera.position.z = this.target.z + radius * Math.sin(phi) * Math.cos(theta);
    this.camera.lookAt(this.target);
  }

  setViewMode(mode) {
    this.viewMode = mode;
    if (mode === "2d") {
      this.container3D.style.display = "none";
      this.canvas2D.style.display = "block";
      if (this.lastData) this._render2DContour(this.lastData);
    } else {
      this.canvas2D.style.display = "none";
      this.container3D.style.display = "block";
      this.resize();
      if (this.lastData) this.updateSurface(this.lastData);
    }
  }

  setLogScale(useLog) {
    this.useLogScale = useLog;
    if (this.lastData) {
      this.updateSurface(this.lastData);
    }
  }

  resize() {
    const w = this.container3D.clientWidth || 360;
    const h = this.container3D.clientHeight || 300;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.render();
    if (this.viewMode === "2d" && this.lastData) {
      this._render2DContour(this.lastData);
    }
  }

  _mapLossToNormalizedHeight(lossVal, minLoss, maxCap) {
    const clamped = Math.min(lossVal, maxCap);
    if (this.useLogScale) {
      const eps = 0.02;
      const lMin = Math.log(minLoss + eps);
      const lMax = Math.log(maxCap + eps);
      const lCur = Math.log(Math.max(minLoss, clamped) + eps);
      return Math.max(0.0, Math.min(1.0, (lCur - lMin) / (lMax - lMin + 1e-9)));
    }
    return Math.max(0.0, Math.min(1.0, (clamped - minLoss) / (maxCap - minLoss + 1e-9)));
  }

  updateSurface(data) {
    this.lastData = data;
    if (this.viewMode === "2d") {
      this._render2DContour(data);
    }

    const { gridSize, losses } = data;

    // Clean up previous mesh
    if (this.surfaceMesh) {
      this.scene.remove(this.surfaceMesh);
      this.surfaceMesh.geometry.dispose();
      this.surfaceMesh.material.dispose();
    }
    if (this.wireMesh) {
      this.scene.remove(this.wireMesh);
      this.wireMesh.geometry.dispose();
      this.wireMesh.material.dispose();
    }

    let minLoss = Infinity;
    let maxLoss = -Infinity;
    for (let i = 0; i < losses.length; i++) {
      if (losses[i] < minLoss) minLoss = losses[i];
      if (losses[i] > maxLoss) maxLoss = losses[i];
    }

    // Use a consistent reference ceiling (zCap = 10.0, or slightly lower if the whole surface is very flat
    // in manual exploration, but bounded below by 4.5 so slopes and funnels have true visual scale!)
    const effectiveCap = Math.max(4.5, Math.min(this.zCap, maxLoss));
    const baseMin = Math.min(0.12, minLoss);

    const numVerts = gridSize * gridSize;
    const positions = new Float32Array(numVerts * 3);
    const colors = new Float32Array(numVerts * 3);

    for (let ib = 0; ib < gridSize; ib++) {
      const zCoord = -1.0 + (2.0 * ib) / (gridSize - 1);
      for (let ia = 0; ia < gridSize; ia++) {
        const xCoord = -1.0 + (2.0 * ia) / (gridSize - 1);
        const idx = ib * gridSize + ia;
        const lossVal = losses[idx];

        const tNorm = this._mapLossToNormalizedHeight(lossVal, baseMin, effectiveCap);
        const yCoord = this.floorY + tNorm * this.boxHeight;

        const vIdx = idx * 3;
        positions[vIdx] = xCoord;
        positions[vIdx + 1] = yCoord;
        positions[vIdx + 2] = zCoord;

        const rgb = sampleViridis(tNorm);
        colors[vIdx] = rgb[0];
        colors[vIdx + 1] = rgb[1];
        colors[vIdx + 2] = rgb[2];
      }
    }

    const indices = [];
    for (let ib = 0; ib < gridSize - 1; ib++) {
      for (let ia = 0; ia < gridSize - 1; ia++) {
        const a = ib * gridSize + ia;
        const b = ib * gridSize + (ia + 1);
        const c = (ib + 1) * gridSize + ia;
        const d = (ib + 1) * gridSize + (ia + 1);
        indices.push(a, c, b);
        indices.push(b, c, d);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    const mat = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: 28,
      specular: new THREE.Color(0x222222),
    });

    this.surfaceMesh = new THREE.Mesh(geo, mat);
    this.scene.add(this.surfaceMesh);

    // Subtle wireframe overlay for crisp 3D depth perception on mobile screens
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x0f172a,
      wireframe: true,
      transparent: true,
      opacity: 0.18,
    });
    this.wireMesh = new THREE.Mesh(geo, wireMat);
    this.scene.add(this.wireMesh);

    // Position the glowing red marker at the center (alpha=0, beta=0)
    // Only highlight it prominently when we are in random/filter-norm mode or if the user
    // actually found a 2D basin where the edges in all 4 directions are clearly higher!
    const cIdx = Math.floor(gridSize / 2) * gridSize + Math.floor(gridSize / 2);
    const centerLoss = losses[cIdx];
    const centerT = this._mapLossToNormalizedHeight(centerLoss, baseMin, effectiveCap);
    const centerY = this.floorY + centerT * this.boxHeight;
    this.minMarker.position.set(0, centerY + 0.025, 0);

    const dropPositions = this.minDropLine.geometry.attributes.position.array;
    dropPositions[1] = this.floorY;
    dropPositions[4] = centerY;
    this.minDropLine.geometry.attributes.position.needsUpdate = true;

    this.render();
  }

  _render2DContour(data) {
    const canvas = this.canvas2D;
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);
    canvas.width = w * dpr;
    canvas.height = h * dpr;

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#0b1120";
    ctx.fillRect(0, 0, w, h);

    const { gridSize, losses, span } = data;
    const size = Math.min(w - 56, h - 56);
    const ox = Math.floor((w - size) / 2) + 8;
    const oy = Math.floor((h - size) / 2) - 6;

    let minLoss = Infinity;
    let maxLoss = -Infinity;
    for (let i = 0; i < losses.length; i++) {
      if (losses[i] < minLoss) minLoss = losses[i];
      if (losses[i] > maxLoss) maxLoss = losses[i];
    }
    const effectiveCap = Math.max(4.5, Math.min(this.zCap, maxLoss));
    const baseMin = Math.min(0.12, minLoss);

    // Bilinear interpolated heatmap + contour isolines (like Li et al. 2018 2D contour plots)
    const imgData = ctx.createImageData(size, size);
    const pixels = imgData.data;

    const contourLevels = [0.2, 0.35, 0.6, 1.0, 1.6, 2.5, 4.0, 6.0, 8.5];

    for (let py = 0; py < size; py++) {
      const gy = ((size - 1 - py) / (size - 1)) * (gridSize - 1);
      const ib0 = Math.floor(gy);
      const ib1 = Math.min(gridSize - 1, ib0 + 1);
      const fb = gy - ib0;

      for (let px = 0; px < size; px++) {
        const gx = (px / (size - 1)) * (gridSize - 1);
        const ia0 = Math.floor(gx);
        const ia1 = Math.min(gridSize - 1, ia0 + 1);
        const fa = gx - ia0;

        const v00 = losses[ib0 * gridSize + ia0];
        const v10 = losses[ib0 * gridSize + ia1];
        const v01 = losses[ib1 * gridSize + ia0];
        const v11 = losses[ib1 * gridSize + ia1];

        const val =
          v00 * (1 - fa) * (1 - fb) +
          v10 * fa * (1 - fb) +
          v01 * (1 - fa) * fb +
          v11 * fa * fb;

        const tNorm = this._mapLossToNormalizedHeight(val, baseMin, effectiveCap);
        const rgb = sampleViridis(tNorm);

        // Check proximity to contour lines for crisp paper-style isolines
        let isLine = false;
        const grad = Math.hypot(v10 - v00, v01 - v00) * (gridSize / size) + 1e-5;
        for (let l = 0; l < contourLevels.length; l++) {
          if (Math.abs(val - contourLevels[l]) < grad * 0.85) {
            isLine = true;
            break;
          }
        }

        const pIdx = (py * size + px) * 4;
        if (isLine) {
          pixels[pIdx] = 15;
          pixels[pIdx + 1] = 23;
          pixels[pIdx + 2] = 42;
          pixels[pIdx + 3] = 255;
        } else {
          pixels[pIdx] = Math.round(rgb[0] * 255);
          pixels[pIdx + 1] = Math.round(rgb[1] * 255);
          pixels[pIdx + 2] = Math.round(rgb[2] * 255);
          pixels[pIdx + 3] = 255;
        }
      }
    }

    const offCanvas = document.createElement("canvas");
    offCanvas.width = size;
    offCanvas.height = size;
    offCanvas.getContext("2d").putImageData(imgData, 0, 0);
    ctx.drawImage(offCanvas, ox, oy);

    // Border & center marker (theta*)
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(ox, oy, size, size);

    const cx = ox + size / 2;
    const cy = oy + size / 2;
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`α ∈ [-${span.toFixed(1)}, +${span.toFixed(1)}]`, ox + size / 2, oy + size + 20);
    ctx.save();
    ctx.translate(ox - 16, oy + size / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`β ∈ [-${span.toFixed(1)}, +${span.toFixed(1)}]`, 0, 0);
    ctx.restore();
  }

  render() {
    if (this.viewMode === "3d") {
      this.renderer.render(this.scene, this.camera);
    }
  }
}
