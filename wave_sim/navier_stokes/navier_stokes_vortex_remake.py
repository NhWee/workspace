"""Create a clean real-time Navier-Stokes vortex/foam wave viewer.

This file writes a standalone HTML experiment. It keeps the physics lightweight:

du/dt + (u dot grad)u = -grad(p) + viscosity * laplacian(u) + force
div(u) = 0
d2 eta/dt2 = c^2 laplacian(eta) + swirl/speed forcing - damping * d eta/dt

The goal is not full 3D CFD. It is a focused 2.5D free-surface experiment where
vortices, foam, and spray are easy to see and tune in the browser.
"""

import argparse
from pathlib import Path


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Navier-Stokes Vortex Foam Remake</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      overflow: hidden;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #06101a;
      color: #e8f5ff;
    }
    #stage {
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
      touch-action: none;
      cursor: crosshair;
    }
    .toolbar {
      position: fixed;
      left: 16px;
      top: 16px;
      width: min(390px, calc(100vw - 32px));
      padding: 14px;
      border: 1px solid rgba(156, 214, 255, 0.18);
      background: rgba(5, 13, 24, 0.74);
      backdrop-filter: blur(12px);
      border-radius: 8px;
      box-shadow: 0 16px 60px rgba(0, 0, 0, 0.34);
    }
    h1 {
      margin: 0 0 10px;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 720;
      letter-spacing: 0;
    }
    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px 12px;
    }
    label {
      display: grid;
      gap: 4px;
      color: rgba(232, 245, 255, 0.76);
      font-size: 12px;
    }
    label span {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    output { color: #bceaff; font-variant-numeric: tabular-nums; }
    input[type="range"] { width: 100%; accent-color: #61c9ff; }
    .buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    button {
      min-height: 32px;
      color: #e8f5ff;
      border: 1px solid rgba(150, 213, 255, 0.24);
      background: rgba(35, 79, 119, 0.62);
      border-radius: 7px;
      padding: 7px 10px;
      cursor: pointer;
      font-weight: 650;
    }
    button:hover { background: rgba(52, 112, 166, 0.74); }
    .readout {
      position: fixed;
      left: 16px;
      bottom: 14px;
      color: rgba(225, 243, 255, 0.78);
      font: 12px/1.35 ui-monospace, SFMono-Regular, Consolas, monospace;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.65);
    }
    .hint {
      position: fixed;
      right: 16px;
      bottom: 14px;
      max-width: min(430px, calc(100vw - 32px));
      color: rgba(225, 243, 255, 0.68);
      font-size: 12px;
      line-height: 1.45;
      text-align: right;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.65);
    }
    @media (max-width: 760px) {
      .controls { grid-template-columns: 1fr; }
      .hint { display: none; }
    }
  </style>
</head>
<body>
  <canvas id="stage"></canvas>
  <section class="toolbar" aria-label="simulation controls">
    <h1>Navier-Stokes Vortex Foam Remake</h1>
    <div class="controls">
      <label><span>Wind <output id="windValue"></output></span><input id="wind" type="range" min="0" max="2.6" step="0.01" value="1.25"></label>
      <label><span>Vortex <output id="vortexValue"></output></span><input id="vortex" type="range" min="0" max="8" step="0.05" value="5.8"></label>
      <label><span>Wave <output id="waveValue"></output></span><input id="wave" type="range" min="0.06" max="0.42" step="0.005" value="0.22"></label>
      <label><span>Coupling <output id="couplingValue"></output></span><input id="coupling" type="range" min="0" max="0.010" step="0.0001" value="0.0048"></label>
      <label><span>Viscosity <output id="viscosityValue"></output></span><input id="viscosity" type="range" min="0" max="0.002" step="0.00001" value="0.00016"></label>
      <label><span>Foam <output id="foamValue"></output></span><input id="foam" type="range" min="0" max="0.40" step="0.002" value="0.18"></label>
    </div>
    <div class="buttons">
      <button id="pause">Pause</button>
      <button id="reset">Reset</button>
      <button id="kick">Kick Vortices</button>
      <button id="vectors">Flow Off</button>
      <button id="swirl">Swirl Overlay Off</button>
    </div>
  </section>
  <div class="readout" id="readout"></div>
  <div class="hint">Drag on the water to inject local velocity. This is a 2D incompressible flow coupled to a 3D-looking height field, not full volumetric CFD.</div>

  <script>
  const TAU = Math.PI * 2;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const smoothstep = (x) => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
  const ix2 = (x, y, n) => x + (n + 2) * y;

  function rand(seed) {
    let a = seed >>> 0;
    return () => {
      a += 0x6D2B79F5;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class Fluid {
    constructor(opts = {}) {
      this.n = opts.n || 76;
      this.dt = opts.dt || 0.018;
      this.len = (this.n + 2) * (this.n + 2);
      this.time = 0;
      this.wind = 1.25;
      this.vortexStrength = 5.8;
      this.viscosity = 0.00016;
      this.diffusion = 0.00005;
      this.pressureIterations = 34;
      this.diffuseIterations = 7;
      this.waveSpeed = 0.22;
      this.surfaceCoupling = 0.0048;
      this.surfaceDamping = 0.989;
      this.foamBirth = 0.18;
      this.foamDecay = 0.984;
      this.maxSpray = 1400;
      this.random = rand(opts.seed || 20260619);
      this.u = new Float32Array(this.len);
      this.v = new Float32Array(this.len);
      this.u0 = new Float32Array(this.len);
      this.v0 = new Float32Array(this.len);
      this.p = new Float32Array(this.len);
      this.div = new Float32Array(this.len);
      this.eta = new Float32Array(this.len);
      this.eta0 = new Float32Array(this.len);
      this.etaVel = new Float32Array(this.len);
      this.foam = new Float32Array(this.len);
      this.foam0 = new Float32Array(this.len);
      this.curl = new Float32Array(this.len);
      this.spray = [];
      this.mouse = { active: false, x: 0.5, y: 0.5, px: 0.5, py: 0.5 };
      this.seed();
    }

    ix(x, y) { return ix2(x, y, this.n); }

    seed() {
      const n = this.n;
      this.u.fill(0); this.v.fill(0); this.eta.fill(0); this.etaVel.fill(0); this.foam.fill(0); this.spray.length = 0;
      for (let y = 1; y <= n; y++) {
        for (let x = 1; x <= n; x++) {
          const nx = x / n, ny = y / n, id = this.ix(x, y);
          this.eta[id] = 0.025 * Math.sin(13 * nx + 2.6 * Math.sin(8 * ny)) + 0.010 * Math.sin(TAU * (2.4 * nx + 1.2 * ny));
          this.u[id] = 0.16 * Math.sin(TAU * ny);
          this.v[id] = 0.09 * Math.cos(TAU * nx);
        }
      }
      this.addVortex(0.30, 0.52, 7.8, 0.19);
      this.addVortex(0.68, 0.47, -8.4, 0.17);
      this.addVortex(0.50, 0.72, 5.2, 0.15);
      this.addVortex(0.42, 0.30, -3.8, 0.13);
      this.project();
    }

    setBoundary(b, a) {
      const n = this.n;
      for (let i = 1; i <= n; i++) {
        a[this.ix(0, i)] = b === 1 ? -a[this.ix(1, i)] : a[this.ix(1, i)];
        a[this.ix(n + 1, i)] = b === 1 ? -a[this.ix(n, i)] : a[this.ix(n, i)];
        a[this.ix(i, 0)] = b === 2 ? -a[this.ix(i, 1)] : a[this.ix(i, 1)];
        a[this.ix(i, n + 1)] = b === 2 ? -a[this.ix(i, n)] : a[this.ix(i, n)];
      }
      a[this.ix(0, 0)] = 0.5 * (a[this.ix(1, 0)] + a[this.ix(0, 1)]);
      a[this.ix(0, n + 1)] = 0.5 * (a[this.ix(1, n + 1)] + a[this.ix(0, n)]);
      a[this.ix(n + 1, 0)] = 0.5 * (a[this.ix(n, 0)] + a[this.ix(n + 1, 1)]);
      a[this.ix(n + 1, n + 1)] = 0.5 * (a[this.ix(n, n + 1)] + a[this.ix(n + 1, n)]);
    }

    bilerp(a, x, y) {
      const n = this.n;
      x = clamp(x, 0.5, n + 0.5); y = clamp(y, 0.5, n + 0.5);
      const x0 = Math.floor(x), y0 = Math.floor(y), x1 = x0 + 1, y1 = y0 + 1;
      const sx = x - x0, sy = y - y0;
      const aa = lerp(a[this.ix(x0, y0)], a[this.ix(x1, y0)], sx);
      const bb = lerp(a[this.ix(x0, y1)], a[this.ix(x1, y1)], sx);
      return lerp(aa, bb, sy);
    }

    diffuse(b, dst, src, diff) {
      const n = this.n, a = this.dt * diff * n * n;
      dst.set(src);
      for (let k = 0; k < this.diffuseIterations; k++) {
        for (let y = 1; y <= n; y++) for (let x = 1; x <= n; x++) {
          const id = this.ix(x, y);
          dst[id] = (src[id] + a * (dst[this.ix(x - 1, y)] + dst[this.ix(x + 1, y)] + dst[this.ix(x, y - 1)] + dst[this.ix(x, y + 1)])) / (1 + 4 * a);
        }
        this.setBoundary(b, dst);
      }
    }

    advect(b, dst, src, u, v) {
      const n = this.n, dt0 = this.dt * n;
      for (let y = 1; y <= n; y++) for (let x = 1; x <= n; x++) {
        const id = this.ix(x, y);
        dst[id] = this.bilerp(src, x - dt0 * u[id], y - dt0 * v[id]);
      }
      this.setBoundary(b, dst);
    }

    project() {
      const n = this.n;
      for (let y = 1; y <= n; y++) for (let x = 1; x <= n; x++) {
        const id = this.ix(x, y);
        this.div[id] = -0.5 * (this.u[this.ix(x + 1, y)] - this.u[this.ix(x - 1, y)] + this.v[this.ix(x, y + 1)] - this.v[this.ix(x, y - 1)]) / n;
        this.p[id] = 0;
      }
      this.setBoundary(0, this.div); this.setBoundary(0, this.p);
      for (let k = 0; k < this.pressureIterations; k++) {
        for (let y = 1; y <= n; y++) for (let x = 1; x <= n; x++) {
          this.p[this.ix(x, y)] = (this.div[this.ix(x, y)] + this.p[this.ix(x - 1, y)] + this.p[this.ix(x + 1, y)] + this.p[this.ix(x, y - 1)] + this.p[this.ix(x, y + 1)]) / 4;
        }
        this.setBoundary(0, this.p);
      }
      for (let y = 1; y <= n; y++) for (let x = 1; x <= n; x++) {
        const id = this.ix(x, y);
        this.u[id] -= 0.5 * n * (this.p[this.ix(x + 1, y)] - this.p[this.ix(x - 1, y)]);
        this.v[id] -= 0.5 * n * (this.p[this.ix(x, y + 1)] - this.p[this.ix(x, y - 1)]);
      }
      this.setBoundary(1, this.u); this.setBoundary(2, this.v);
    }

    curlAt(x, y) {
      return 0.5 * (this.v[this.ix(x + 1, y)] - this.v[this.ix(x - 1, y)] - this.u[this.ix(x, y + 1)] + this.u[this.ix(x, y - 1)]) * this.n;
    }

    addVortex(cx, cy, strength, radius) {
      const n = this.n, r2 = radius * radius;
      for (let y = 1; y <= n; y++) {
        const yy = y / n - cy;
        for (let x = 1; x <= n; x++) {
          const xx = x / n - cx, d2 = xx * xx + yy * yy, w = Math.exp(-d2 / Math.max(1e-6, r2));
          const id = this.ix(x, y);
          this.u[id] += -yy * strength * w;
          this.v[id] += xx * strength * w;
          this.eta[id] += 0.017 * Math.sign(strength) * w;
        }
      }
    }

    setMouse(active, x, y, px, py) {
      this.mouse.active = active; this.mouse.x = clamp(x, 0, 1); this.mouse.y = clamp(y, 0, 1);
      this.mouse.px = clamp(px ?? x, 0, 1); this.mouse.py = clamp(py ?? y, 0, 1);
    }

    kick() {
      this.addVortex(0.33 + 0.15 * this.random(), 0.48 + 0.18 * (this.random() - 0.5), 7.0, 0.14);
      this.addVortex(0.66 - 0.15 * this.random(), 0.52 + 0.18 * (this.random() - 0.5), -7.4, 0.14);
      this.project();
    }

    applyForces() {
      const n = this.n, t = this.time;
      for (let y = 1; y <= n; y++) {
        const ny = y / n;
        for (let x = 1; x <= n; x++) {
          const nx = x / n, id = this.ix(x, y);
          const inlet = Math.exp(-Math.pow(nx / 0.18, 2));
          const wave = Math.sin(18 * ny - 5.8 * t) + 0.44 * Math.sin(31 * ny + 2.6 * t);
          this.u[id] += this.dt * this.wind * inlet * (0.82 + 0.18 * wave);
          this.etaVel[id] += this.dt * 0.15 * inlet * wave;
        }
      }
      const s = this.vortexStrength;
      const centers = [
        [0.34 + 0.08 * Math.sin(0.63 * t), 0.49 + 0.11 * Math.cos(0.82 * t), s, 0.17],
        [0.66 + 0.07 * Math.cos(0.54 * t), 0.54 + 0.10 * Math.sin(0.88 * t), -s * 1.06, 0.16],
        [0.50 + 0.16 * Math.sin(0.36 * t), 0.73 + 0.04 * Math.sin(1.25 * t), s * 0.65, 0.13]
      ];
      for (const [cx, cy, strength, radius] of centers) this.addVortex(cx, cy, strength * this.dt, radius);
      if (this.mouse.active) {
        const dx = this.mouse.x - this.mouse.px, dy = this.mouse.y - this.mouse.py;
        const base = Math.hypot(dx, dy) > 1e-5 ? 115 : 10;
        for (let y = 1; y <= n; y++) for (let x = 1; x <= n; x++) {
          const nx = x / n, ny = y / n, qx = nx - this.mouse.x, qy = ny - this.mouse.y;
          const w = Math.exp(-(qx * qx + qy * qy) / 0.014);
          const id = this.ix(x, y);
          this.u[id] += base * dx * w;
          this.v[id] += base * dy * w;
          this.etaVel[id] += 0.12 * w;
        }
      }
    }

    updateSurfaceFoam() {
      const n = this.n, dt = this.dt;
      this.eta0.set(this.eta); this.foam0.set(this.foam);
      this.advect(0, this.eta, this.eta0, this.u, this.v);
      this.advect(0, this.foam, this.foam0, this.u, this.v);
      for (let y = 1; y <= n; y++) for (let x = 1; x <= n; x++) {
        const id = this.ix(x, y);
        const lap = this.eta[this.ix(x - 1, y)] + this.eta[this.ix(x + 1, y)] + this.eta[this.ix(x, y - 1)] + this.eta[this.ix(x, y + 1)] - 4 * this.eta[id];
        const curl = this.curlAt(x, y), speed = Math.hypot(this.u[id], this.v[id]);
        this.curl[id] = curl;
        const crest = Math.max(0, this.eta[id] - 0.018);
        this.etaVel[id] = this.surfaceDamping * (this.etaVel[id] + dt * (this.waveSpeed * this.waveSpeed * n * n * lap + this.surfaceCoupling * Math.abs(curl) + 0.012 * speed));
        this.etaVel[id] = clamp(this.etaVel[id], -0.38, 0.38);
        this.eta[id] = clamp(this.eta[id] + dt * this.etaVel[id], -0.13, 0.18);
        const vortexSource = smoothstep((Math.abs(curl) - 2.1) / 7.5);
        const speedSource = smoothstep((speed - 0.48) / 1.5);
        const breaking = vortexSource * speedSource + crest * 2.2;
        this.foam[id] = clamp(this.foam[id] * this.foamDecay + this.foamBirth * breaking, 0, 1);
        if (this.spray.length < this.maxSpray && breaking > 0.22 && this.random() < Math.min(0.20, breaking * 0.05)) {
          const a = Math.atan2(this.v[id], this.u[id]) + (this.random() - 0.5) * 1.35;
          const burst = 0.18 + 0.48 * this.random() + 0.12 * Math.min(2.0, breaking);
          this.spray.push({
            x: x / n, y: y / n, z: Math.max(0.014, this.eta[id] + 0.030),
            vx: Math.cos(a) * burst * 0.25 + this.u[id] * 0.030,
            vy: Math.sin(a) * burst * 0.25 + this.v[id] * 0.030,
            vz: 0.32 + burst * 0.92,
            age: 0, life: 0.50 + this.random() * 0.70,
            size: 1.2 + this.random() * 2.7
          });
        }
      }
      this.setBoundary(0, this.eta); this.setBoundary(0, this.foam);
    }

    updateSpray() {
      const alive = [];
      for (const p of this.spray) {
        p.age += this.dt; p.vz -= 1.55 * this.dt;
        p.x += p.vx * this.dt; p.y += p.vy * this.dt; p.z += p.vz * this.dt;
        if (p.age < p.life && p.z > -0.025 && p.x > -0.1 && p.x < 1.1 && p.y > -0.1 && p.y < 1.1) alive.push(p);
      }
      this.spray = alive;
    }

    meanAbsDivergence() {
      let sum = 0, count = 0, n = this.n;
      for (let y = 1; y <= n; y++) for (let x = 1; x <= n; x++) {
        sum += Math.abs(0.5 * (this.u[this.ix(x + 1, y)] - this.u[this.ix(x - 1, y)] + this.v[this.ix(x, y + 1)] - this.v[this.ix(x, y - 1)]) / n);
        count++;
      }
      return sum / count;
    }

    step(dt) {
      if (dt) this.dt = dt;
      this.time += this.dt;
      this.applyForces();
      this.u0.set(this.u); this.v0.set(this.v);
      this.diffuse(1, this.u, this.u0, this.viscosity);
      this.diffuse(2, this.v, this.v0, this.viscosity);
      this.project();
      this.u0.set(this.u); this.v0.set(this.v);
      this.advect(1, this.u, this.u0, this.u0, this.v0);
      this.advect(2, this.v, this.v0, this.u0, this.v0);
      this.project();
      this.updateSurfaceFoam();
      this.updateSpray();
    }
  }

  class Renderer {
    constructor(canvas, sim) {
      this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.sim = sim;
      this.paused = false; this.showVectors = true; this.showSwirl = true; this.last = 0; this.fps = 0;
      this.resize(); addEventListener('resize', () => this.resize()); this.bindPointer();
    }
    resize() {
      const dpr = Math.min(2, devicePixelRatio || 1), r = this.canvas.getBoundingClientRect();
      this.canvas.width = Math.max(1, Math.floor(r.width * dpr)); this.canvas.height = Math.max(1, Math.floor(r.height * dpr));
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); this.w = r.width; this.h = r.height;
    }
    bindPointer() {
      const pos = e => { const r = this.canvas.getBoundingClientRect(); return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height }; };
      this.canvas.addEventListener('pointerdown', e => { this.canvas.setPointerCapture(e.pointerId); const p = pos(e); this.prev = p; this.sim.setMouse(true, p.x, p.y, p.x, p.y); });
      this.canvas.addEventListener('pointermove', e => { if (e.buttons) { const p = pos(e), q = this.prev || p; this.sim.setMouse(true, p.x, p.y, q.x, q.y); this.prev = p; } });
      this.canvas.addEventListener('pointerup', () => this.sim.setMouse(false, this.sim.mouse.x, this.sim.mouse.y));
      this.canvas.addEventListener('pointercancel', () => this.sim.setMouse(false, this.sim.mouse.x, this.sim.mouse.y));
    }
    project(nx, ny, z) {
      const cx = this.w * 0.52, top = this.h * 0.13, sx = this.w * 0.74, sy = this.h * 0.40;
      return { x: cx + (nx - 0.5) * sx + (ny - 0.5) * sx * 0.26, y: top + ny * sy - z * this.h * 1.08 };
    }
    waterColor(eta, foam, curl, speed) {
      const d = clamp((eta + 0.12) / 0.30, 0, 1), c = clamp(Math.abs(curl) / 12, 0, 1), s = clamp(speed / 2.5, 0, 1);
      const r = Math.round(5 + 25 * d + 45 * foam + 58 * c);
      const g = Math.round(58 + 112 * d + 85 * foam + 26 * s);
      const b = Math.round(112 + 118 * d + 82 * foam);
      return `rgb(${r},${g},${b})`;
    }
    draw() {
      const ctx = this.ctx, sim = this.sim, n = sim.n;
      ctx.clearRect(0, 0, this.w, this.h);
      const bg = ctx.createLinearGradient(0, 0, 0, this.h);
      bg.addColorStop(0, '#07111f'); bg.addColorStop(0.55, '#10283f'); bg.addColorStop(1, '#03070c');
      ctx.fillStyle = bg; ctx.fillRect(0, 0, this.w, this.h);
      const step = Math.max(2, Math.floor(n / 52));
      for (let y = n; y >= 1; y -= step) for (let x = 1; x <= n; x += step) {
        const id = sim.ix(x, y), nx = x / n, ny = y / n, z = sim.eta[id] * 1.45;
        const p = this.project(nx, ny, z), p2 = this.project(Math.min(1, (x + step) / n), Math.min(1, (y + step) / n), z);
        const curl = sim.curl[id], speed = Math.hypot(sim.u[id], sim.v[id]), radius = Math.max(2.2, Math.abs(p2.x - p.x) * 1.15);
        ctx.globalAlpha = 0.90; ctx.fillStyle = this.waterColor(sim.eta[id], sim.foam[id], curl, speed);
        ctx.beginPath(); ctx.ellipse(p.x, p.y, radius * 0.94, radius * 0.36, -0.10, 0, TAU); ctx.fill();
        if (sim.foam[id] > 0.10) {
          ctx.globalAlpha = clamp(sim.foam[id] * 0.82, 0, 0.82); ctx.fillStyle = '#f4fbff';
          ctx.beginPath(); ctx.ellipse(p.x, p.y - 1.5, radius * (0.18 + sim.foam[id] * 0.35), radius * 0.14, -0.08, 0, TAU); ctx.fill();
        }
      }
      ctx.globalAlpha = 1;
      if (this.showVectors) {
        ctx.strokeStyle = 'rgba(165,225,255,0.28)'; ctx.lineWidth = 1;
        const vstep = Math.max(8, Math.floor(n / 10));
        for (let y = 4; y <= n; y += vstep) for (let x = 4; x <= n; x += vstep) {
          const id = sim.ix(x, y), p = this.project(x / n, y / n, sim.eta[id] * 1.45 + 0.010);
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x + sim.u[id] * 18, p.y + sim.v[id] * 9); ctx.stroke();
        }
      }
      if (this.showSwirl) {
        const sstep = Math.max(3, Math.floor(n / 24));
        for (let y = 2; y <= n; y += sstep) for (let x = 2; x <= n; x += sstep) {
          const curl = sim.curlAt(x, y);
          if (Math.abs(curl) > 2.4) {
            const id = sim.ix(x, y), p = this.project(x / n, y / n, sim.eta[id] * 1.45 + 0.018);
            ctx.strokeStyle = curl > 0 ? 'rgba(255,151,66,0.42)' : 'rgba(103,185,255,0.42)';
            ctx.lineWidth = clamp(Math.abs(curl) * 0.09, 0.65, 2.0);
            ctx.beginPath(); ctx.arc(p.x, p.y, clamp(Math.abs(curl) * 0.75, 3, 11), sim.time * 4.4, sim.time * 4.4 + Math.PI * 1.58); ctx.stroke();
          }
        }
      }
      ctx.fillStyle = '#f5fcff';
      for (const sp of sim.spray) {
        const p = this.project(sp.x, sp.y, sp.z), a = clamp(1 - sp.age / sp.life, 0, 1);
        ctx.globalAlpha = a * 0.86; ctx.beginPath(); ctx.arc(p.x, p.y, sp.size * (0.55 + a), 0, TAU); ctx.fill();
      }
      ctx.globalAlpha = 1;
      document.getElementById('readout').textContent =
        `t=${sim.time.toFixed(2)}  fps=${this.fps.toFixed(0)}  spray=${sim.spray.length}  div=${sim.meanAbsDivergence().toExponential(2)}`;
    }
    frame(ts) {
      if (!this.last) this.last = ts;
      const raw = clamp((ts - this.last) / 1000, 0.001, 0.050); this.last = ts;
      this.fps = this.fps ? lerp(this.fps, 1 / raw, 0.08) : 1 / raw;
      if (!this.paused) {
        const sub = raw > 0.030 ? 2 : 1;
        for (let i = 0; i < sub; i++) this.sim.step(clamp(raw / sub, 0.012, 0.024));
      }
      this.draw(); requestAnimationFrame(t => this.frame(t));
    }
    start() { requestAnimationFrame(t => this.frame(t)); }
  }

  const canvas = document.getElementById('stage');
  let sim = new Fluid({ n: 76, dt: 0.018, seed: 20260619 });
  const renderer = new Renderer(canvas, sim);

  function bind(id, prop, digits = 2) {
    const input = document.getElementById(id), output = document.getElementById(id + 'Value');
    const apply = () => { sim[prop] = Number(input.value); output.textContent = Number(input.value).toFixed(digits); };
    input.addEventListener('input', apply); apply();
  }
  bind('wind', 'wind', 2); bind('vortex', 'vortexStrength', 2); bind('wave', 'waveSpeed', 3);
  bind('coupling', 'surfaceCoupling', 4); bind('viscosity', 'viscosity', 5); bind('foam', 'foamBirth', 3);

  document.getElementById('pause').onclick = () => {
    renderer.paused = !renderer.paused;
    document.getElementById('pause').textContent = renderer.paused ? 'Resume' : 'Pause';
  };
  document.getElementById('reset').onclick = () => {
    const old = sim;
    sim = new Fluid({ n: old.n, dt: old.dt, seed: Math.floor(Math.random() * 1e9) });
    renderer.sim = sim;
    for (const id of ['wind', 'vortex', 'wave', 'coupling', 'viscosity', 'foam']) document.getElementById(id).dispatchEvent(new Event('input'));
  };
  document.getElementById('kick').onclick = () => sim.kick();
  document.getElementById('vectors').onclick = () => {
    renderer.showVectors = !renderer.showVectors;
    document.getElementById('vectors').textContent = renderer.showVectors ? 'Flow Off' : 'Flow On';
  };
  document.getElementById('swirl').onclick = () => {
    renderer.showSwirl = !renderer.showSwirl;
    document.getElementById('swirl').textContent = renderer.showSwirl ? 'Swirl Overlay Off' : 'Swirl Overlay On';
  };
  renderer.start();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a real-time Navier-Stokes vortex/foam HTML viewer.")
    parser.add_argument("--output", type=Path, default=Path("outputs/navier_stokes_vortex_remake.html"), help="Output HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML, encoding="utf-8")
    print(f"Saved real-time vortex remake viewer: {args.output}")


if __name__ == "__main__":
    main()
