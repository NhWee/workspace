"""Write a high-visual-quality GPU ocean surface viewer.

This is the "most realistic practical" direction for the current machine:
it uses a real-time WebGL viewer with a choppy multi-component Gerstner ocean,
crest foam, spray particles, moving vortex/wake disturbances, sun/sky shading,
and interactive controls.

It is not full volumetric 3D Navier-Stokes. A true 3D air-water solver at this
visual resolution would be much heavier than an RTX 4060 Ti can run
interactively. This viewer focuses computation on the visible free surface.
"""

import argparse
from pathlib import Path


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Realistic GPU Ocean Fluid Viewer</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      overflow: hidden;
      background: #03070d;
      color: #eef8ff;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    #stage { position: fixed; inset: 0; width: 100vw; height: 100vh; display: block; }
    .panel {
      position: fixed;
      left: 16px;
      top: 16px;
      width: min(390px, calc(100vw - 32px));
      padding: 14px;
      border: 1px solid rgba(174, 221, 255, 0.18);
      border-radius: 8px;
      background: rgba(4, 12, 22, 0.68);
      backdrop-filter: blur(12px);
      box-shadow: 0 18px 70px rgba(0, 0, 0, 0.38);
    }
    h1 { margin: 0 0 10px; font-size: 18px; line-height: 1.2; letter-spacing: 0; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 12px; }
    label { display: grid; gap: 4px; color: rgba(238, 248, 255, 0.76); font-size: 12px; }
    label span { display: flex; justify-content: space-between; gap: 10px; }
    output { color: #bdeaff; font-variant-numeric: tabular-nums; }
    input[type="range"] { width: 100%; accent-color: #60c9ff; }
    .buttons { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    button {
      min-height: 32px;
      border: 1px solid rgba(151, 214, 255, 0.24);
      background: rgba(35, 79, 119, 0.64);
      color: #eef8ff;
      border-radius: 7px;
      padding: 7px 10px;
      cursor: pointer;
      font-weight: 650;
    }
    button:hover { background: rgba(52, 112, 166, 0.78); }
    .readout {
      position: fixed;
      left: 16px;
      bottom: 14px;
      color: rgba(232, 246, 255, 0.82);
      font: 12px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.70);
    }
    .note {
      position: fixed;
      right: 16px;
      bottom: 14px;
      max-width: min(470px, calc(100vw - 32px));
      text-align: right;
      color: rgba(232, 246, 255, 0.68);
      font-size: 12px;
      line-height: 1.45;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.70);
    }
    .loading {
      position: fixed;
      inset: 0;
      display: grid;
      place-items: center;
      background: #03070d;
      color: #dff4ff;
      font-size: 14px;
      z-index: 10;
    }
    @media (max-width: 760px) {
      .grid { grid-template-columns: 1fr; }
      .note { display: none; }
    }
  </style>
</head>
<body>
  <canvas id="stage"></canvas>
  <div class="loading" id="loading">Loading WebGL ocean...</div>
  <section class="panel">
    <h1>Realistic GPU Ocean Fluid Viewer</h1>
    <div class="grid">
      <label><span>Wind <output id="windValue"></output></span><input id="wind" type="range" min="0.25" max="2.5" step="0.01" value="1.25"></label>
      <label><span>Choppiness <output id="chopValue"></output></span><input id="chop" type="range" min="0.1" max="2.0" step="0.01" value="1.05"></label>
      <label><span>Swell <output id="swellValue"></output></span><input id="swell" type="range" min="0.2" max="2.2" step="0.01" value="1.0"></label>
      <label><span>Foam <output id="foamValue"></output></span><input id="foam" type="range" min="0" max="2.2" step="0.01" value="0.70"></label>
      <label><span>Spray <output id="sprayValue"></output></span><input id="spray" type="range" min="0" max="2.0" step="0.01" value="0.85"></label>
      <label><span>Sun <output id="sunValue"></output></span><input id="sun" type="range" min="0" max="1" step="0.01" value="0.72"></label>
      <label><span>Glint <output id="glintValue"></output></span><input id="glint" type="range" min="0" max="1.6" step="0.01" value="0.88"></label>
    </div>
    <div class="buttons">
      <button id="pause">Pause</button>
      <button id="storm">Storm preset</button>
      <button id="calm">Calm preset</button>
      <button id="sprayToggle">Spray Off</button>
      <button id="foamToggle">Foam Off</button>
    </div>
  </section>
  <div class="readout" id="readout"></div>
  <div class="note">A practical high-realism free-surface model: choppy Gerstner spectrum, crest foam, spray, moving vortex wakes, and WebGL shading. It is optimized for visible ocean motion, not full volumetric CFD.</div>

  <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
  <script>
  const canvas = document.getElementById('stage');
  const loading = document.getElementById('loading');
  const readout = document.getElementById('readout');
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const smoothstep = (edge0, edge1, x) => {
    const t = clamp((x - edge0) / Math.max(1e-6, edge1 - edge0), 0, 1);
    return t * t * (3 - 2 * t);
  };

  if (!window.THREE) {
    loading.textContent = 'Three.js failed to load. Check internet access, then reload this HTML.';
    throw new Error('THREE unavailable');
  }

  const state = {
    wind: 1.25,
    chop: 1.05,
    swell: 1.0,
    foam: 0.70,
    spray: 0.85,
    sun: 0.72,
    glint: 0.88,
    paused: false,
    showSpray: true,
    showFoam: true,
    fps: 0,
    sprayCount: 0
  };

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x07131f, 0.012);

  const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 800);
  camera.position.set(0, 32, 66);
  camera.lookAt(0, 0, 0);

  const hemi = new THREE.HemisphereLight(0x9fd8ff, 0x06111f, 1.35);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff2cc, 2.8);
  sun.position.set(-40, 70, 28);
  scene.add(sun);

  const skyGeo = new THREE.SphereGeometry(360, 32, 16);
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: { sunAmount: { value: state.sun } },
    vertexShader: `
      varying vec3 vWorld;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorld = normalize(wp.xyz);
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: `
      varying vec3 vWorld;
      uniform float sunAmount;
      void main() {
        float h = clamp(vWorld.y * 0.5 + 0.5, 0.0, 1.0);
        vec3 low = vec3(0.015, 0.045, 0.075);
        vec3 high = vec3(0.33, 0.58, 0.78);
        vec3 col = mix(low, high, pow(h, 1.25));
        float glow = pow(max(0.0, dot(normalize(vWorld), normalize(vec3(-0.42, 0.78, 0.25)))), 36.0);
        col += vec3(1.0, 0.78, 0.42) * glow * sunAmount;
        gl_FragColor = vec4(col, 1.0);
      }
    `
  });
  scene.add(new THREE.Mesh(skyGeo, skyMat));

  const grid = 190;
  const size = 118;
  const geo = new THREE.PlaneGeometry(size, size, grid - 1, grid - 1);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  const baseX = new Float32Array(pos.count);
  const baseZ = new Float32Array(pos.count);
  const foamAttr = new Float32Array(pos.count);
  const colors = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    baseX[i] = pos.getX(i);
    baseZ[i] = pos.getZ(i);
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const waterMat = new THREE.MeshPhysicalMaterial({
    vertexColors: true,
    roughness: 0.44,
    metalness: 0.0,
    transmission: 0.0,
    clearcoat: 0.42,
    clearcoatRoughness: 0.30,
    side: THREE.DoubleSide
  });
  const water = new THREE.Mesh(geo, waterMat);
  scene.add(water);

  const deep = new THREE.Color(0x05243a);
  const mid = new THREE.Color(0x08708f);
  const crest = new THREE.Color(0x6faec0);
  const foamColor = new THREE.Color(0xc8e2e6);
  const reflectionColor = new THREE.Color(0x5aa8c8);
  const sunGlintColor = new THREE.Color(0xfff3c9);
  const sunDir = new THREE.Vector3(-0.46, 0.82, 0.30).normalize();

  const waves = [
    // A directional spectrum: dominant swell plus weaker crossing components.
    // Varied wavelengths and phases avoid a repeating row-of-hills appearance.
    { a: 1.42, l: 38.0, d: 0.10, s: 0.94, q: 0.54, p: 0.12 },
    { a: 1.00, l: 27.0, d: 0.21, s: 1.03, q: 0.50, p: 1.42 },
    { a: 0.72, l: 19.0, d: -0.09, s: 1.11, q: 0.46, p: 2.36 },
    { a: 0.52, l: 15.0, d: 0.46, s: 1.16, q: 0.40, p: 0.76 },
    { a: 0.40, l: 11.5, d: -0.38, s: 1.23, q: 0.35, p: 2.84 },
    { a: 0.31, l: 8.4, d: 0.78, s: 1.30, q: 0.28, p: 1.67 },
    { a: 0.25, l: 6.4, d: -0.72, s: 1.38, q: 0.23, p: 3.02 },
    { a: 0.20, l: 5.1, d: 1.16, s: 1.46, q: 0.19, p: 0.39 },
    { a: 0.16, l: 4.0, d: -1.19, s: 1.55, q: 0.15, p: 2.12 },
    { a: 0.12, l: 3.2, d: 1.62, s: 1.67, q: 0.12, p: 1.05 }
  ].map(w => {
    const dir = new THREE.Vector2(Math.cos(w.d), Math.sin(w.d)).normalize();
    return { ...w, dir, k: Math.PI * 2 / w.l };
  });

  const wakeCenters = [
    { x: -18, z: -7, spin: 1.0, r: 17, speed: 0.38, phase: 0.2 },
    { x: 18, z: 10, spin: -1.0, r: 15, speed: 0.31, phase: 2.1 }
  ];

  const sprayGeo = new THREE.BufferGeometry();
  const maxSpray = 1400;
  const sprayPositions = new Float32Array(maxSpray * 3);
  const sprayColors = new Float32Array(maxSpray * 3);
  const spraySizes = new Float32Array(maxSpray);
  sprayGeo.setAttribute('position', new THREE.BufferAttribute(sprayPositions, 3));
  sprayGeo.setAttribute('color', new THREE.BufferAttribute(sprayColors, 3));
  sprayGeo.setAttribute('size', new THREE.BufferAttribute(spraySizes, 1));
  const sprayCanvas = document.createElement('canvas');
  sprayCanvas.width = sprayCanvas.height = 64;
  const sprayContext = sprayCanvas.getContext('2d');
  const sprayGradient = sprayContext.createRadialGradient(32, 32, 2, 32, 32, 30);
  sprayGradient.addColorStop(0.0, 'rgba(236, 250, 255, 0.95)');
  sprayGradient.addColorStop(0.35, 'rgba(210, 238, 248, 0.48)');
  sprayGradient.addColorStop(1.0, 'rgba(180, 220, 240, 0.0)');
  sprayContext.fillStyle = sprayGradient;
  sprayContext.fillRect(0, 0, 64, 64);
  const sprayTexture = new THREE.CanvasTexture(sprayCanvas);
  const sprayMat = new THREE.PointsMaterial({
    size: 0.34,
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
    vertexColors: true,
    opacity: 0.78,
    map: sprayTexture,
    alphaMap: sprayTexture,
    sizeAttenuation: true
  });
  const sprayPoints = new THREE.Points(sprayGeo, sprayMat);
  scene.add(sprayPoints);
  const spray = [];

  function foamBreakup(x, z, t) {
    // A cheap, drifting multi-frequency pattern. It breaks the foam edge into
    // patches without adding a per-frame texture allocation or random flicker.
    const flow = t * (0.72 + state.wind * 0.34);
    const broad = Math.sin(x * 0.46 + z * 0.19 - flow);
    const cross = Math.sin(-x * 0.27 + z * 0.63 + flow * 0.71 + 1.8);
    const fine = Math.sin(x * 1.18 + z * 0.81 - flow * 1.43 + 0.5);
    return clamp(0.50 + broad * 0.24 + cross * 0.17 + fine * 0.09, 0, 1);
  }

  function sampleOcean(x, z, t) {
    let y = 0, dx = 0, dz = 0, foam = 0, breaking = 0, slopeX = 0, slopeZ = 0;
    for (const w of waves) {
      const windScale = state.wind;
      const amp = w.a * state.swell * (0.72 + 0.28 * windScale);
      const angularFrequency = Math.sqrt(9.81 * w.k) * 0.68 * w.s * (0.72 + 0.28 * windScale);
      const phase = w.k * (w.dir.x * x + w.dir.y * z) - t * angularFrequency + w.p;
      const sn = Math.sin(phase);
      const cs = Math.cos(phase);
      y += amp * sn;
      const q = w.q * state.chop;
      dx += q * amp * w.dir.x * cs;
      dz += q * amp * w.dir.y * cs;
      slopeX += amp * w.k * w.dir.x * cs;
      slopeZ += amp * w.k * w.dir.y * cs;
      const crestSample = smoothstep(0.70, 0.98, sn);
      foam += crestSample * w.q * amp * w.k * state.foam;
      breaking += crestSample * w.q * amp * w.k * 4.4;
    }
    for (const c of wakeCenters) {
      const cx = c.x + Math.sin(t * c.speed + c.phase) * 11.0;
      const cz = c.z + Math.cos(t * c.speed * 0.8 + c.phase) * 9.0;
      const rx = x - cx, rz = z - cz;
      const r2 = rx * rx + rz * rz;
      const g = Math.exp(-r2 / (c.r * c.r));
      const angle = Math.atan2(rz, rx) + c.spin * t * 1.4;
      y += Math.sin(angle * 2.0 + t * 1.7) * g * 0.45 * state.chop;
      foam += g * 0.55 * state.foam;
      dx += -rz / c.r * g * c.spin * 0.9;
      dz += rx / c.r * g * c.spin * 0.9;
    }
    const slope = Math.sqrt(slopeX * slopeX + slopeZ * slopeZ);
    foam += smoothstep(0.42, 0.96, slope) * 0.55 * state.foam;
    const crestBreak = clamp(breaking * state.foam, 0, 1);
    foam = clamp(foam, 0, 1);
    const breakup = foamBreakup(x + dx, z + dz, t);
    const foamCore = smoothstep(0.72, 0.96, foam);
    // Retain dense breaking crests, while eroding lower-density foam into
    // short-lived, advecting patches instead of a uniform white blanket.
    foam = clamp(foam * (0.26 + 0.94 * breakup) + foamCore * 0.24, 0, 1);
    return { x: x + dx, y, z: z + dz, foam, breaking: crestBreak, slope, slopeX, slopeZ };
  }

  let frameIndex = 0;
  function updateOcean(t, dt) {
    let foamSum = 0;
    for (let i = 0; i < pos.count; i++) {
      const s = sampleOcean(baseX[i], baseZ[i], t);
      pos.setXYZ(i, s.x, s.y, s.z);
      foamAttr[i] = lerp(foamAttr[i], s.foam, 0.08);
      foamSum += foamAttr[i];
      const h = clamp((s.y + 3.0) / 7.5, 0, 1);
      const invNormal = 1.0 / Math.sqrt(s.slopeX * s.slopeX + s.slopeZ * s.slopeZ + 1.0);
      const nx = -s.slopeX * invNormal;
      const ny = invNormal;
      const nz = -s.slopeZ * invNormal;
      const vx = camera.position.x - s.x;
      const vy = camera.position.y - s.y;
      const vz = camera.position.z - s.z;
      const invView = 1.0 / Math.sqrt(vx * vx + vy * vy + vz * vz + 1.0e-6);
      const vdx = vx * invView;
      const vdy = vy * invView;
      const vdz = vz * invView;
      const ndotv = clamp(nx * vdx + ny * vdy + nz * vdz, 0, 1);
      const fresnel = Math.pow(1.0 - ndotv, 3.2);
      const hx = sunDir.x + vdx;
      const hy = sunDir.y + vdy;
      const hz = sunDir.z + vdz;
      const invHalf = 1.0 / Math.sqrt(hx * hx + hy * hy + hz * hz + 1.0e-6);
      const ndoth = clamp(nx * hx * invHalf + ny * hy * invHalf + nz * hz * invHalf, 0, 1);
      const sparkle = Math.pow(ndoth, 84.0) * smoothstep(0.35, 1.0, s.slope) * state.sun * state.glint;
      const reflection = fresnel * state.sun * state.glint * 0.42;
      const c = deep.clone()
        .lerp(mid, h)
        .lerp(crest, smoothstep(0.72, 1.0, h) * 0.25)
        .lerp(reflectionColor, clamp(reflection, 0, 0.42))
        .lerp(sunGlintColor, clamp(sparkle, 0, 0.68))
        .lerp(foamColor, state.showFoam ? foamAttr[i] * 0.48 : 0);
      colors[i * 3 + 0] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
    pos.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    geo.computeVertexNormals();

    if (state.showSpray && state.spray > 0 && frameIndex % 2 === 0) {
      for (let k = 0; k < 18 * state.spray; k++) {
        const x = (Math.random() - 0.5) * size * 0.82;
        const z = (Math.random() - 0.5) * size * 0.82;
        const s = sampleOcean(x, z, t);
        if (s.breaking > 0.30 && spray.length < maxSpray && Math.random() < s.breaking * 0.30 * state.spray) {
          spray.push({
            x: s.x, y: s.y + 0.35, z: s.z,
            vx: 0.22 * state.wind + (Math.random() - 0.5) * 0.56,
            vy: 0.95 + Math.random() * 2.15 * state.spray + s.breaking * 1.05,
            vz: 0.04 * state.wind + (Math.random() - 0.5) * 0.46,
            age: 0,
            life: 0.34 + Math.random() * 0.52,
            size: 0.10 + Math.random() * 0.20
          });
        }
      }
    }

    let alive = 0;
    for (const p of spray) {
      p.age += dt;
      p.vy -= 4.8 * dt;
      p.x += p.vx * dt * 8.0;
      p.y += p.vy * dt;
      p.z += p.vz * dt * 8.0;
      if (p.age < p.life && p.y > -2.0) {
        const a = 1 - p.age / p.life;
        sprayPositions[alive * 3 + 0] = p.x;
        sprayPositions[alive * 3 + 1] = p.y;
        sprayPositions[alive * 3 + 2] = p.z;
        sprayColors[alive * 3 + 0] = 0.80 + 0.20 * a;
        sprayColors[alive * 3 + 1] = 0.92 + 0.08 * a;
        sprayColors[alive * 3 + 2] = 1.00;
        spraySizes[alive] = p.size;
        spray[alive] = p;
        alive++;
      }
    }
    spray.length = alive;
    sprayGeo.setDrawRange(0, alive);
    sprayGeo.attributes.position.needsUpdate = true;
    sprayGeo.attributes.color.needsUpdate = true;
    sprayGeo.attributes.size.needsUpdate = true;
    state.sprayCount = alive;
    frameIndex++;
    return foamSum / pos.count;
  }

  const moonGeo = new THREE.CircleGeometry(8, 48);
  const moonMat = new THREE.MeshBasicMaterial({ color: 0xffdca3, transparent: true, opacity: 0.85, depthWrite: false });
  const sunDisk = new THREE.Mesh(moonGeo, moonMat);
  sunDisk.position.set(-46, 54, -95);
  sunDisk.lookAt(camera.position);
  scene.add(sunDisk);

  function resize() {
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();

  function bind(id, prop, digits = 2) {
    const input = document.getElementById(id);
    const output = document.getElementById(id + 'Value');
    const apply = () => {
      state[prop] = Number(input.value);
      output.textContent = Number(input.value).toFixed(digits);
      if (prop === 'sun') {
        sun.intensity = 1.5 + 2.5 * state.sun;
        skyMat.uniforms.sunAmount.value = state.sun;
        moonMat.opacity = 0.35 + 0.55 * state.sun;
      }
    };
    input.addEventListener('input', apply);
    apply();
  }
  bind('wind', 'wind', 2);
  bind('chop', 'chop', 2);
  bind('swell', 'swell', 2);
  bind('foam', 'foam', 2);
  bind('spray', 'spray', 2);
  bind('sun', 'sun', 2);
  bind('glint', 'glint', 2);

  function setControl(id, value) {
    const input = document.getElementById(id);
    input.value = value;
    input.dispatchEvent(new Event('input'));
  }
  document.getElementById('pause').onclick = () => {
    state.paused = !state.paused;
    document.getElementById('pause').textContent = state.paused ? 'Resume' : 'Pause';
  };
  document.getElementById('storm').onclick = () => {
    // Keep the summed Gerstner steepness below the self-intersection range.
    setControl('wind', 1.85); setControl('chop', 1.16); setControl('swell', 1.30); setControl('foam', 1.05); setControl('spray', 1.30); setControl('sun', 0.62); setControl('glint', 1.08);
  };
  document.getElementById('calm').onclick = () => {
    setControl('wind', 0.65); setControl('chop', 0.62); setControl('swell', 0.58); setControl('foam', 0.25); setControl('spray', 0.20); setControl('sun', 0.90); setControl('glint', 0.74);
  };
  document.getElementById('sprayToggle').onclick = () => {
    state.showSpray = !state.showSpray;
    document.getElementById('sprayToggle').textContent = state.showSpray ? 'Spray Off' : 'Spray On';
    if (!state.showSpray) spray.length = 0;
  };
  document.getElementById('foamToggle').onclick = () => {
    state.showFoam = !state.showFoam;
    document.getElementById('foamToggle').textContent = state.showFoam ? 'Foam Off' : 'Foam On';
  };

  let orbit = 0;
  let last = performance.now();
  let foamAverage = 0;
  function animate(now) {
    const rawDt = clamp((now - last) / 1000, 0.001, 0.050);
    last = now;
    state.fps = state.fps ? lerp(state.fps, 1 / rawDt, 0.06) : 1 / rawDt;
    if (!state.paused) {
      const t = now * 0.001;
      foamAverage = updateOcean(t, rawDt);
      orbit += rawDt * 0.035;
      camera.position.x = Math.sin(orbit) * 8.0;
      camera.position.z = 66 + Math.cos(orbit) * 5.0;
      camera.lookAt(0, 0.3, 0);
      sunDisk.lookAt(camera.position);
    }
    waterMat.roughness = 0.54 - 0.16 * state.sun;
    renderer.render(scene, camera);
    readout.textContent = `fps=${state.fps.toFixed(0)}  vertices=${pos.count}  foam=${foamAverage.toFixed(3)}  spray=${state.sprayCount}`;
    requestAnimationFrame(animate);
  }

  loading.style.display = 'none';
  requestAnimationFrame(animate);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a realistic WebGL ocean free-surface viewer.")
    parser.add_argument("--output", type=Path, default=Path("outputs/realistic_gpu_ocean.html"), help="Output HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML, encoding="utf-8")
    print(f"Saved realistic GPU ocean viewer: {args.output}")


if __name__ == "__main__":
    main()
