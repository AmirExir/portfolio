/*
 * Cinematic, conceptual engineering artwork. These fields do not represent
 * measured signals, network topology, model results, or operating conditions.
 * The host owns canvas sizing, DPR, animation scheduling and motion preferences.
 */
(() => {
  'use strict';

  const TAU = Math.PI * 2;
  const MODE_NAMES = ['grid', 'knowledge', 'learning'];
  const WHITE = '#fff3d8';
  const GOLD = '#d7a663';
  const ICE = '#84c9c4';
  const modeStates = new WeakMap();
  const transitionLayers = new WeakMap();
  let powerJourneyFailed = false;
  const powerJourneyAvailable = () => !powerJourneyFailed && typeof window.PowerJourney?.draw === 'function';
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const smooth = value => value * value * (3 - 2 * value);

  function randomSequence(seed) {
    let state = seed;
    return () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 4294967296;
    };
  }

  const random = randomSequence(28137);
  const dust = new Float32Array(680 * 5);
  for (let i = 0; i < dust.length; i += 1) dust[i] = random();

  // The sampled meshes and their screen buffers are reused between frames.
  const meshCache = new Map();
  function meshBuffer(rows, columns) {
    const key = `${rows}:${columns}`;
    if (!meshCache.has(key)) meshCache.set(key, new Float32Array(rows * columns * 4));
    return meshCache.get(key);
  }

  function viewFor(width, height, time, pointerX, pointerY, type) {
    const hero = type === 'field';
    const mobile = width <= 640;
    const yaw = (hero ? -0.12 : -0.19) + pointerX * 0.17 + Math.sin(time * 0.08) * 0.04;
    const pitch = (hero ? 0.12 : 0.14) + pointerY * 0.1;
    return {
      width, height, pointerX, pointerY,
      cx: width * (hero || mobile ? 0.5 : type === 'evidence' ? 0.76 : 0.28) + pointerX * width * 0.006,
      cy: height * (hero ? 0.725 : mobile ? 0.735 : 0.51) + pointerY * height * 0.006,
      scale: hero
        ? Math.max(width * 0.172, Math.min(height * 0.325, width * 0.33))
        : mobile ? Math.min(width * 0.143, height * 0.18) : Math.min(width * 0.081, height * 0.3),
      verticalScale: hero ? (mobile ? 0.74 : 0.6) : 1,
      sinYaw: Math.sin(yaw), cosYaw: Math.cos(yaw),
      sinPitch: Math.sin(pitch), cosPitch: Math.cos(pitch),
    };
  }

  function project(x, y, z, view, buffer, index) {
    const rx = x * view.cosYaw + z * view.sinYaw;
    const rz = z * view.cosYaw - x * view.sinYaw;
    const ry = y * view.cosPitch - rz * view.sinPitch;
    const depth = rz * view.cosPitch + y * view.sinPitch;
    const perspective = 6.3 / (6.3 + depth);
    buffer[index] = view.cx + rx * view.scale * perspective;
    buffer[index + 1] = view.cy - ry * view.scale * perspective * view.verticalScale;
    buffer[index + 2] = depth;
    buffer[index + 3] = perspective;
  }

  function haze(ctx, x, y, radius, color, opacity, squash = 1) {
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(1, squash);
    const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
    gradient.addColorStop(0, `rgba(${color},${opacity})`);
    gradient.addColorStop(0.28, `rgba(${color},${opacity * 0.38})`);
    gradient.addColorStop(1, `rgba(${color},0)`);
    ctx.fillStyle = gradient;
    ctx.fillRect(-radius, -radius, radius * 2, radius * 2);
    ctx.restore();
  }

  function drawDust(ctx, width, height, time, hero = false) {
    const density = width < 600 ? 280 : 680;
    ctx.fillStyle = WHITE;
    for (let i = 0; i < density; i += 1) {
      const index = i * 5;
      const x = (dust[index] * width + Math.sin(time * 0.06 + dust[index + 2] * TAU) * 5);
      const y = dust[index + 1] * height;
      const clearText = hero ? clamp((y / height - 0.28) * 2.4, 0.05, 1) : 1;
      const twinkle = 0.55 + Math.sin(time * 0.23 + dust[index + 3] * TAU) * 0.3;
      ctx.globalAlpha = (0.035 + dust[index + 2] * 0.2) * clearText * twinkle;
      const size = dust[index + 4] > 0.975 ? 1.4 : 0.65;
      ctx.fillRect(x, y, size, size);
    }
    ctx.globalAlpha = 1;
  }

  function modeWeights(ctx, mode, time) {
    const target = Math.max(0, MODE_NAMES.indexOf(mode));
    let state = modeStates.get(ctx);
    if (!state) {
      state = { target, start: time, from: [0, 0, 0], weights: [0, 0, 0] };
      state.weights[target] = 1;
      state.from[target] = 1;
      modeStates.set(ctx, state);
    }
    if (state.target !== target) {
      state.from = state.weights.slice();
      state.target = target;
      state.start = time;
    }
    // time=0 is a useful deterministic, immediate mode for reduced motion.
    const progress = time === 0 ? 1 : smooth(clamp((time - state.start) / 1.4, 0, 1));
    for (let i = 0; i < 3; i += 1) state.weights[i] = state.from[i] * (1 - progress) + (i === target ? progress : 0);
    return state.weights;
  }

  function fieldGeometry(buffer, rows, columns, view, time, weights, secondary) {
    const knowledge = weights[1];
    const learning = weights[2];
    const phase = secondary ? 1.9 : 0;
    for (let row = 0; row < rows; row += 1) {
      const v = row / (rows - 1) * 2 - 1;
      for (let col = 0; col < columns; col += 1) {
        const u = col / (columns - 1) * 9.8 - 4.9;
        const distance = Math.sqrt(u * u + 0.065) / 4.9;
        const flare = 0.2 + Math.pow(distance, 0.82) * (1.22 + knowledge * 0.18);
        const twist = u * (0.77 + knowledge * 0.17 + learning * 0.04) + time * 0.058 + phase + knowledge * Math.sin(u * 0.82) * 0.6;
        const fold = Math.sin(u * 1.12 - time * 0.13) * 0.3 + Math.sin(u * 2.0 + v * 1.6 - time * 0.11) * 0.11;
        const ripple = Math.sin(u * 3.2 + v * 4.2 - time * 0.24) * learning * 0.065;
        const x = u + Math.sin(v * Math.PI) * knowledge * 0.1;
        const y = fold + v * flare * Math.cos(twist) + ripple + (secondary ? -0.16 : 0);
        const z = Math.cos(u * 0.79 + time * 0.07) * 0.61 + v * flare * Math.sin(twist) + (secondary ? 0.48 : 0);
        project(x, y, z, view, buffer, (row * columns + col) * 4);
      }
    }
  }

  function drawMesh(ctx, buffer, rows, columns, view, time, secondary, weights) {
    const gradient = ctx.createLinearGradient(0, 0, view.width, view.height * 0.16);
    gradient.addColorStop(0, secondary ? '#529eac' : '#80aaa8');
    gradient.addColorStop(0.32, secondary ? '#92c6c9' : '#c6b68f');
    gradient.addColorStop(0.54, '#fff1d5');
    gradient.addColorStop(0.78, weights[2] > 0.5 ? '#93c7bd' : '#d8ac6a');
    gradient.addColorStop(1, '#a6703b');
    ctx.strokeStyle = gradient;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    const alpha = secondary ? 0.35 : 0.68;
    // A few soft, wide strokes give the fine fabric an optical bloom.
    for (let row = 3; row < rows - 3; row += 9) {
      const start = row * columns * 4;
      ctx.beginPath();
      ctx.moveTo(buffer[start], buffer[start + 1]);
      for (let col = 1; col < columns; col += 1) {
        const index = start + col * 4;
        ctx.lineTo(buffer[index], buffer[index + 1]);
      }
      ctx.globalAlpha = secondary ? 0.025 : 0.044;
      ctx.lineWidth = 7.2;
      ctx.stroke();
    }
    for (let row = 0; row < rows; row += 1) {
      const start = row * columns * 4;
      const edge = Math.sin(row / (rows - 1) * Math.PI);
      ctx.globalAlpha = alpha * (0.34 + edge * 0.66);
      ctx.lineWidth = row % 9 === 0 ? 0.65 : 0.36;
      ctx.beginPath();
      ctx.moveTo(buffer[start], buffer[start + 1]);
      for (let col = 1; col < columns; col += 1) {
        const index = start + col * 4;
        ctx.lineTo(buffer[index], buffer[index + 1]);
      }
      ctx.stroke();
    }
    // Delicate cross-weave makes the object read as a folded surface in space.
    ctx.lineWidth = 0.35;
    for (let col = 0; col < columns; col += 4) {
      ctx.globalAlpha = (secondary ? 0.09 : 0.15) * Math.sin(col / (columns - 1) * Math.PI);
      ctx.beginPath();
      for (let row = 0; row < rows; row += 1) {
        const index = (row * columns + col) * 4;
        if (row === 0) ctx.moveTo(buffer[index], buffer[index + 1]);
        else ctx.lineTo(buffer[index], buffer[index + 1]);
      }
      ctx.stroke();
    }
    ctx.fillStyle = WHITE;
    // Dots sit on the surface rather than in an unrelated particle overlay.
    for (let row = 0; row < rows; row += 2) {
      for (let col = row % 3; col < columns; col += 3) {
        const index = (row * columns + col) * 4;
        const shimmer = Math.sin(col * 0.44 + row * 0.73 - time * 0.9);
        const falloff = Math.sin(col / (columns - 1) * Math.PI);
        const depth = clamp((1.7 - buffer[index + 2]) * 0.35, 0.12, 1);
        ctx.globalAlpha = (secondary ? 0.34 : 0.74) * falloff * depth * (0.63 + shimmer * 0.37);
        const size = (row % 12 === 0 ? 1.4 : 0.85) * buffer[index + 3];
        ctx.fillRect(buffer[index] - size * 0.5, buffer[index + 1] - size * 0.5, size, size);
      }
    }
    // Sparse bright impulses travel along complete filaments.
    ctx.fillStyle = secondary ? ICE : WHITE;
    for (let row = 4; row < rows; row += 11) {
      const progress = ((time * (secondary ? 0.036 : 0.046) + row * 0.063) % 1 + 1) % 1;
      const offset = progress * (columns - 2);
      const col = Math.floor(offset);
      const part = offset - col;
      const index = (row * columns + col) * 4;
      const x = buffer[index] + (buffer[index + 4] - buffer[index]) * part;
      const y = buffer[index + 1] + (buffer[index + 5] - buffer[index + 1]) * part;
      const fade = Math.sin(progress * Math.PI);
      ctx.globalAlpha = 0.11 * fade;
      ctx.beginPath(); ctx.arc(x, y, 5.2, 0, TAU); ctx.fill();
      ctx.globalAlpha = 0.92 * fade;
      ctx.fillRect(x - 0.85, y - 0.85, 1.7, 1.7);
    }
    ctx.globalAlpha = 1;
  }

  function drawGridField(ctx, view, time) {
    const { width, height } = view;
    const weights = [1, 0, 0];
    drawDust(ctx, width, height, time, true);
    haze(ctx, width * 0.52, height * 0.71, Math.min(width * 0.32, height * 0.42), '205,135,63', 0.12, 0.4);
    haze(ctx, width * 0.23, height * 0.72, width * 0.24, '48,124,134', 0.045, 0.4);
    const columns = width < 600 ? 128 : 176;
    const primaryRows = width < 600 ? 68 : 94;
    const secondaryRows = width < 600 ? 34 : 46;
    const secondary = meshBuffer(secondaryRows, columns);
    const primary = meshBuffer(primaryRows, columns);
    ctx.globalCompositeOperation = 'lighter';
    fieldGeometry(secondary, secondaryRows, columns, view, time, weights, true);
    drawMesh(ctx, secondary, secondaryRows, columns, view, time, true, weights);
    fieldGeometry(primary, primaryRows, columns, view, time, weights, false);
    drawMesh(ctx, primary, primaryRows, columns, view, time, false, weights);
    ctx.globalCompositeOperation = 'source-over';
  }

  const sheetBuffer = new Float32Array(38 * 32 * 4);
  const projectedPoint = new Float32Array(4);
  const projectedNext = new Float32Array(4);

  function evidenceSheet(ctx, view, time, sheet) {
    const rows = 32;
    const columns = 38;
    const orbit = sheet * 0.43 + Math.sin(time * 0.11) * 0.07;
    const centreX = -3.1 + sheet * 0.54;
    const centreY = Math.sin(orbit * 1.3) * 0.4;
    const centreZ = Math.cos(orbit) * 0.88;
    const tilt = -0.5 + sheet * 0.19;
    const contraction = 1 - sheet * 0.045;
    for (let row = 0; row < rows; row += 1) {
      const v = (row / (rows - 1) * 2 - 1) * contraction;
      for (let col = 0; col < columns; col += 1) {
        const u = (col / (columns - 1) * 2 - 1) * 0.57;
        const curl = Math.sin(u * 2.2 + orbit + time * 0.13) * 0.25;
        const x = centreX + u * Math.cos(tilt) - v * Math.sin(tilt) * 0.48;
        const y = centreY + v * 0.83 + curl;
        const z = centreZ + u * Math.sin(tilt) + v * v * 0.38 + curl;
        project(x, y, z, view, sheetBuffer, (row * columns + col) * 4);
      }
    }
    ctx.strokeStyle = sheet % 3 === 0 ? ICE : WHITE;
    ctx.lineWidth = 0.48;
    for (let row = 0; row < rows; row += 1) {
      const perimeter = row === 0 || row === rows - 1;
      ctx.globalAlpha = perimeter ? 0.75 : 0.17 + (sheet / 7) * 0.06;
      ctx.lineWidth = perimeter ? 0.7 : 0.36;
      const offset = row * columns * 4;
      ctx.beginPath();
      ctx.moveTo(sheetBuffer[offset], sheetBuffer[offset + 1]);
      for (let col = 1; col < columns; col += 1) {
        const index = offset + col * 4;
        ctx.lineTo(sheetBuffer[index], sheetBuffer[index + 1]);
      }
      ctx.stroke();
    }
    for (let col = 0; col < columns; col += 1) {
      const perimeter = col === 0 || col === columns - 1;
      ctx.globalAlpha = perimeter ? 0.76 : 0.13;
      ctx.lineWidth = perimeter ? 0.7 : 0.32;
      ctx.beginPath();
      for (let row = 0; row < rows; row += 1) {
        const index = (row * columns + col) * 4;
        if (row === 0) ctx.moveTo(sheetBuffer[index], sheetBuffer[index + 1]);
        else ctx.lineTo(sheetBuffer[index], sheetBuffer[index + 1]);
      }
      ctx.stroke();
    }
    // Short illuminated bands suggest structured source passages, without text.
    ctx.strokeStyle = GOLD;
    for (let row = 5; row < rows - 4; row += 5) {
      ctx.globalAlpha = 0.38 + Math.sin(time * 0.55 - sheet * 0.8 + row) * 0.12;
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      for (let col = 5; col < columns - 5 - row % 8; col += 1) {
        const index = (row * columns + col) * 4;
        if (col === 5) ctx.moveTo(sheetBuffer[index], sheetBuffer[index + 1]);
        else ctx.lineTo(sheetBuffer[index], sheetBuffer[index + 1]);
      }
      ctx.stroke();
    }
  }

  function aperturePoint(angle, radius, depth, time) {
    // A rounded square aperture is visually ordered, unlike a generic orb.
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const x = Math.sign(c) * Math.pow(Math.abs(c), 0.46) * radius;
    const y = Math.sign(s) * Math.pow(Math.abs(s), 0.46) * radius;
    const rotate = -0.09 + Math.sin(time * 0.09) * 0.06;
    return [1.68 + x * Math.cos(rotate) - y * Math.sin(rotate), 0.12 + x * Math.sin(rotate) + y * Math.cos(rotate), depth];
  }

  function drawEvidence(ctx, view, time) {
    drawDust(ctx, view.width, view.height, time);
    haze(ctx, view.cx + view.scale * 1.68, view.cy, view.scale * 1.95, '207,148,69', 0.11, 0.92);
    ctx.globalCompositeOperation = 'lighter';
    for (let sheet = 0; sheet < 7; sheet += 1) evidenceSheet(ctx, view, time, sheet);
    ctx.strokeStyle = GOLD;
    // Source traces converge on the opening, preserving their separate strands.
    for (let trace = 0; trace < 44; trace += 1) {
      const v = trace / 43 * 2 - 1;
      ctx.beginPath();
      for (let step = 0; step <= 64; step += 1) {
        const t = step / 64;
        const x = -2.65 + t * 4.02;
        const y = v * (0.82 * (1 - t) + 0.12) + Math.sin(t * Math.PI) * (0.37 + Math.sin(v * 3 + time * 0.1) * 0.24);
        const z = Math.sin(v * 2.8 + t * 3.0) * (1 - t) * 0.86;
        project(x, y, z, view, projectedPoint, 0);
        if (step === 0) ctx.moveTo(projectedPoint[0], projectedPoint[1]);
        else ctx.lineTo(projectedPoint[0], projectedPoint[1]);
      }
      ctx.globalAlpha = trace % 7 === 0 ? 0.42 : 0.14;
      ctx.lineWidth = trace % 7 === 0 ? 0.6 : 0.34;
      ctx.stroke();
    }
    for (let ring = 0; ring < 10; ring += 1) {
      const radius = 0.46 + ring * 0.058;
      const depth = (ring - 4) * 0.095;
      ctx.strokeStyle = ring % 4 === 0 ? ICE : WHITE;
      ctx.globalAlpha = 0.25 + (1 - ring / 10) * 0.5;
      ctx.lineWidth = ring === 0 ? 1.45 : 0.55;
      ctx.beginPath();
      for (let step = 0; step <= 128; step += 1) {
        const p = aperturePoint(step / 128 * TAU, radius, depth, time);
        project(p[0], p[1], p[2], view, projectedPoint, 0);
        if (step === 0) ctx.moveTo(projectedPoint[0], projectedPoint[1]);
        else ctx.lineTo(projectedPoint[0], projectedPoint[1]);
      }
      ctx.stroke();
    }
    // The aperture holds a dense ordered point lattice with a moving light front.
    ctx.fillStyle = WHITE;
    for (let row = 0; row < 19; row += 1) {
      for (let col = 0; col < 19; col += 1) {
        const x = 1.68 + (col / 18 - 0.5) * 0.81;
        const y = 0.12 + (row / 18 - 0.5) * 0.81;
        project(x, y, -0.22, view, projectedPoint, 0);
        const intensity = Math.pow(Math.max(0, Math.sin(row * 0.19 + col * 0.1 - time * 0.8)), 5);
        ctx.globalAlpha = 0.27 + intensity * 0.7;
        const size = 1.05 + intensity * 0.75;
        ctx.fillRect(projectedPoint[0], projectedPoint[1], size, size);
      }
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
  }

  // Fixed-seed, spatially local graph. This is artwork, not an actual ML model.
  const NODE_COUNT = 860;
  const graphNodes = new Float32Array(NODE_COUNT * 4);
  const graphScreen = new Float32Array(NODE_COUNT * 4);
  for (let i = 0; i < NODE_COUNT; i += 1) {
    const angle = random() * TAU;
    const polar = Math.acos(random() * 2 - 1);
    const radius = 0.44 + Math.pow(random(), 0.35) * 0.96;
    const lobe = i % 3;
    graphNodes[i * 4] = Math.cos(angle) * Math.sin(polar) * radius * 1.2 + (lobe - 1) * 1.35;
    graphNodes[i * 4 + 1] = Math.sin(angle) * Math.sin(polar) * radius * (lobe === 1 ? 0.75 : 0.83) + Math.sin(lobe * 2.8) * 0.21;
    graphNodes[i * 4 + 2] = Math.cos(polar) * radius * 0.9;
    graphNodes[i * 4 + 3] = random();
  }
  const graphEdges = [];
  for (let i = 0; i < NODE_COUNT; i += 1) {
    const closest = [[Infinity, -1], [Infinity, -1], [Infinity, -1]];
    for (let j = 0; j < NODE_COUNT; j += 1) {
      if (i === j) continue;
      const dx = graphNodes[i * 4] - graphNodes[j * 4];
      const dy = graphNodes[i * 4 + 1] - graphNodes[j * 4 + 1];
      const dz = graphNodes[i * 4 + 2] - graphNodes[j * 4 + 2];
      const distance = dx * dx + dy * dy + dz * dz;
      if (distance >= closest[2][0]) continue;
      closest[2] = [distance, j];
      closest.sort((a, b) => a[0] - b[0]);
    }
    closest.forEach(([distance, j]) => {
      if (j > i && distance < 0.8) graphEdges.push(i, j);
    });
  }

  const LEARNING_DUST_COUNT = 3200;
  const learningDust = new Float32Array(LEARNING_DUST_COUNT * 4);
  for (let i = 0; i < LEARNING_DUST_COUNT; i += 1) {
    const x = random() * 6.6 - 3.3;
    const angle = random() * TAU;
    const radius = (0.52 + Math.cos(x * 1.65) * 0.15) * Math.pow(random(), 0.4);
    learningDust[i * 4] = x;
    learningDust[i * 4 + 1] = Math.sin(angle + x) * radius;
    learningDust[i * 4 + 2] = Math.cos(angle + x) * radius;
    learningDust[i * 4 + 3] = random();
  }

  function drawLearning(ctx, view, time) {
    drawDust(ctx, view.width, view.height, time);
    haze(ctx, view.cx - view.scale * 0.45, view.cy, view.scale * 2.05, '108,177,178', 0.09, 0.77);
    haze(ctx, view.cx + view.scale * 0.9, view.cy, view.scale * 1.65, '198,128,60', 0.09, 0.7);
    for (let i = 0; i < NODE_COUNT; i += 1) {
      const index = i * 4;
      const phase = graphNodes[index + 3] * TAU;
      const x = graphNodes[index] + Math.sin(time * 0.17 + graphNodes[index + 1] * 1.7) * 0.07;
      const y = graphNodes[index + 1] + Math.sin(time * 0.22 + graphNodes[index] * 0.9) * 0.065;
      const z = graphNodes[index + 2] + Math.sin(time * 0.14 + phase) * 0.04;
      project(x, y, z, view, graphScreen, index);
    }
    ctx.globalCompositeOperation = 'lighter';
    // Fine suspended material gives the sparse graph a luminous spatial body.
    for (let i = 0; i < LEARNING_DUST_COUNT; i += 1) {
      const index = i * 4;
      const x = learningDust[index];
      const angle = time * 0.07 + x * 0.23;
      const y = learningDust[index + 1] * Math.cos(angle) - learningDust[index + 2] * Math.sin(angle) + Math.sin(x * 0.92 + time * 0.1) * 0.2;
      const z = learningDust[index + 1] * Math.sin(angle) + learningDust[index + 2] * Math.cos(angle);
      project(x, y, z, view, projectedNext, 0);
      ctx.fillStyle = x > 0.4 ? '#efd2a0' : '#b3d7d3';
      ctx.globalAlpha = (0.04 + learningDust[index + 3] * 0.28) * Math.pow(Math.max(0, 1 - Math.abs(x) / 3.4), 0.45);
      const size = learningDust[index + 3] > 0.94 ? 1.15 : 0.7;
      ctx.fillRect(projectedNext[0], projectedNext[1], size, size);
    }
    for (let edge = 0; edge < graphEdges.length; edge += 2) {
      const a = graphEdges[edge] * 4;
      const b = graphEdges[edge + 1] * 4;
      const depth = (graphScreen[a + 2] + graphScreen[b + 2]) * 0.5;
      const pulse = Math.max(0, Math.sin(time * 0.63 - graphNodes[a] * 1.0));
      ctx.strokeStyle = graphNodes[a] > 0.5 ? GOLD : ICE;
      ctx.globalAlpha = clamp(0.18 - depth * 0.055 + pulse * 0.16, 0.04, 0.43);
      ctx.lineWidth = edge % 9 === 0 ? 0.8 : 0.5;
      ctx.beginPath();
      ctx.moveTo(graphScreen[a], graphScreen[a + 1]);
      ctx.lineTo(graphScreen[b], graphScreen[b + 1]);
      ctx.stroke();
      if (edge % 12 === 0) {
        const progress = ((time * 0.32 + edge * 0.032) % 1 + 1) % 1;
        const x = graphScreen[a] + (graphScreen[b] - graphScreen[a]) * progress;
        const y = graphScreen[a + 1] + (graphScreen[b + 1] - graphScreen[a + 1]) * progress;
        ctx.fillStyle = WHITE;
        ctx.globalAlpha = 0.58 * Math.sin(progress * Math.PI);
        ctx.fillRect(x - 0.65, y - 0.65, 1.3, 1.3);
      }
    }
    for (let i = 0; i < NODE_COUNT; i += 1) {
      const index = i * 4;
      const depth = graphScreen[index + 2];
      const phase = graphNodes[index + 3] * TAU;
      const pulse = Math.pow(Math.max(0, Math.sin(time * 0.63 - graphNodes[index] + phase * 0.15)), 6);
      const near = clamp(1 - (depth + 1) * 0.29, 0.15, 1);
      const size = (0.7 + graphNodes[index + 3] * 0.85 + pulse * 0.3) * graphScreen[index + 3];
      ctx.fillStyle = graphNodes[index] > 0.5 ? '#edc489' : '#bae0da';
      if (i % 7 === 0) {
        ctx.globalAlpha = 0.04 + pulse * 0.06;
        ctx.beginPath(); ctx.arc(graphScreen[index], graphScreen[index + 1], size * 3.5, 0, TAU); ctx.fill();
      }
      ctx.globalAlpha = (0.57 + pulse * 0.7) * near;
      ctx.fillRect(graphScreen[index] - size * 0.5, graphScreen[index + 1] - size * 0.5, size, size);
    }
    // Long-range filaments weave through the locally connected cloud.
    ctx.strokeStyle = WHITE;
    for (let line = 0; line < 46; line += 1) {
      const phase = line / 46 * TAU;
      ctx.beginPath();
      for (let step = 0; step <= 104; step += 1) {
        const t = step / 104;
        const x = t * 6.6 - 3.3;
        const taper = Math.pow(Math.sin(t * Math.PI), 0.6);
        const y = Math.sin(t * TAU * 0.8 + phase + time * 0.085) * 0.66 * taper;
        const z = Math.cos(t * TAU + phase + time * 0.05) * 0.92 * taper;
        project(x, y, z, view, projectedNext, 0);
        if (step === 0) ctx.moveTo(projectedNext[0], projectedNext[1]);
        else ctx.lineTo(projectedNext[0], projectedNext[1]);
      }
      ctx.globalAlpha = line % 7 === 0 ? 0.23 : 0.07;
      ctx.lineWidth = 0.4;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
  }

  function drawHeroMode(ctx, view, time, mode) {
    if (mode === 0) {
      if (powerJourneyAvailable()) {
        ctx.save();
        try {
          window.PowerJourney.draw(ctx, {
            width: view.width, height: view.height, time,
            pointerX: view.pointerX, pointerY: view.pointerY,
            eventTime: view.eventTime, reducedMotion: view.reducedMotion,
            suppressFlash: view.suppressFlash,
          });
          return;
        } catch (error) {
          // This optional artwork must not disable the working AI illustrations.
          powerJourneyFailed = true;
          console.warn('Power journey unavailable; using the abstract grid illustration.', error);
        } finally {
          ctx.restore();
        }
        ctx.clearRect(0, 0, view.width, view.height);
      }
      drawGridField(ctx, view, time);
      return;
    }
    const mobile = view.width <= 640;
    const composition = {
      ...view,
      cx: view.width * (mode === 1 ? 0.52 : 0.5),
      cy: view.height * 0.72,
      scale: mode === 1
        ? Math.min(view.width * 0.15, view.height * 0.19)
        : Math.min(view.width * 0.18, view.height * 0.22),
      verticalScale: mobile ? 1 : mode === 1 ? 0.7 : 0.55,
    };
    if (mode === 1) drawEvidence(ctx, composition, time);
    else drawLearning(ctx, composition, time);
  }

  function transitionLayer(ctx, width, height) {
    if (typeof OffscreenCanvas !== 'function') return null;
    const pixelWidth = ctx.canvas.width;
    const pixelHeight = ctx.canvas.height;
    let layer = transitionLayers.get(ctx);
    if (!layer) {
      const canvas = new OffscreenCanvas(pixelWidth, pixelHeight);
      const context = canvas.getContext('2d');
      if (!context) return null;
      layer = { canvas, context };
      transitionLayers.set(ctx, layer);
    }
    if (layer.canvas.width !== pixelWidth || layer.canvas.height !== pixelHeight) {
      layer.canvas.width = pixelWidth;
      layer.canvas.height = pixelHeight;
    }
    layer.context.setTransform(pixelWidth / width, 0, 0, pixelHeight / height, 0, 0);
    return layer;
  }

  function drawField(ctx, view, time, weights) {
    // No lightning while entering or leaving the AI artwork, even when the
    // independent event clock was frozen partway through a strike.
    view.suppressFlash = view.suppressFlash || weights[0] < 0.998;
    const active = [];
    for (let i = 0; i < weights.length; i += 1) if (weights[i] > 0.002) active.push(i);
    const layer = active.length > 1 ? transitionLayer(ctx, view.width, view.height) : null;
    if (layer) {
      // Only a mode transition uses a reusable intermediate bitmap. Steady-state
      // rendering remains direct; older browsers switch compositions immediately.
      for (const mode of active) {
        layer.context.clearRect(0, 0, view.width, view.height);
        drawHeroMode(layer.context, view, time, mode);
        ctx.globalAlpha = weights[mode];
        ctx.drawImage(layer.canvas, 0, 0, view.width, view.height);
      }
      ctx.globalAlpha = 1;
    } else {
      let strongest = 0;
      for (let i = 1; i < weights.length; i += 1) if (weights[i] > weights[strongest]) strongest = i;
      drawHeroMode(ctx, view, time, strongest);
    }
    // Keep title/copy clear and let the sculpture disappear before the mode
    // controls, rather than placing bright strands behind their hit targets.
    ctx.save();
    ctx.globalCompositeOperation = 'destination-in';
    const mask = ctx.createLinearGradient(0, 0, 0, view.height);
    const powerWeight = powerJourneyAvailable() ? weights[0] : 0;
    mask.addColorStop(0, 'rgba(0,0,0,0)');
    mask.addColorStop(0.52 - powerWeight * 0.10, 'rgba(0,0,0,0)');
    mask.addColorStop(0.605 - powerWeight * 0.095, 'rgba(0,0,0,1)');
    mask.addColorStop(0.755 + powerWeight * 0.035, 'rgba(0,0,0,1)');
    mask.addColorStop(0.838 + powerWeight * 0.012, 'rgba(0,0,0,0)');
    mask.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = mask;
    ctx.fillRect(0, 0, view.width, view.height);
    ctx.restore();
  }

  /** Draw a transparent frame in CSS pixels; time is expressed in seconds. */
  function draw(ctx, { type = 'field', width, height, time = 0, pointerX = 0, pointerY = 0, mode = 'grid', eventTime = time, reducedMotion = false, suppressFlash = false }) {
    if (!ctx || !Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;
    const t = Number.isFinite(time) ? time : 0;
    const px = Number.isFinite(pointerX) ? clamp(pointerX, -1, 1) : 0;
    const py = Number.isFinite(pointerY) ? clamp(pointerY, -1, 1) : 0;
    const view = viewFor(width, height, t, px, py, type);
    view.eventTime = eventTime;
    view.reducedMotion = reducedMotion;
    view.suppressFlash = suppressFlash || mode !== 'grid';
    ctx.save();
    ctx.clearRect(0, 0, width, height);
    if (type === 'evidence') drawEvidence(ctx, view, t);
    else if (type === 'learning') drawLearning(ctx, view, t);
    else drawField(ctx, view, t, modeWeights(ctx, mode, t));
    ctx.restore();
  }

  window.CinematicFields = Object.freeze({
    types: Object.freeze(['field', 'evidence', 'learning']), draw,
    get powerJourneyAvailable() { return powerJourneyAvailable(); },
  });
})();
