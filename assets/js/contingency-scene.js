/*
 * A staged, single-circuit-contingency illustration for AELab.
 * The four-bus diamond represents two initially symmetric transfer corridors.
 * Light density is explanatory artwork, not a computed loading or rating.
 * The shared controller owns canvas sizing, motion preferences and scheduling.
 */
(() => {
  'use strict';
  const K = window.ProjectSceneKit;
  if (!K) return;
  const { colors, clamp, lerp, smooth } = K;
  const TAU = Math.PI * 2;
  const CYCLE = 12;
  const labels = Object.freeze({ base: 'Base case', open: 'Circuit opened', redistributed: 'Redistributed flow', restoring: 'Restoring circuit' });
  const MANUAL = new Set(['base', 'open', 'redistributed']);
  const models = new Map();
  const cleanTime = time => Number.isFinite(time) ? Math.max(0, time) : 0;
  const p = (x, y, z) => [x, y, z];

  /** Return the accessible stage identifier, without reading or changing state. */
  function getStage({ time = 0, state = 'auto' } = {}) {
    if (MANUAL.has(state)) return state;
    const phase = cleanTime(time) % CYCLE;
    if (phase < 3) return 'base';
    if (phase < 4.5) return 'open';
    if (phase < 9) return 'redistributed';
    return 'restoring';
  }

  function stageState(time, state) {
    const stage = getStage({ time, state });
    if (MANUAL.has(state)) return { stage, opening: state === 'base' ? 0 : 1, gain: state === 'redistributed' ? 1 : 0 };
    const phase = time % CYCLE;
    if (stage === 'base') return { stage, opening: 0, gain: 0 };
    if (stage === 'open') return { stage, opening: smooth((phase - 3) / 0.75), gain: 0 };
    if (stage === 'redistributed') return { stage, opening: 1, gain: smooth((phase - 4.5) / 0.7) };
    return { stage, opening: 1 - smooth((phase - 9.35) / 1.2), gain: 1 - smooth((phase - 10.55) / 1.2) };
  }

  class Model {
    constructor() { this.items = [];this.upper = [];this.lower = [];this.breakers = [];this.buses = []; }
    line(points, style = {}, tag = '') {
      this.items.push({ kind: 'line', points, style, tag, depth: points.reduce((sum, v) => sum + v[2] + v[1] * 0.25, 0) / points.length });
    }
    face(points, style = {}) {
      this.items.push({ kind: 'face', points, style, depth: points.reduce((sum, v) => sum + v[2] + v[1] * 0.25, 0) / points.length });
    }
    box(options) { this.items.push({ kind: 'box', options, depth: options.z + (options.y + options.h * 0.5) * 0.25 }); }
    finish() { this.items.sort((a, b) => a.depth - b.depth);return this; }
  }

  function ring(model, x, y, z, radius, { plane = 'xz', ...style } = {}) {
    const points = [];
    for (let i = 0; i <= 22; i += 1) {
      const angle = i / 22 * TAU;
      points.push(plane === 'xy' ? p(x + Math.cos(angle) * radius, y + Math.sin(angle) * radius, z) : p(x + Math.cos(angle) * radius, y, z + Math.sin(angle) * radius));
    }
    model.line(points, style);
  }

  function insulator(model, x, y, z, height = 0.46, radius = 0.075) {
    model.line([p(x, y, z), p(x, y + height, z)], { color: colors.gold, alpha: 0.68, width: 1.05 });
    for (let i = 1; i <= 6; i += 1) ring(model, x, y + i / 7 * height, z, radius * (1 - i * 0.035), { color: colors.ivory, alpha: 0.7, width: 0.62 });
    return p(x, y + height, z);
  }

  function bus(model, x, z, name) {
    model.box({ x, y: 0, z, w: 1.14, h: 0.09, d: 1.02, color: colors.muted, alpha: 0.65 });
    const terminals = [];
    for (let phase = 0; phase < 3; phase += 1) {
      const offset = (phase - 1) * 0.28;
      insulator(model, x - 0.32, 0.1, z + offset, 0.72, 0.062);
      insulator(model, x + 0.32, 0.1, z + offset, 0.72, 0.062);
      model.line([p(x - 0.48, 0.85, z + offset), p(x + 0.48, 0.85, z + offset)], { color: colors.ivory, alpha: 0.88, width: 1.0 });
      terminals.push(p(x, 0.85, z + offset));
    }
    model.buses.push({ point: p(x, 0.08, z), name });
    return terminals;
  }

  function tower(model, x, z, crossX, crossZ) {
    const height = 2.66;
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        model.box({ x: x + sx * 0.21, y: 0, z: z + sz * 0.25, w: 0.16, h: 0.06, d: 0.17, alpha: 0.45 });
        model.line([p(x + sx * 0.21, 0.06, z + sz * 0.25), p(x + sx * 0.095, 1.99, z + sz * 0.095), p(x, height, z)], { color: colors.ivory, alpha: sz > 0 ? 0.8 : 0.4, width: sz > 0 ? 0.82 : 0.57 });
      }
    }
    for (let level = 0; level < 6; level += 1) {
      const y0 = 0.1 + level * 0.31, y1 = y0 + 0.31;
      const rx0 = lerp(0.21, 0.095, level / 6), rx1 = lerp(0.21, 0.095, (level + 1) / 6);
      const rz0 = lerp(0.25, 0.095, level / 6), rz1 = lerp(0.25, 0.095, (level + 1) / 6);
      for (const side of [-1, 1]) {
        model.line([p(x - rx0, y0, z + side * rz0), p(x + rx1, y1, z + side * rz1), p(x - rx1, y1, z + side * rz1), p(x + rx0, y0, z + side * rz0)], { color: colors.ivory, alpha: side > 0 ? 0.55 : 0.28, width: 0.53 });
        model.line([p(x + side * rx0, y0, z - rz0), p(x + side * rx1, y1, z + rz1)], { color: colors.ice, alpha: 0.44, width: 0.52 });
      }
    }
    model.line([p(x - crossX * 0.72, 2.15, z - crossZ * 0.72), p(x + crossX * 0.72, 2.15, z + crossZ * 0.72)], { color: colors.ivory, alpha: 0.84, width: 0.88 });
    model.line([p(x - crossX * 0.72, 2.15, z - crossZ * 0.72), p(x, 2.48, z), p(x + crossX * 0.72, 2.15, z + crossZ * 0.72)], { color: colors.ivory, alpha: 0.59, width: 0.65 });
    const terminals = [];
    for (let phase = 0; phase < 3; phase += 1) {
      const offset = phase === 1 ? 0 : phase === 0 ? -0.64 : 0.64;
      const y = phase === 1 ? 2.29 : 1.79;
      insulator(model, x + crossX * offset, y, z + crossZ * offset, 0.32, 0.061);
      terminals.push(p(x + crossX * offset, y, z + crossZ * offset));
    }
    return terminals;
  }

  function source(model, x, z) {
    model.box({ x, y: 0.02, z, w: 1.35, h: 0.18, d: 1.08, color: colors.ice, alpha: 0.55 });
    model.box({ x, y: 0.22, z: z - 0.28, w: 1.22, h: 0.92, d: 0.64, color: colors.ice, alpha: 0.64 });
    for (let i = 0; i < 9; i += 1) {
      const angle = i / 9 * TAU;
      model.line([p(x + Math.cos(angle) * 0.4, 0.67 + Math.sin(angle) * 0.4, z - 0.07), p(x + Math.cos(angle) * 0.4, 0.67 + Math.sin(angle) * 0.4, z + 0.57)], { color: colors.ice, alpha: 0.52, width: 0.6 });
    }
    ring(model, x, 0.67, z + 0.58, 0.4, { plane: 'xy', color: colors.ice, alpha: 0.87, width: 0.9 });
    ring(model, x, 0.67, z + 0.59, 0.29, { plane: 'xy', color: colors.ivory, alpha: 0.6, width: 0.65 });
    ring(model, x, 0.67, z + 0.60, 0.1, { plane: 'xy', color: colors.gold, alpha: 0.84, width: 0.8 });
    for (let i = 0; i < 10; i += 1) {
      const angle = i / 10 * TAU;
      model.line([p(x + Math.cos(angle) * 0.11, 0.67 + Math.sin(angle) * 0.11, z + 0.60), p(x + Math.cos(angle + 0.16) * 0.285, 0.67 + Math.sin(angle + 0.16) * 0.285, z + 0.60)], { color: colors.ivory, alpha: 0.64, width: 0.6 });
    }
  }

  function load(model, x, z) {
    model.box({ x, y: 0.03, z, w: 1.25, h: 0.13, d: 1.0, color: colors.muted, alpha: 0.55 });
    model.box({ x, y: 0.16, z, w: 1.14, h: 1.13, d: 0.82, color: colors.ice, alpha: 0.7 });
    model.box({ x, y: 1.29, z, w: 1.22, h: 0.05, d: 0.9, color: colors.ivory, alpha: 0.62 });
    for (let bay = 0; bay < 3; bay += 1) {
      const bx = x - 0.42 + bay * 0.4;
      model.line([p(bx, 0.24, z + 0.42), p(bx, 1.19, z + 0.42)], { color: colors.ivory, alpha: 0.4, width: 0.62 });
      for (let row = 0; row < 7; row += 1) model.line([p(bx + 0.05, 0.33 + row * 0.115, z + 0.43), p(bx + 0.25, 0.33 + row * 0.115, z + 0.43)], { color: colors.ice, alpha: 0.64, width: 0.7 });
    }
  }

  function sag(from, to, amount = 0.2) {
    const points = [];
    for (let i = 0; i <= 24; i += 1) {
      const t = i / 24;
      points.push(p(lerp(from[0], to[0], t), lerp(from[1], to[1], t) - Math.sin(t * Math.PI) * amount, lerp(from[2], to[2], t)));
    }
    return points;
  }

  function route(model, terminals, tag) {
    const all = [];
    for (let i = 0; i < terminals.length - 1; i += 1) {
      const section = sag(terminals[i], terminals[i + 1], i === 0 ? 0.15 : 0.2);
      model.line(section, { color: colors.gold, alpha: 0.66, width: 0.68 }, tag);
      all.push(...(i === 0 ? section : section.slice(1)));
    }
    return all;
  }

  function measuredRoute(points) {
    const lengths = [0];
    for (let i = 1; i < points.length; i += 1) {
      const a = points[i - 1], b = points[i];
      lengths.push(lengths[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
    }
    return { points, lengths, length: lengths[lengths.length - 1] };
  }

  function build(mobile) {
    const model = new Model();
    const depth = mobile ? 5.0 : 3.55;
    const a = bus(model, -4.93, 0, 'A');
    const b = bus(model, 0, -depth, 'B');
    const c = bus(model, 0, depth, 'C');
    const d = bus(model, 4.93, 0, 'D');
    source(model, -6.75, 0);load(model, 6.68, 0);
    const ul = tower(model, -3.2, -depth * 0.57, 0.57, 0.82);
    const ur = tower(model, 3.05, -depth * 0.57, -0.57, 0.82);
    const ll = tower(model, -3.2, depth * 0.57, -0.57, 0.82);
    const lr = tower(model, 3.05, depth * 0.57, 0.57, 0.82);
    model.review = p(0, 0.06, depth);
    model.box({ x: -1.23, y: 0.025, z: -depth + 0.3, w: 1.22, h: 0.11, d: 1.04, color: colors.gold, alpha: 0.55 });
    for (let phase = 0; phase < 3; phase += 1) {
      const z = -depth + 0.3 + (phase - 1) * 0.29;
      const left = insulator(model, -1.64, 0.14, z, 0.8, 0.062);
      const right = insulator(model, -0.82, 0.14, z, 0.8, 0.062);
      model.breakers.push({ left, right });
      const upperLeft = route(model, [a[phase], ul[phase], left], 'upper');
      const upperRight = route(model, [right, b[phase], ur[phase], d[phase]], 'upper');
      model.upper.push(measuredRoute([...upperLeft, right, ...upperRight.slice(1)]));
      model.lower.push(measuredRoute(route(model, [a[phase], ll[phase], c[phase], lr[phase], d[phase]], 'lower')));
      model.line(sag(p(-6.06, 0.67, (phase - 1) * 0.2), a[phase], 0.08), { color: colors.ice, alpha: 0.66, width: 0.72 });
      model.line(sag(d[phase], p(6.06, 0.55, (phase - 1) * 0.2), 0.08), { color: colors.ice, alpha: 0.66, width: 0.72 });
    }
    return model.finish();
  }

  function at(route, progress) {
    const distance = clamp(progress, 0, 1) * route.length;
    let index = 1;
    while (index < route.lengths.length - 1 && route.lengths[index] < distance) index += 1;
    const a = route.points[index - 1], b = route.points[index];
    const fraction = (distance - route.lengths[index - 1]) / Math.max(1e-6, route.lengths[index] - route.lengths[index - 1]);
    return p(lerp(a[0], b[0], fraction), lerp(a[1], b[1], fraction), lerp(a[2], b[2], fraction));
  }

  function packets(ctx, camera, routes, time, gain = 0, opacity = 1) {
    routes.forEach((route, phase) => {
      for (let packet = 0; packet < 4; packet += 1) {
        const intensity = (packet < 2 ? 1 : gain) * opacity;
        if (intensity < 0.01) continue;
        const offset = [0, 0.5, 0.25, 0.75][packet];
        const progress = (time * 0.12 + offset + phase * 0.035) % 1;
        const position = at(route, progress);
        K.line(ctx, [at(route, Math.max(0, progress - 0.015)), position], camera, { color: colors.white, alpha: intensity * 0.65, width: 1.15 });
        K.glow(ctx, position, camera, { color: gain > 0.4 ? colors.gold : colors.ivory, alpha: intensity * 0.92, radius: camera.width <= 640 ? 1.35 : 1.85 });
      }
    });
  }

  function breaker(ctx, camera, contacts, opening) {
    contacts.forEach(({ left, right }) => {
      const length = right[0] - left[0];
      const angle = opening * 1.12;
      const end = p(left[0] + Math.cos(angle) * length, left[1] + Math.sin(angle) * length, left[2]);
      K.line(ctx, [left, end], camera, { color: opening > 0.1 ? colors.gold : colors.ivory, alpha: 0.92, width: 1.65 });
      K.line(ctx, [p(left[0], left[1] + 0.04, left[2]), p(end[0], end[1] + 0.04, end[2])], camera, { color: colors.ivory, alpha: 0.43, width: 0.55 });
      K.glow(ctx, left, camera, { color: colors.gold, alpha: 0.55, radius: 1.2 });
      K.glow(ctx, right, camera, { color: opening > 0.1 ? colors.muted : colors.ivory, alpha: 0.45, radius: 1.05 });
    });
  }

  function reviewHalo(ctx, camera, centre, gain, time) {
    if (gain < 0.01) return;
    for (let ringIndex = 0; ringIndex < 3; ringIndex += 1) {
      const radius = 0.72 + ringIndex * 0.27 + Math.sin(time * 0.65) * 0.035;
      const points = [];
      for (let i = 0; i <= 52; i += 1) {
        const angle = i / 52 * TAU;
        points.push(p(centre[0] + Math.cos(angle) * radius, centre[1], centre[2] + Math.sin(angle) * radius));
      }
      K.line(ctx, points, camera, { color: colors.gold, alpha: gain * (0.37 - ringIndex * 0.08), width: ringIndex === 0 ? 0.95 : 0.55 });
    }
  }

  /** Paint one frame; manual state fixes switching geometry independently of time. */
  function draw(ctx, { width, height, time = 0, pointerX = 0, pointerY = 0, state = 'auto' }) {
    if (!ctx || !Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;
    const t = cleanTime(time), mobile = width <= 640;
    const status = stageState(t, state);
    const camera = K.view(width, height, {
      extent: mobile ? 16.7 : 17.3, heightExtent: mobile ? 9.3 : 7.5,
      centerY: mobile ? 0.54 : 0.60, depthX: mobile ? 0.08 : 0.24,
      depthY: mobile ? 0.82 : 0.44,
      pointerX: Number.isFinite(pointerX) ? pointerX : 0,
      pointerY: Number.isFinite(pointerY) ? pointerY : 0,
    });
    if (!models.has(mobile)) models.set(mobile, build(mobile));
    const model = models.get(mobile);
    ctx.save();
    try {
      ctx.clearRect(0, 0, width, height);
      K.backdrop(ctx, width, height);
      K.floor(ctx, camera, { extent: 7.5, depth: mobile ? 5.5 : 4.5, spacing: 0.65, alpha: 0.09 });
      reviewHalo(ctx, camera, model.review, status.gain, MANUAL.has(state) ? 0 : t);
      for (const item of model.items) {
        if (item.kind === 'box') { K.box(ctx, camera, item.options);continue; }
        if (item.kind === 'face') { K.face(ctx, item.points, camera, item.style);continue; }
        let style = item.style;
        if (item.tag === 'upper') style = { ...style, alpha: style.alpha * (1 - status.opening * 0.65), color: status.opening > 0.5 ? colors.muted : colors.gold };
        if (item.tag === 'lower' && status.gain > 0.01) {
          K.line(ctx, item.points, camera, { color: colors.gold, alpha: status.gain * 0.038, width: 4.7 });
          style = { ...style, alpha: style.alpha + status.gain * 0.17 };
        }
        K.line(ctx, item.points, camera, style);
      }
      breaker(ctx, camera, model.breakers, status.opening);
      // No packet crosses the upper corridor while any contact is separated.
      if (status.opening < 0.001 && status.stage !== 'open') packets(ctx, camera, model.upper, t, 0, status.stage === 'restoring' ? 1 - status.gain : 1);
      packets(ctx, camera, model.lower, t, status.gain);
      K.glow(ctx, p(-6.75, 0.67, 0.61), camera, { color: colors.ice, alpha: 0.66, radius: mobile ? 1.65 : 2.6 });
    } finally { ctx.restore(); }
  }

  window.ContingencyScene = Object.freeze({ types: Object.freeze(['contingency']), draw, getStage, labels });
})();
