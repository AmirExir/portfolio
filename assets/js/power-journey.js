/*
 * A conceptual power-system journey, drawn as cinematic industrial linework.
 * Moving lights illustrate energy transfer, not electron trajectories, measured
 * power, a study-case topology, protection behavior, or an AC simulation.
 * The caller owns DPR, canvas dimensions, scheduling and motion preferences.
 */
(() => {
  'use strict';

  const TAU = Math.PI * 2;
  const COLORS = { ivory: '#e5e1d0', gold: '#d9b47b', ice: '#8abfc5', muted: '#657f85', white: '#fff2d5' };
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const lerp = (a, b, t) => a + (b - a) * t;
  const point = (x, y, z) => [x, y, z];
  const palettes = {
    steel: ['rgba(162,182,178,.21)', 'rgba(57,75,83,.76)', 'rgba(87,103,108,.55)'],
    warm: ['rgba(215,185,135,.27)', 'rgba(93,84,69,.73)', 'rgba(121,110,90,.5)'],
    dark: ['rgba(100,129,134,.13)', 'rgba(18,30,38,.87)', 'rgba(32,48,55,.74)'],
  };

  class Geometry {
    constructor() {
      this.items = [];
      this.routes = [];
      this.rotors = [];
      this.lights = [];
      this.flux = [];
    }

    add(kind, points, color, alpha = 1, width = 0.65, order = 0) {
      this.items.push({ kind, points, color, alpha, width,
        depth: points.reduce((sum, p) => sum + p[2] + p[1] * 0.28, 0) / points.length + order,
        screen: new Float32Array(points.length * 2) });
    }

    line(points, color = COLORS.ivory, alpha = 0.5, width = 0.65, order = 0) {
      this.add('line', points, color, alpha, width, order);
    }

    face(points, fill, alpha = 1, order = 0) { this.add('face', points, fill, alpha, 0, order); }

    box(x, y, z, width, height, depth, palette = 'steel', opacity = 1) {
      const [top, front, side] = palettes[palette];
      const x0 = x - width / 2, x1 = x + width / 2;
      const z0 = z - depth / 2, z1 = z + depth / 2, y1 = y + height;
      const a = point(x0, y, z1), b = point(x1, y, z1), c = point(x1, y1, z1), d = point(x0, y1, z1);
      this.face([a, b, c, d], front, opacity);
      this.face([point(x1, y, z0), b, c, point(x1, y1, z0)], side, opacity);
      this.face([point(x0, y1, z0), point(x1, y1, z0), c, d], top, opacity);
      this.line([a, b, c, d, a], COLORS.ivory, 0.39 * opacity, 0.65, 0.002);
      this.line([d, point(x0, y1, z0), point(x1, y1, z0), c], COLORS.ivory, 0.56 * opacity, 0.65, 0.003);
      this.line([point(x1, y1, z0), point(x1, y, z0), b], COLORS.ice, 0.28 * opacity, 0.6, 0.002);
    }

    route(points, phase, start, duration, color = COLORS.gold, width = 0.65) {
      this.line(points, color, 0.48, width, 0.02);
      this.routes.push({ points, phase, start, duration, color });
    }

    finish() { this.items.sort((a, b) => a.depth - b.depth); return this; }
  }

  class Group {
    constructor(geometry, x, z, scale = 1) { this.g = geometry; this.x = x; this.z = z; this.s = scale; }
    p(x, y, z) { return point(this.x + x * this.s, y * this.s, this.z + z * this.s); }
    line(points, ...style) { this.g.line(points.map(p => this.p(...p)), ...style); }
    face(points, ...style) { this.g.face(points.map(p => this.p(...p)), ...style); }
    box(x, y, z, w, h, d, palette = 'steel', opacity = 1) { this.g.box(this.x + x * this.s, y * this.s, this.z + z * this.s, w * this.s, h * this.s, d * this.s, palette, opacity); }
    circle(x, y, z, radius, plane = 'xz', color = COLORS.ivory, alpha = 0.5, width = 0.65, segments = 28) {
      const points = [];
      for (let i = 0; i <= segments; i += 1) {
        const angle = i / segments * TAU;
        points.push(plane === 'xy' ? point(x + Math.cos(angle) * radius, y + Math.sin(angle) * radius, z)
          : point(x + Math.cos(angle) * radius, y, z + Math.sin(angle) * radius));
      }
      this.line(points, color, alpha, width);
    }
    insulator(x, y, z, height = 0.52, radius = 0.095) {
      this.line([point(x, y, z), point(x, y + height, z)], COLORS.gold, 0.7, 1.25);
      for (let i = 1; i <= 7; i += 1) this.circle(x, y + height * i / 8, z, radius * (1.08 - i * 0.045), 'xz', COLORS.ivory, 0.72, 0.7, 16);
      this.circle(x, y + height, z, radius * 0.42, 'xz', COLORS.gold, 0.9, 0.8, 16);
      return this.p(x, y + height, z);
    }
    cylinder(x, y, z, radius, length, color = COLORS.ivory) {
      const segments = 28;
      for (let i = 0; i < segments; i += 1) {
        const a = i / segments * TAU, b = (i + 1) / segments * TAU;
        const shade = 0.07 + Math.max(0, Math.sin(a)) * 0.15;
        this.face([point(x + Math.cos(a) * radius, y + Math.sin(a) * radius, z - length / 2), point(x + Math.cos(b) * radius, y + Math.sin(b) * radius, z - length / 2), point(x + Math.cos(b) * radius, y + Math.sin(b) * radius, z + length / 2), point(x + Math.cos(a) * radius, y + Math.sin(a) * radius, z + length / 2)], `rgba(163,190,191,${shade})`);
        if (i % 2 === 0) this.line([point(x + Math.cos(a) * radius, y + Math.sin(a) * radius, z - length / 2), point(x + Math.cos(a) * radius, y + Math.sin(a) * radius, z + length / 2)], color, 0.4, 0.5);
      }
      this.circle(x, y, z - length / 2, radius, 'xy', color, 0.54, 0.7);
      this.circle(x, y, z + length / 2, radius, 'xy', color, 0.85, 1.05);
    }
  }

  function generator(g) {
    g.box(0, 0, 0, 2.52, 0.11, 1.95, 'dark');
    g.box(-0.08, 0.13, -0.48, 2.12, 1.32, 0.92, 'steel');
    g.box(0, 0.14, 0.32, 1.78, 0.2, 1.35, 'dark');
    g.cylinder(-0.06, 0.94, 0.36, 0.63, 1.18, COLORS.ice);
    g.circle(-0.06, 0.94, 0.967, 0.49, 'xy', COLORS.ivory, 0.62, 0.75);
    g.circle(-0.06, 0.94, 0.971, 0.21, 'xy', COLORS.gold, 0.88, 1.05);
    g.g.rotors.push({ group: g, x: -0.06, y: 0.94, z: 0.978, radius: 0.45 });
    for (let i = 0; i < 11; i += 1) {
      const x = -1.02 + i * 0.18;
      g.line([point(x, 0.31, -0.001), point(x, 1.32, -0.001)], COLORS.ivory, 0.32, 0.65);
    }
    for (let i = 0; i < 5; i += 1) g.box(-0.7 + i * 0.34, 1.46, -0.49, 0.19, 0.13, 0.45, 'dark');
    g.box(1.02, 0.26, 0.31, 0.35, 0.58, 0.68, 'warm');
    const outputs = [];
    for (let phase = 0; phase < 3; phase += 1) outputs.push(g.insulator(1.05, 0.85, -0.04 + phase * 0.27, 0.28, 0.06));
    // Guardrail and a service pipe supply scale without adding a toy-like base.
    g.line([point(-1.1, 0.33, 1.03), point(-1.1, 0.57, 1.03), point(1.2, 0.57, 1.03), point(1.2, 0.33, 1.03)], COLORS.ice, 0.46, 0.65);
    for (let i = 0; i < 5; i += 1) g.line([point(-0.97 + i * 0.48, 0.33, 1.03), point(-0.97 + i * 0.48, 0.57, 1.03)], COLORS.ivory, 0.35, 0.5);
    return outputs;
  }

  function transformer(g, stepDown = false) {
    g.box(0, 0, 0, 1.96, 0.12, 1.8, 'dark');
    g.box(0, 0.16, -0.04, 1.27, 1.07, 1.12, 'steel');
    g.box(0, 1.23, -0.04, 1.42, 0.07, 1.24, 'warm');
    // Radiator banks and headers remain visible as separate industrial parts.
    for (const side of [-1, 1]) {
      for (let fin = 0; fin < 11; fin += 1) {
        const x = -0.76 + fin * 0.151;
        g.line([point(x, 0.28, side * 0.72), point(x, 1.14, side * 0.72)], COLORS.ivory, side === 1 ? 0.66 : 0.28, 1.0);
        g.line([point(x + 0.025, 0.3, side * 0.77), point(x + 0.025, 1.12, side * 0.77)], COLORS.ice, 0.32, 0.5);
      }
      g.line([point(-0.85, 0.3, side * 0.73), point(0.84, 0.3, side * 0.73)], COLORS.ivory, 0.52, 1.0);
      g.line([point(-0.85, 1.13, side * 0.73), point(0.84, 1.13, side * 0.73)], COLORS.ivory, 0.61, 1.0);
    }
    g.cylinder(-0.49, 1.49, -0.43, 0.15, 0.92, COLORS.ivory);
    g.line([point(-0.49, 1.34, -0.25), point(-0.49, 1.25, -0.25)], COLORS.gold, 0.55, 0.9);
    const high = [], low = [];
    for (let phase = 0; phase < 3; phase += 1) {
      // Each winding side has its own three insulated, unshorted terminals.
      high.push(g.insulator(-0.43 + phase * 0.43, 1.31, -0.21, 0.65, 0.098));
      low.push(g.insulator(-0.43 + phase * 0.43, 1.31, 0.4, 0.32, 0.07));
    }
    g.box(0.73, 0.4, 0.22, 0.22, 0.48, 0.43, 'dark');
    g.g.flux.push({ point: g.p(0, 0.88, 0.59), radius: 0.4 * g.s, phase: stepDown ? 0.73 : 0.18 });
    return { high, low };
  }

  function tower(g, height = 3.58) {
    const footX = 0.32, footZ = 0.47;
    const shoulder = 0.12;
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        g.box(sx * footX, 0, sz * footZ, 0.21, 0.08, 0.23, 'dark');
        g.line([point(sx * footX, 0.07, sz * footZ), point(sx * shoulder, height * 0.74, sz * shoulder), point(0, height, 0)], COLORS.ivory, sz > 0 ? 0.77 : 0.33, sz > 0 ? 0.92 : 0.65);
      }
    }
    for (let level = 0; level < 7; level += 1) {
      const y0 = 0.1 + level / 7 * height * 0.72;
      const y1 = 0.1 + (level + 1) / 7 * height * 0.72;
      const rx0 = lerp(footX, shoulder, level / 7), rx1 = lerp(footX, shoulder, (level + 1) / 7);
      const rz0 = lerp(footZ, shoulder, level / 7), rz1 = lerp(footZ, shoulder, (level + 1) / 7);
      for (const sign of [-1, 1]) {
        g.line([point(-rx0, y0, sign * rz0), point(rx1, y1, sign * rz1), point(-rx1, y1, sign * rz1), point(rx0, y0, sign * rz0)], COLORS.ivory, sign > 0 ? 0.54 : 0.21, 0.57);
        g.line([point(sign * rx0, y0, -rz0), point(sign * rx1, y1, rz1), point(sign * rx1, y1, -rz1), point(sign * rx0, y0, rz0)], COLORS.ice, 0.36, 0.55);
      }
    }
    const armY = height * 0.77;
    g.line([point(0, armY, -1.03), point(0, armY, 1.03)], COLORS.ivory, 0.9, 1.05);
    g.line([point(0, armY, -1.03), point(0, armY + 0.36, 0), point(0, armY, 1.03)], COLORS.ivory, 0.6, 0.72);
    for (const z of [-0.74, -0.38, 0.38, 0.74]) g.line([point(0, armY, z), point(0, armY + (1 - Math.abs(z)) * 0.34, z)], COLORS.ice, 0.6, 0.6);
    // Conductors run along the route; crossarms extend perpendicular to it.
    const terminals = [];
    terminals.push(g.insulator(0, armY - 0.36, -0.94, 0.34, 0.075));
    terminals.push(g.insulator(0, height - 0.39, 0, 0.34, 0.075));
    terminals.push(g.insulator(0, armY - 0.36, 0.94, 0.34, 0.075));
    // Connections are at the lower ends of suspension insulator strings.
    terminals[0][1] -= 0.34 * g.s;
    terminals[1][1] -= 0.34 * g.s;
    terminals[2][1] -= 0.34 * g.s;
    return terminals;
  }

  function sag(a, b, amount = 0.28, steps = 30) {
    const route = [];
    for (let i = 0; i <= steps; i += 1) {
      const t = i / steps;
      route.push(point(lerp(a[0], b[0], t), lerp(a[1], b[1], t) - 4 * amount * t * (1 - t), lerp(a[2], b[2], t)));
    }
    return route;
  }

  function receivingYard(g) {
    // Three parallel bays, each on insulating columns. The steel gantry is
    // structural; the phase conductors hang below it on separate insulators.
    for (const z of [-1.2, 1.2]) {
      g.line([point(-1.25, 0.03, z), point(-1.25, 2.5, z)], COLORS.ivory, 0.56, 1.2);
      g.line([point(-1.44, 0.03, z), point(-1.25, 0.65, z), point(-1.44, 1.25, z), point(-1.25, 1.88, z), point(-1.44, 2.5, z)], COLORS.ice, 0.33, 0.55);
    }
    g.line([point(-1.25, 2.5, -1.23), point(-1.25, 2.5, 1.23)], COLORS.ivory, 0.62, 1.0);
    const inputs = [], outputs = [];
    for (let phase = 0; phase < 3; phase += 1) {
      const z = -0.79 + phase * 0.79;
      g.insulator(-1.25, 2.09, z, 0.4, 0.075);
      const input = g.p(-1.25, 2.09, z);
      inputs.push(input);
      g.box(-0.79, 0.02, z, 0.38, 0.09, 0.31, 'dark');
      g.insulator(-0.79, 0.11, z, 0.78, 0.082);
      g.box(-0.25, 0.03, z, 0.53, 0.13, 0.35, 'dark');
      g.insulator(-0.42, 0.18, z, 0.69, 0.075);
      g.insulator(0.03, 0.18, z, 0.69, 0.075);
      g.line([point(-0.57, 0.91, z), point(0.16, 0.91, z)], COLORS.ivory, 0.72, 1.75);
      g.line([point(-0.79, 0.89, z), point(-0.46, 0.89, z)], COLORS.gold, 0.65, 0.75);
      outputs.push(g.p(0.16, 0.91, z));
      g.g.route([...sag(input, g.p(-0.79, 0.91, z), 0.04, 12), g.p(0.16, 0.91, z)], phase, 0.575, 0.04, COLORS.gold, 0.6);
    }
    for (let i = 0; i < 9; i += 1) g.line([point(-1.6 + i * 0.24, 0.02, 1.4), point(-1.6 + i * 0.24, 0.35, 1.4)], COLORS.muted, 0.22, 0.45);
    g.line([point(-1.6, 0.35, 1.4), point(0.4, 0.35, 1.4)], COLORS.ice, 0.2, 0.45);
    return { inputs, outputs };
  }

  function house(g, arrival) {
    g.box(0, 0.02, 0, 0.95, 0.69, 0.87, 'warm');
    const ridge = point(0, 1.08, 0), left = point(-0.59, 0.71, 0), right = point(0.59, 0.71, 0);
    g.face([point(left[0], left[1], -0.54), point(ridge[0], ridge[1], -0.54), point(ridge[0], ridge[1], 0.54), point(left[0], left[1], 0.54)], 'rgba(103,110,105,.32)');
    g.face([point(ridge[0], ridge[1], -0.54), point(right[0], right[1], -0.54), point(right[0], right[1], 0.54), point(ridge[0], ridge[1], 0.54)], 'rgba(160,157,137,.29)');
    g.face([point(-0.47, 0.71, 0.44), point(0, 1.01, 0.44), point(0.47, 0.71, 0.44)], 'rgba(153,125,82,.34)');
    g.line([point(-0.59, 0.71, 0.54), point(0, 1.08, 0.54), point(0.59, 0.71, 0.54)], COLORS.gold, 0.72, 0.85);
    g.line([point(0, 1.08, 0.54), point(0, 1.08, -0.54), point(0.59, 0.71, -0.54), point(0.59, 0.71, 0.54)], COLORS.ivory, 0.41, 0.7);
    for (let row = 0; row < 2; row += 1) {
      for (const x of [-0.27, 0.2]) {
        const y = 0.21 + row * 0.27;
        const corners = [g.p(x, y, 0.443), g.p(x + 0.13, y, 0.443), g.p(x + 0.13, y + 0.16, 0.443), g.p(x, y + 0.16, 0.443)];
        g.g.face(corners, 'rgba(239,179,94,.22)', 1, 0.008);
        g.g.lights.push({ points: corners, color: COLORS.gold, phase: arrival, strength: 0.52 });
      }
    }
    g.box(-0.33, 0.93, -0.26, 0.13, 0.23, 0.16, 'steel');
    return g.p(-0.45, 0.51, -0.14);
  }

  function dataCenter(g) {
    g.box(0, 0.02, 0, 2.54, 1.0, 1.45, 'dark');
    g.box(0, 1.02, 0, 2.64, 0.055, 1.53, 'steel');
    for (let bay = 0; bay < 6; bay += 1) {
      const x = -1.12 + bay * 0.42;
      g.box(x + 0.035, 0.13, 0.744, 0.27, 0.76, 0.025, 'steel', 0.45);
      for (let row = 0; row < 7; row += 1) {
        const y = 0.2 + row * 0.092;
        g.line([point(x - 0.07, y, 0.766), point(x + 0.13, y, 0.766)], COLORS.ice, 0.45, 0.6);
        const corners = [g.p(x + 0.16, y, 0.776), g.p(x + 0.19, y, 0.776), g.p(x + 0.19, y + 0.028, 0.776), g.p(x + 0.16, y + 0.028, 0.776)];
        g.g.lights.push({ points: corners, color: COLORS.ice, phase: 0.94 + row * 0.002 + bay * 0.003, strength: 0.65 });
      }
    }
    for (let x = -0.86; x < 1; x += 0.84) {
      for (const z of [-0.32, 0.34]) {
        g.box(x, 1.08, z, 0.56, 0.18, 0.47, 'steel');
        g.circle(x, 1.269, z, 0.17, 'xz', COLORS.ice, 0.57, 0.7, 24);
        g.circle(x, 1.271, z, 0.075, 'xz', COLORS.ivory, 0.5, 0.65, 20);
        for (let fin = 0; fin < 5; fin += 1) g.line([point(x - 0.21 + fin * 0.1, 1.13, z + 0.242), point(x - 0.21 + fin * 0.1, 1.23, z + 0.242)], COLORS.ivory, 0.4, 0.48);
      }
    }
    g.box(-1.39, 0.03, -0.12, 0.31, 0.65, 0.53, 'warm');
    return g.p(-1.54, 0.35, -0.12);
  }

  function connectThree(geometry, from, to, start, duration, amount = 0.13, color = COLORS.gold) {
    for (let phase = 0; phase < 3; phase += 1) geometry.route(sag(from[phase], to[phase], amount), phase, start, duration, color);
  }

  function build(mobile) {
    const geometry = new Geometry();
    const layout = mobile ? {
      generation: [-4.25, -2.12, 0.65], stepUp: [-2.04, -2.1, 0.66],
      towers: [[-0.1, -2.0, 0.74], [1.75, -1.9, 0.74], [3.5, -1.8, 0.74]],
      yard: [3.89, 1.06, 0.54], receiving: [1.98, 1.13, 0.59],
      data: [-3.17, 1.42, 0.76], houses: [[-0.72, 2.04, 0.73], [0.41, 2.13, 0.67]],
    } : {
      generation: [-11.02, 0.28, 1.04], stepUp: [-7.55, 0.02, 1.02],
      towers: [[-4.45, -0.22, 1], [-1.17, -0.36, 1.08], [2.03, -0.2, 1]],
      yard: [4.95, 0.0, 0.93], receiving: [6.52, 0.1, 0.87],
      data: [10.55, -1.69, 1.02], houses: [[9.0, 1.49, 0.91], [10.46, 1.98, 0.88], [11.92, 1.42, 0.94]],
    };
    const make = values => new Group(geometry, ...values);
    const generation = generator(make(layout.generation));
    const stepUp = transformer(make(layout.stepUp));
    const towers = layout.towers.map((position, index) => tower(make(position), index === 1 ? 3.68 : 3.52));
    const yard = receivingYard(make(layout.yard));
    const receiving = transformer(make(layout.receiving), true);
    const data = dataCenter(make(layout.data));
    const homes = layout.houses.map((position, index) => house(make(position), 0.94 + index * 0.02));

    connectThree(geometry, generation, stepUp.low, 0.015, 0.11, mobile ? 0.045 : 0.16, COLORS.ice);
    connectThree(geometry, stepUp.high, towers[0], 0.2, 0.095, mobile ? 0.075 : 0.2);
    connectThree(geometry, towers[0], towers[1], 0.295, 0.105, mobile ? 0.18 : 0.4);
    connectThree(geometry, towers[1], towers[2], 0.40, 0.105, mobile ? 0.18 : 0.39);
    connectThree(geometry, towers[2], yard.inputs, 0.505, 0.07, mobile ? 0.07 : 0.2);
    connectThree(geometry, yard.outputs, receiving.high, 0.62, 0.085, 0.055);

    // Separate outgoing feeder runs terminate at load interfaces. They never
    // bridge the transformer's primary and secondary winding terminals.
    const direction = mobile ? -1 : 1;
    const forkX = mobile ? 0.82 : 7.92;
    const forkZ = mobile ? 1.42 : 0.15;
    const fork = point(forkX, 0.36, forkZ);
    for (let phase = 0; phase < 3; phase += 1) {
      const terminal = receiving.low[phase];
      const phaseFork = point(forkX, 0.26 + phase * 0.06, forkZ + phase * 0.11);
      geometry.route([terminal, point(terminal[0] + direction * 0.15, 0.54, terminal[2]), phaseFork], phase, 0.77, 0.065, COLORS.gold, 0.7);
      const dataEnd = point(data[0], data[1] + phase * 0.04, data[2] + phase * 0.09);
      geometry.route([phaseFork, point(forkX + direction * 0.35, 0.24 + phase * 0.06, dataEnd[2]), dataEnd], phase, 0.835, 0.105, COLORS.ice, 0.65);
    }
    homes.forEach((home, index) => {
      const start = point(fork[0], fork[1] - 0.1 + index * 0.06, fork[2] + index * 0.11);
      geometry.route([start, point(start[0] + direction * 0.16, 0.24, home[2]), point(home[0] - direction * 0.1, 0.24, home[2]), home], index, 0.835, 0.105 + index * 0.02, COLORS.gold, 0.65);
    });

    // A faint site survey plane grounds the equipment without a boxed diorama.
    const halfWidth = mobile ? 5.9 : 13.4;
    for (let z = -3.1; z < 3.5; z += 0.65) geometry.line([point(-halfWidth, -0.065, z), point(halfWidth, -0.065, z)], COLORS.muted, 0.07, 0.45, -30);
    for (let x = -halfWidth; x < halfWidth; x += mobile ? 0.7 : 1.15) geometry.line([point(x, -0.068, -3.1), point(x, -0.068, 3.5)], COLORS.muted, 0.07, 0.45, -30);
    return geometry.finish();
  }

  const models = new Map();
  const temp = new Float32Array(2);
  const temp2 = new Float32Array(2);
  const lightBuffer = new Float32Array(8);

  function projection(p, view, output, index = 0) {
    const depth = p[2];
    const perspective = 28 / (28 - depth * 0.3 - p[1] * 0.1);
    output[index] = view.cx + (p[0] + depth * view.depthX) * view.scale * perspective;
    output[index + 1] = view.ground + (depth * view.depthY - p[1] * 0.95) * view.scale * perspective;
  }

  function path(ctx, points, view, buffer) {
    ctx.beginPath();
    for (let i = 0; i < points.length; i += 1) {
      projection(points[i], view, buffer, i * 2);
      if (i === 0) ctx.moveTo(buffer[0], buffer[1]);
      else ctx.lineTo(buffer[i * 2], buffer[i * 2 + 1]);
    }
  }

  function glow(ctx, x, y, radius, color, opacity) {
    ctx.fillStyle = color;
    ctx.globalAlpha = opacity * 0.055;
    ctx.beginPath(); ctx.arc(x, y, radius * 3.8, 0, TAU); ctx.fill();
    ctx.globalAlpha = opacity * 0.13;
    ctx.beginPath(); ctx.arc(x, y, radius * 1.9, 0, TAU); ctx.fill();
    ctx.globalAlpha = opacity;
    ctx.beginPath(); ctx.arc(x, y, Math.max(0.55, radius * 0.5), 0, TAU); ctx.fill();
  }

  function eventPulse(time, phase, spread = 0.1) {
    const spacing = 1 / 3;
    const cycle = ((time / 9.2 - phase) % spacing + spacing) % spacing;
    return Math.exp(-Math.pow(Math.min(cycle, spacing - cycle) / (spread * 0.72), 2));
  }

  function drawRoutes(ctx, geometry, view, time) {
    ctx.globalCompositeOperation = 'lighter';
    for (const route of geometry.routes) {
      for (let train = 0; train < 3; train += 1) {
      const clock = ((time / 9.2 - route.start - route.phase * 0.018 + train / 3) % 1 + 1) % 1;
      if (clock > route.duration) continue;
      const progress = clock / route.duration;
      const scaled = progress * (route.points.length - 1);
      const index = Math.min(route.points.length - 2, Math.floor(scaled));
      const part = scaled - index;
      const a = route.points[index], b = route.points[index + 1];
      const position = point(lerp(a[0], b[0], part), lerp(a[1], b[1], part), lerp(a[2], b[2], part));
      projection(position, view, temp);
      const fade = Math.min(1, Math.sin(progress * Math.PI) * 3.0);
      glow(ctx, temp[0], temp[1], view.mobile ? 1.5 : 1.9, route.color, fade * 0.92);
      // A short luminous trailing section makes direction legible at a glance.
      const previous = Math.max(0, scaled - 1.4);
      const pi = Math.min(route.points.length - 2, Math.floor(previous));
      const pa = route.points[pi], pb = route.points[pi + 1], pf = previous - pi;
      projection(point(lerp(pa[0], pb[0], pf), lerp(pa[1], pb[1], pf), lerp(pa[2], pb[2], pf)), view, temp2);
      ctx.strokeStyle = COLORS.white; ctx.globalAlpha = fade * 0.6; ctx.lineWidth = 1.0;
      ctx.beginPath();ctx.moveTo(temp2[0], temp2[1]);ctx.lineTo(temp[0], temp[1]);ctx.stroke();
      }
    }
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
  }

  function drawDetails(ctx, geometry, view, time) {
    for (const rotor of geometry.rotors) {
      const g = rotor.group;
      for (let blade = 0; blade < 14; blade += 1) {
        const angle = blade / 14 * TAU + time * 0.2;
        const a = g.p(rotor.x + Math.cos(angle) * rotor.radius * 0.3, rotor.y + Math.sin(angle) * rotor.radius * 0.3, rotor.z);
        const b = g.p(rotor.x + Math.cos(angle + 0.2) * rotor.radius, rotor.y + Math.sin(angle + 0.2) * rotor.radius, rotor.z);
        projection(a, view, temp); projection(b, view, temp2);
        ctx.strokeStyle = blade % 2 === 0 ? COLORS.ice : COLORS.ivory;ctx.globalAlpha = 0.8;ctx.lineWidth = 0.75;
        ctx.beginPath();ctx.moveTo(temp[0], temp[1]);ctx.lineTo(temp2[0], temp2[1]);ctx.stroke();
      }
      projection(g.p(rotor.x, rotor.y, rotor.z + 0.01), view, temp);
      glow(ctx, temp[0], temp[1], view.mobile ? 1.5 : 2.3, COLORS.ice, 0.55 + eventPulse(time, 0.01) * 0.35);
    }
    for (const flux of geometry.flux) {
      projection(flux.point, view, temp);
      const pulse = eventPulse(time, flux.phase, 0.06);
      ctx.strokeStyle = COLORS.gold;
      for (let coil = 0; coil < 2; coil += 1) {
        ctx.globalAlpha = 0.23 + pulse * 0.49;
        ctx.lineWidth = 0.75;
        ctx.beginPath();
        for (let step = 0; step <= 42; step += 1) {
          const angle = step / 42 * Math.PI * 6;
          const x = temp[0] + (coil - 0.5) * flux.radius * view.scale * 0.65 + Math.cos(angle) * flux.radius * view.scale * 0.14;
          const y = temp[1] - flux.radius * view.scale * 0.41 + step / 42 * flux.radius * view.scale * 0.82;
          if (step === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      // The two winding symbols remain separated; light conveys transfer
      // through the transformer without drawing a conductive bridge.
      if (pulse > 0.15) glow(ctx, temp[0], temp[1], view.mobile ? 1.4 : 2, COLORS.gold, pulse * 0.24);
    }
    for (const light of geometry.lights) {
      const pulse = eventPulse(time, light.phase, 0.11);
      path(ctx, light.points, view, lightBuffer);
      ctx.closePath();ctx.fillStyle = light.color;ctx.globalAlpha = 0.16 + pulse * light.strength;ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  function drawTerrain(ctx, view, time, width) {
    const half = view.mobile ? 6.4 : 14.4;
    const rows = view.mobile ? 22 : 36;
    const columns = view.mobile ? 65 : 113;
    const color = ctx.createLinearGradient(width * 0.03, 0, width * 0.97, 0);
    color.addColorStop(0, 'rgba(132,187,195,0)');
    color.addColorStop(0.15, 'rgba(132,187,195,.42)');
    color.addColorStop(0.52, 'rgba(222,219,187,.58)');
    color.addColorStop(0.83, 'rgba(204,158,94,.45)');
    color.addColorStop(1, 'rgba(204,158,94,0)');
    ctx.strokeStyle = color;
    ctx.globalCompositeOperation = 'lighter';
    for (let row = 0; row < rows; row += 1) {
      const z = -3.8 + row / (rows - 1) * 7.6;
      ctx.lineWidth = row % 7 === 0 ? 0.74 : 0.43;
      ctx.globalAlpha = (row % 7 === 0 ? 0.31 : 0.16) * (z > 1.5 ? 0.5 : 1);
      ctx.beginPath();
      for (let column = 0; column < columns; column += 1) {
        const x = -half + column / (columns - 1) * half * 2;
        const curve = Math.sin(x * 0.24 + z * 0.6 + time * 0.07);
        const y = -0.1 - curve * curve * 0.15;
        projection(point(x, y, z + Math.sin(x * 0.38 + time * 0.045) * 0.15), view, temp);
        if (column === 0) ctx.moveTo(temp[0], temp[1]); else ctx.lineTo(temp[0], temp[1]);
      }
      ctx.stroke();
    }
    // A few vertical survey threads give the contours real depth and scale.
    for (let column = 0; column < 32; column += 1) {
      const x = -half + column / 31 * half * 2;
      ctx.lineWidth = 0.4;ctx.globalAlpha = 0.11;
      ctx.beginPath();
      for (let row = 0; row < 24; row += 1) {
        const z = -3.8 + row / 23 * 7.6;
        const curve = Math.sin(x * 0.24 + z * 0.6 + time * 0.07);
        projection(point(x, -0.1 - curve * curve * 0.15, z), view, temp);
        if (row === 0) ctx.moveTo(temp[0], temp[1]); else ctx.lineTo(temp[0], temp[1]);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;ctx.globalCompositeOperation = 'source-over';
  }

  function conductorBloom(ctx, geometry, view) {
    ctx.globalCompositeOperation = 'lighter';
    for (const route of geometry.routes) {
      if (route.points[0][1] < 1.2 && route.points[route.points.length - 1][1] < 1.2) continue;
      ctx.beginPath();
      route.points.forEach((p, index) => {
        projection(p, view, temp);
        if (index === 0) ctx.moveTo(temp[0], temp[1]); else ctx.lineTo(temp[0], temp[1]);
      });
      ctx.strokeStyle = route.color;ctx.lineWidth = 4.8;ctx.globalAlpha = 0.028;ctx.stroke();
      ctx.lineWidth = 2.2;ctx.globalAlpha = 0.045;ctx.stroke();
    }
    ctx.globalAlpha = 1;ctx.globalCompositeOperation = 'source-over';
  }

  function atmosphere(ctx, width, height, view) {
    ctx.save();
    try {
      ctx.translate(width * 0.51, view.ground - view.scale * 0.7);
      ctx.scale(1, 0.24);
      const radius = width * 0.53;
      const light = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
      light.addColorStop(0, 'rgba(151,148,111,.13)');
      light.addColorStop(0.5, 'rgba(82,126,142,.065)');
      light.addColorStop(1, 'rgba(82,126,142,0)');
      ctx.fillStyle = light;ctx.fillRect(-radius, -radius, radius * 2, radius * 2);
    } finally {
      ctx.restore();
    }
    // Deterministic site particles give the fine architecture a grounded scale.
    ctx.fillStyle = COLORS.ivory;
    for (let i = 0; i < (view.mobile ? 420 : 1250); i += 1) {
      const x = Math.sin(i * 127.13) * (view.mobile ? 5.7 : 13.2);
      const z = Math.cos(i * 51.77) * 2.8;
      projection(point(x, -0.06, z), view, temp);
      ctx.globalAlpha = 0.07 + (i % 7) * 0.012;
      ctx.fillRect(temp[0], temp[1], i % 11 === 0 ? 1 : 0.65, 0.65);
    }
    ctx.globalAlpha = 1;
  }

  /** Paint one transparent, resolution-independent frame of the power journey. */
  function draw(ctx, { width, height, time = 0, pointerX = 0, pointerY = 0 }) {
    if (!ctx || !Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;
    const mobile = width <= 640;
    const t = Number.isFinite(time) ? time : 0;
    const px = Number.isFinite(pointerX) ? clamp(pointerX, -1, 1) : 0;
    const py = Number.isFinite(pointerY) ? clamp(pointerY, -1, 1) : 0;
    const view = {
      mobile, cx: width * 0.5,
      ground: height * (mobile ? 0.735 : 0.742) - (mobile ? clamp((640 - width) / 16, 0, 20) : 0),
      scale: mobile ? Math.min(width / 12.7, height * 0.038) : Math.min(width / 28.7, height * 0.05),
      depthX: 0.43 + px * 0.055,
      depthY: 0.29 + py * 0.025,
    };
    if (!models.has(mobile)) models.set(mobile, build(mobile));
    const geometry = models.get(mobile);
    ctx.save();
    try {
      ctx.clearRect(0, 0, width, height);
      ctx.lineJoin = 'round';ctx.lineCap = 'round';
      atmosphere(ctx, width, height, view);
      drawTerrain(ctx, view, t, width);
      conductorBloom(ctx, geometry, view);
      for (const item of geometry.items) {
        path(ctx, item.points, view, item.screen);
        ctx.globalAlpha = item.alpha;
        if (item.kind === 'face') { ctx.closePath();ctx.fillStyle = item.color;ctx.fill(); }
        else { ctx.strokeStyle = item.color;ctx.lineWidth = item.width;ctx.stroke(); }
      }
      ctx.globalAlpha = 1;
      drawDetails(ctx, geometry, view, t);
      drawRoutes(ctx, geometry, view, t);
    } finally {
      ctx.restore();
    }
  }

  window.PowerJourney = Object.freeze({ draw });
})();
