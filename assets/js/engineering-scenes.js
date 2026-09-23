/*
 * Conceptual engineering illustrations, not study results or actual grid topology.
 * Dependency-free orthographic 3D renderer. Animation, visibility, reduced-motion
 * preferences, and canvas resolution are owned by the page controller.
 */
(() => {
  'use strict';

  const TAU = Math.PI * 2;
  const IVORY = '#e5e7ce';
  const SAGE = '#9dbcaa';
  const LIME = '#dbef9b';
  const GOLD = '#d0ad72';
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const mix = (a, b, t) => a + (b - a) * t;
  const point = (x, y, z) => ({ x, y, z });
  const between = (a, b, t) => point(mix(a.x, b.x, t), mix(a.y, b.y, t), mix(a.z, b.z, t));
  const hex = (color, opacity) => {
    const rgb = Number.parseInt(color.slice(1), 16);
    return `rgba(${rgb >> 16},${(rgb >> 8) & 255},${rgb & 255},${opacity})`;
  };

  /** Small painter's-algorithm renderer for gently moving isometric assemblies. */
  class Scene {
    constructor(ctx, width, height, time, pointerX, pointerY, type) {
      this.ctx = ctx;
      this.width = width;
      this.height = height;
      this.time = time;
      this.items = [];
      this.scale = Math.min(width / 7.8, height / 4.3);
      this.cx = width * 0.5;
      this.cy = height * 0.66;
      this.yaw = -0.38 + pointerX * 0.22 + Math.sin(time * 0.18) * 0.06;
      this.pitch = 0.52 + pointerY * 0.065;
      if (type === 'rag') this.yaw = -0.26 + pointerX * 0.2 + Math.sin(time * 0.18) * 0.06;
      if (type === 'forecast') this.yaw = -0.27 + pointerX * 0.2 + Math.sin(time * 0.18) * 0.06;
      this.cosYaw = Math.cos(this.yaw);
      this.sinYaw = Math.sin(this.yaw);
      this.cosPitch = Math.cos(this.pitch);
      this.sinPitch = Math.sin(this.pitch);
    }

    project(p) {
      const x = p.x * this.cosYaw - p.z * this.sinYaw;
      const z = p.x * this.sinYaw + p.z * this.cosYaw;
      return {
        x: this.cx + x * this.scale,
        y: this.cy + (z * this.sinPitch - p.y * this.cosPitch) * this.scale,
        depth: z * this.cosPitch + p.y * this.sinPitch,
      };
    }

    polygon(points, fill, stroke = null, lineWidth = 0.8, order = 0) {
      const projected = points.map(p => this.project(p));
      const depth = projected.reduce((sum, p) => sum + p.depth, 0) / projected.length;
      this.items.push({ kind: 'polygon', points: projected, fill, stroke, lineWidth, depth: depth + order });
    }

    line(points, color = SAGE, width = 1, opacity = 1, order = 0) {
      const projected = points.map(p => this.project(p));
      const depth = projected.reduce((sum, p) => sum + p.depth, 0) / projected.length;
      this.items.push({ kind: 'line', points: projected, color, width, opacity, depth: depth + order });
    }

    sphere(p, radius, color = IVORY, glow = 0, order = 0) {
      const projected = this.project(p);
      this.items.push({ kind: 'sphere', p: projected, radius: radius * this.scale, color, glow, depth: projected.depth + radius + order });
    }

    box(x, y, z, width, height, depth, palette = 'ivory', order = 0) {
      const colors = {
        ivory: ['#edf0db', '#829a90', '#b4c5b3'],
        dark: ['#315e59', '#142f32', '#224541'],
        sage: ['#a8c7ad', '#476e61', '#739782'],
        gold: ['#e4cb95', '#806942', '#b29a66'],
        lime: ['#e2f1ad', '#778852', '#b9cf83'],
      }[palette];
      const x0 = x - width / 2;
      const x1 = x + width / 2;
      const z0 = z - depth / 2;
      const z1 = z + depth / 2;
      const top = y + height;
      this.polygon([point(x0, y, z1), point(x1, y, z1), point(x1, top, z1), point(x0, top, z1)], colors[1], hex(IVORY, 0.16), 0.8, order);
      this.polygon([point(x0, y, z0), point(x0, y, z1), point(x0, top, z1), point(x0, top, z0)], colors[2], hex(IVORY, 0.16), 0.8, order);
      this.polygon([point(x0, top, z0), point(x1, top, z0), point(x1, top, z1), point(x0, top, z1)], colors[0], hex(IVORY, 0.38), 0.8, order);
    }

    tube(a, b, width = 0.035, color = IVORY) {
      this.line([a, b], '#102f2d', (width + 0.035) * this.scale, 0.9);
      this.line([a, b], color, width * this.scale, 0.95, 0.002);
      this.line([a, b], '#ffffff', Math.max(0.4, width * this.scale * 0.22), 0.45, 0.004);
    }

    packet(points, progress, color = LIME, radius = 0.06) {
      const step = ((progress % 1) + 1) % 1 * (points.length - 1);
      const index = Math.min(points.length - 2, Math.floor(step));
      const p = between(points[index], points[index + 1], step - index);
      this.sphere(p, radius, color, 0.32, 0.04);
    }

    trace(points, phase = 0, color = LIME, speed = 0.22, width = 1.2) {
      this.line(points, color, width * 3.2, 0.055, 0.004);
      this.line(points, color, width, 0.47, 0.008);
      this.packet(points, this.time * speed + phase, color);
    }

    ring(x, y, z, radius, color = SAGE, opacity = 0.5, segments = 40) {
      const points = [];
      for (let i = 0; i <= segments; i += 1) {
        const angle = i / segments * TAU;
        points.push(point(x + Math.cos(angle) * radius, y, z + Math.sin(angle) * radius));
      }
      this.line(points, color, 0.85, opacity);
    }

    platform(width = 6.4, depth = 3.0) {
      const ctx = this.ctx;
      const shadow = ctx.createRadialGradient(this.cx, this.cy + this.scale * 0.5, this.scale * 0.2, this.cx, this.cy + this.scale * 0.5, this.scale * 3.4);
      shadow.addColorStop(0, 'rgba(0,0,0,.36)');
      shadow.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.save();
      ctx.translate(this.cx, this.cy + this.scale * 0.54);
      ctx.scale(1, 0.3);
      ctx.translate(-this.cx, -(this.cy + this.scale * 0.54));
      ctx.fillStyle = shadow;
      ctx.fillRect(this.cx - this.scale * 3.9, this.cy - this.scale * 3.3, this.scale * 7.8, this.scale * 7.8);
      ctx.restore();
      this.box(0, -0.18, 0, width, 0.15, depth, 'dark', -30);
      for (let x = -width / 2 + 0.4; x < width / 2; x += 0.5) {
        this.line([point(x, -0.019, -depth / 2), point(x, -0.019, depth / 2)], SAGE, 0.55, 0.12, -25);
      }
      for (let z = -depth / 2 + 0.3; z < depth / 2; z += 0.5) {
        this.line([point(-width / 2, -0.018, z), point(width / 2, -0.018, z)], SAGE, 0.55, 0.12, -25);
      }
      this.line([point(-width / 2, 0, depth / 2), point(width / 2, 0, depth / 2), point(width / 2, 0, -depth / 2)], LIME, 1, 0.28, -24);
    }

    render() {
      const ctx = this.ctx;
      this.items.sort((a, b) => a.depth - b.depth);
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      for (const item of this.items) {
        if (item.kind === 'sphere') {
          const { p, radius, color, glow } = item;
          if (glow > 0) {
            const aura = ctx.createRadialGradient(p.x, p.y, radius * 0.3, p.x, p.y, radius * 4.5);
            aura.addColorStop(0, hex(color, glow));
            aura.addColorStop(1, hex(color, 0));
            ctx.fillStyle = aura;
            ctx.beginPath();
            ctx.arc(p.x, p.y, radius * 4.5, 0, TAU);
            ctx.fill();
          }
          const shade = ctx.createRadialGradient(p.x - radius * 0.34, p.y - radius * 0.38, radius * 0.03, p.x, p.y, radius);
          shade.addColorStop(0, '#fffde9');
          shade.addColorStop(0.32, color);
          shade.addColorStop(1, '#315249');
          ctx.fillStyle = shade;
          ctx.beginPath();
          ctx.arc(p.x, p.y, radius, 0, TAU);
          ctx.fill();
          ctx.strokeStyle = hex(color, 0.52);
          ctx.lineWidth = 0.65;
          ctx.stroke();
          continue;
        }
        const points = item.points;
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i += 1) ctx.lineTo(points[i].x, points[i].y);
        if (item.kind === 'polygon') {
          ctx.closePath();
          ctx.fillStyle = item.fill;
          ctx.fill();
          if (item.stroke) {
            ctx.strokeStyle = item.stroke;
            ctx.lineWidth = item.lineWidth;
            ctx.stroke();
          }
        } else {
          ctx.globalAlpha = item.opacity;
          ctx.strokeStyle = item.color;
          ctx.lineWidth = item.width;
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }
    }
  }

  function insulator(scene, x, y, z, height = 0.25) {
    scene.tube(point(x, y, z), point(x, y + height, z), 0.045, GOLD);
    for (let i = 1; i < 5; i += 1) scene.ring(x, y + height * i / 5, z, 0.075, IVORY, 0.9, 12);
  }

  function pylon(scene, x, z, height = 2.6) {
    const size = height / 2.6;
    const foot = 0.34 * size;
    const shoulder = 0.14 * size;
    for (const dx of [-1, 1]) {
      for (const dz of [-1, 1]) {
        scene.box(x + dx * foot, 0, z + dz * foot, 0.22, 0.08, 0.22, 'sage');
        scene.tube(point(x + dx * foot, 0.08, z + dz * foot), point(x + dx * shoulder, height * 0.73, z + dz * shoulder), 0.038, IVORY);
        scene.tube(point(x + dx * shoulder, height * 0.73, z + dz * shoulder), point(x, height, z), 0.026, IVORY);
      }
    }
    for (let level = 0; level < 4; level += 1) {
      const y0 = height * 0.72 * level / 4 + 0.08;
      const y1 = height * 0.72 * (level + 1) / 4 + 0.08;
      const r0 = mix(foot, shoulder, level / 4);
      const r1 = mix(foot, shoulder, (level + 1) / 4);
      for (const sign of [-1, 1]) {
        scene.line([point(x - r0, y0, z + sign * r0), point(x + r1, y1, z + sign * r1), point(x - r1, y1, z + sign * r1), point(x + r0, y0, z + sign * r0)], IVORY, 0.8, 0.7);
        scene.line([point(x + sign * r0, y0, z - r0), point(x + sign * r1, y1, z + r1)], IVORY, 0.75, 0.6);
      }
    }
    for (const level of [0.67, 0.84]) {
      const arm = level === 0.67 ? 0.68 * size : 0.54 * size;
      scene.tube(point(x - arm, height * level, z), point(x + arm, height * level, z), 0.045, IVORY);
      scene.line([point(x - arm, height * level, z), point(x, height * level + 0.22, z), point(x + arm, height * level, z)], IVORY, 1, 0.9);
      for (const sign of [-1, 1]) insulator(scene, x + sign * arm, height * level - 0.2, z, 0.18);
    }
  }

  function chip(scene, x, y, z, size = 1.0, float = 0) {
    const top = y + 0.2 + float;
    scene.box(x, y + float, z, size, 0.2, size, 'dark');
    scene.box(x, top, z, size * 0.66, 0.07, size * 0.66, 'gold');
    scene.box(x, top + 0.07, z, size * 0.54, 0.03, size * 0.54, 'dark');
    for (let i = 0; i < 5; i += 1) {
      const offset = (i - 2) * size * 0.16;
      for (const sign of [-1, 1]) {
        scene.box(x + offset, y + 0.06 + float, z + sign * size * 0.57, size * 0.065, 0.065, size * 0.2, 'gold');
        scene.box(x + sign * size * 0.57, y + 0.06 + float, z + offset, size * 0.2, 0.065, size * 0.065, 'gold');
      }
    }
    const height = top + 0.105;
    const nodes = [point(x - size * 0.13, height, z - size * 0.12), point(x + size * 0.13, height, z - size * 0.12), point(x, height, z + size * 0.14)];
    scene.line([...nodes, nodes[0]], LIME, 1.3, 0.85, 0.6);
    nodes.forEach(p => scene.sphere(p, 0.045, LIME, 0.2, 0.61));
  }

  function drawGrid(scene) {
    scene.platform(6.4, 2.75);
    pylon(scene, -2.15, -0.15, 2.5);
    pylon(scene, -0.65, -0.91, 1.9);
    for (let phase = 0; phase < 3; phase += 1) {
      const points = [];
      for (let i = 0; i <= 24; i += 1) {
        const t = i / 24;
        points.push(point(mix(-2.62 + phase * 0.46, -1.04 + phase * 0.37, t), mix(1.9, 1.4, t) - Math.sin(t * Math.PI) * 0.3, mix(-0.15, -0.91, t)));
      }
      scene.line(points, IVORY, 0.95, 0.7);
      scene.packet(points, scene.time * 0.15 + phase / 3, phase === 1 ? GOLD : LIME, 0.035);
    }

    // A transformer tank, radiator banks, and three bushing assemblies.
    const tx = 0.25;
    const tz = 0.54;
    scene.box(tx, 0.02, tz, 1.46, 0.12, 1.02, 'sage');
    scene.box(tx, 0.14, tz, 1.0, 0.7, 0.62, 'ivory');
    scene.box(tx, 0.84, tz, 1.12, 0.07, 0.74, 'gold');
    for (let i = 0; i < 7; i += 1) scene.box(tx - 0.49 + i * 0.16, 0.19, tz + 0.39, 0.075, 0.57, 0.13, 'sage');
    for (let i = 0; i < 3; i += 1) insulator(scene, tx - 0.33 + i * 0.33, 0.91, tz, 0.36);
    scene.trace([point(-0.65, 0.08, -0.6), point(-0.65, 0.08, 0.2), point(-0.46, 0.08, 0.54)], 0.1, GOLD, 0.22);

    const lift = 0.1 * Math.sin(scene.time * 0.8);
    scene.box(2.08, 0.02, -0.07, 1.42, 0.16, 1.42, 'sage');
    scene.box(2.08, 0.18, -0.07, 1.18, 0.22, 1.18, 'dark');
    chip(scene, 2.08, 0.83, -0.07, 1.08, lift);
    for (const dx of [-0.43, 0.43]) {
      for (const dz of [-0.43, 0.43]) scene.line([point(2.08 + dx, 0.41, -0.07 + dz), point(2.08 + dx, 0.82 + lift, -0.07 + dz)], LIME, 0.8, 0.25);
    }
    scene.ring(2.08, 0.48, -0.07, 0.73, LIME, 0.28);
    scene.trace([point(0.74, 0.09, 0.57), point(1.12, 0.09, 0.57), point(1.12, 0.09, -0.07), point(1.55, 0.09, -0.07), point(1.75, 0.87 + lift, -0.07)], 0.35, LIME, 0.19);
    scene.trace([point(0.72, 0.07, 0.82), point(1.4, 0.07, 0.82), point(2.1, 0.07, 0.82)], 0.05, SAGE, 0.15, 0.8);
  }

  const ATLAS_SITES = [
    { x: -1.9, z: -0.65, height: 0.62 },
    { x: -0.86, z: 0.54, height: 0.84 },
    { x: 0.22, z: -0.64, height: 1.03 },
    { x: 1.5, z: -0.05, height: 0.67 },
    { x: 1.95, z: 0.83, height: 0.9 },
  ];

  function drawAtlas(scene) {
    scene.platform(6.1, 3.2);
    scene.box(0, 0.04, 0, 5.7, 0.13, 2.9, 'sage', -20);
    scene.box(0, 0.23, 0, 5.54, 0.07, 2.74, 'dark', -15);
    const land = [[-2.7, -1.3], [-1.4, -1.3], [-0.89, -1.05], [-0.49, -1.3], [0.52, -1.3], [0.87, -0.88], [1.66, -0.92], [2.64, -0.48], [2.64, 1.3], [1.23, 1.3], [0.78, 0.99], [-0.11, 1.31], [-1.09, 0.96], [-1.85, 1.3], [-2.7, 1.3]];
    scene.polygon(land.map(([x, z]) => point(x, 0.314, z)), '#628d75', hex(SAGE, 0.55), 0.8, -10);
    for (let line = 0; line < 7; line += 1) {
      const contour = [];
      for (let i = 0; i <= 32; i += 1) {
        const x = -2.63 + i / 32 * 5.2;
        const z = -1.12 + line * 0.36 + Math.sin(x * 2.1 + line * 0.7) * 0.12;
        contour.push(point(x, 0.328, clamp(z, -1.25, 1.25)));
      }
      scene.line(contour, IVORY, 0.7, 0.16, -5);
    }
    const links = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 4]];
    links.forEach(([a, b], i) => {
      const start = ATLAS_SITES[a];
      const end = ATLAS_SITES[b];
      const route = [point(start.x, 0.365, start.z), point(mix(start.x, end.x, 0.45), 0.365, start.z), point(end.x, 0.365, end.z)];
      scene.trace(route, i * 0.17, i === 3 ? GOLD : LIME, 0.14, 1.2);
    });
    ATLAS_SITES.forEach((site, i) => {
      const floating = Math.sin(scene.time * 0.65 + i) * 0.035;
      scene.ring(site.x, 0.355, site.z, 0.18, LIME, 0.45, 24);
      scene.box(site.x, 0.35, site.z, 0.32, 0.17, 0.28, 'ivory');
      scene.line([point(site.x, 0.53, site.z), point(site.x, site.height + 0.63 + floating, site.z)], IVORY, 0.85, 0.65);
      scene.sphere(point(site.x, site.height + 0.65 + floating, site.z), i === 2 ? 0.14 : 0.095, i === 2 ? GOLD : LIME, 0.16);
      scene.ring(site.x, site.height + 0.64 + floating, site.z, 0.2, i === 2 ? GOLD : SAGE, 0.35, 24);
    });
    // Elevated locator card is intentionally abstract: no real geography or data.
    scene.box(-1.72, 1.0, 1.15, 1.12, 0.045, 0.52, 'ivory');
    for (let line = 0; line < 3; line += 1) scene.line([point(-2.16, 1.054, 1.0 + line * 0.13), point(-1.42 + line * 0.07, 1.054, 1.0 + line * 0.13)], '#365b4e', 1.3, 0.7, 0.1);
  }

  function drawRag(scene) {
    scene.platform(6.7, 2.6);
    for (let sheet = 0; sheet < 4; sheet += 1) {
      const y = 0.17 + sheet * 0.19 + Math.sin(scene.time * 0.6 + sheet * 0.2) * 0.085;
      const x = -2.32 + sheet * 0.07;
      const z = 0.12 - sheet * 0.08;
      scene.box(x, y, z, 1.3, 0.052, 1.56, 'ivory');
      for (let line = 0; line < 5; line += 1) {
        const end = line % 3 === 0 ? x + 0.24 : x + 0.46;
        scene.line([point(x - 0.43, y + 0.058, z - 0.47 + line * 0.2), point(end, y + 0.058, z - 0.47 + line * 0.2)], '#426557', line === 0 ? 2 : 1.1, line === 0 ? 0.85 : 0.46, 0.025);
      }
      scene.box(x + 0.42, y + 0.052, z - 0.57, 0.12, 0.018, 0.12, 'gold');
    }
    scene.box(-0.02, 0.04, -0.05, 1.45, 0.1, 1.48, 'dark');
    for (let row = 0; row < 3; row += 1) {
      for (let col = 0; col < 3; col += 1) {
        const height = 0.28 + 0.12 * Math.sin(scene.time * 0.7 + col * 0.8 + row * 0.6);
        scene.box(-0.49 + col * 0.48, 0.16, -0.52 + row * 0.48, 0.34, height, 0.34, (row + col) % 3 === 0 ? 'gold' : 'sage');
        scene.box(-0.49 + col * 0.48, height + 0.16, -0.52 + row * 0.48, 0.13, 0.014, 0.13, 'lime');
      }
    }
    const fy = Math.sin(scene.time * 0.7) * 0.085;
    // An upright source-linked answer sheet with graphic lines, not generated text.
    scene.box(2.3, 0.33 + fy, 0.02, 1.58, 1.73, 0.105, 'ivory');
    scene.polygon([point(1.57, 1.91 + fy, 0.08), point(3.03, 1.91 + fy, 0.08), point(3.03, 1.54 + fy, 0.08), point(1.57, 1.54 + fy, 0.08)], '#2b6352', null, 1, 0.05);
    scene.line([point(1.73, 1.72 + fy, 0.09), point(2.69, 1.72 + fy, 0.09)], LIME, 2.0, 0.9, 0.08);
    for (let i = 0; i < 4; i += 1) scene.line([point(1.74, 1.34 - i * 0.17 + fy, 0.09), point(i === 3 ? 2.46 : 2.86, 1.34 - i * 0.17 + fy, 0.09)], '#5e7b66', 1.5, 0.8, 0.08);
    for (let i = 0; i < 3; i += 1) {
      const x = 1.78 + i * 0.32;
      scene.polygon([point(x, 0.59 + fy, 0.09), point(x + 0.23, 0.59 + fy, 0.09), point(x + 0.23, 0.43 + fy, 0.09), point(x, 0.43 + fy, 0.09)], i === 0 ? '#ad925b' : '#668f72', null, 1, 0.09);
    }
    scene.trace([point(-1.42, 0.16, 0.32), point(-1.04, 0.16, 0.32), point(-1.04, 0.16, -0.12), point(-0.68, 0.16, -0.12)], 0.0, LIME, 0.2);
    scene.trace([point(0.7, 0.15, -0.06), point(1.09, 0.15, -0.06), point(1.09, 0.76 + fy, 0.12), point(1.52, 0.76 + fy, 0.12)], 0.4, LIME, 0.2);
    const provenance = [point(2.55, 0.1, 0.18), point(2.55, 0.1, 1.05), point(-2.37, 0.1, 1.05), point(-2.37, 0.21, 0.88)];
    scene.trace(provenance, 0.28, GOLD, 0.1, 0.9);
  }

  // An irregular graph with local cycles illustrates neighbor aggregation.
  // These positions and connections are conceptual, not a study-case topology.
  const GRAPH_NODES = [
    point(-2.45, 0.62, 0.7), point(-2.02, 1.54, -0.56),
    point(-0.94, 2.05, -0.83), point(-0.82, 0.84, 0.16),
    point(0.04, 1.42, 0.87), point(0.51, 2.25, -0.69),
    point(1.04, 0.67, -0.43), point(1.59, 1.32, 0.55),
    point(2.31, 2.03, -0.34), point(2.52, 0.55, 0.85),
    point(-0.12, 0.47, -1.02),
  ];
  const GRAPH_EDGES = [
    [0, 1], [0, 3], [1, 2], [1, 3], [2, 3], [2, 5],
    [3, 4], [3, 10], [4, 5], [4, 6], [4, 7], [5, 6],
    [5, 8], [6, 7], [6, 10], [7, 8], [7, 9], [8, 9],
  ];

  function drawNeural(scene) {
    scene.platform(6.4, 2.6);
    GRAPH_EDGES.forEach(([fromIndex, toIndex], edgeIndex) => {
      const from = GRAPH_NODES[fromIndex];
      const to = GRAPH_NODES[toIndex];
      const neighborhood = fromIndex === 4 || toIndex === 4;
      const route = [from, between(from, to, 0.5), to];
      scene.line(route, neighborhood ? GOLD : SAGE, neighborhood ? 1.25 : 0.95, neighborhood ? 0.75 : 0.47);
      // Different phases make the bidirectional message passing visible.
      if (edgeIndex % 2 === 0 || neighborhood) {
        const progress = (Math.sin(scene.time * 0.9 - edgeIndex * 0.57) + 1) * 0.499;
        scene.packet(route, progress, neighborhood ? GOLD : LIME, 0.047);
      }
    });
    GRAPH_NODES.forEach((node, index) => {
      const focal = index === 4;
      const pulse = 0.12 + Math.sin(scene.time * 1.3 - index * 0.47) * 0.065;
      const color = focal ? GOLD : (index % 2 === 0 ? IVORY : SAGE);
      scene.box(node.x, 0.024, node.z, 0.23, 0.035, 0.23, 'dark');
      scene.line([point(node.x, 0.064, node.z), point(node.x, node.y - 0.19, node.z)], SAGE, 0.7, 0.15);
      scene.sphere(node, focal ? 0.24 : 0.165, color, focal ? pulse + 0.1 : pulse);
      scene.ring(node.x, node.y, node.z, focal ? 0.33 : 0.24, focal ? GOLD : LIME, 0.35, 24);
    });
    scene.ring(0.04, 0.066, 0.0, 1.1, GOLD, 0.13);
  }

  function drawForecast(scene) {
    scene.platform(6.3, 3.25);
    const left = -2.67;
    const right = 2.65;
    for (let layer = 0; layer < 3; layer += 1) {
      const z = -0.88 + layer * 0.9;
      const baseline = 0.16;
      scene.line([point(left, 2.4, z), point(left, baseline, z), point(right, baseline, z)], IVORY, 0.9, 0.45);
      for (let grid = 1; grid <= 4; grid += 1) {
        scene.line([point(left, baseline + grid * 0.46, z), point(right, baseline + grid * 0.46, z)], SAGE, 0.55, 0.14);
      }
      const curve = [];
      const color = [SAGE, GOLD, LIME][layer];
      for (let i = 0; i <= 64; i += 1) {
        const t = i / 64;
        const x = mix(left, right, t);
        const envelope = 0.67 + Math.sin(t * Math.PI) * 0.25;
        const y = 1.01 + Math.sin(t * TAU * 1.16 - scene.time * 0.32 + layer * 0.67) * 0.48 * envelope + Math.cos(t * TAU * 2.6 + layer * 0.8) * 0.16 + t * 0.37;
        curve.push(point(x, y, z));
      }
      for (let i = 0; i < curve.length - 1; i += 1) {
        const a = curve[i];
        const b = curve[i + 1];
        scene.polygon([point(a.x, baseline, z), point(b.x, baseline, z), b, a], hex(color, 0.13), null, 0);
        scene.polygon([a, b, point(b.x, b.y, z + 0.1), point(a.x, a.y, z + 0.1)], hex(color, 0.6), null, 0, 0.01);
      }
      scene.line(curve, color, 1.6, 0.95, 0.025);
      scene.packet(curve, scene.time * 0.08 + layer * 0.25, color, 0.064);
      for (let i = 8; i < 60; i += 16) scene.sphere(curve[i], 0.035, IVORY, 0, 0.035);
    }
    const scanX = mix(left, right, (Math.sin(scene.time * 0.22) + 1) / 2);
    scene.polygon([point(scanX, 0.13, -1.25), point(scanX, 2.32, -1.25), point(scanX, 2.32, 1.3), point(scanX, 0.13, 1.3)], 'rgba(217,236,169,.045)', 'rgba(217,236,169,.2)', 0.65, 0.1);
  }

  function moduleIcon(scene, x, y, z, icon) {
    const color = LIME;
    if (icon === 'database') {
      for (let i = 0; i < 3; i += 1) scene.ring(x, y + i * 0.09, z, 0.21, IVORY, 0.95, 28);
      scene.line([point(x - 0.21, y, z), point(x - 0.21, y + 0.18, z)], IVORY, 1, 0.8);
      scene.line([point(x + 0.21, y, z), point(x + 0.21, y + 0.18, z)], IVORY, 1, 0.8);
    } else if (icon === 'network') {
      const vertices = [point(x - 0.2, y, z - 0.16), point(x + 0.2, y, z - 0.16), point(x, y, z + 0.2)];
      scene.line([...vertices, vertices[0]], color, 1.5, 0.9, 0.6);
      vertices.forEach(p => scene.sphere(p, 0.053, IVORY, 0.1, 0.61));
    } else if (icon === 'check') {
      scene.line([point(x - 0.18, y, z), point(x - 0.04, y, z + 0.16), point(x + 0.23, y, z - 0.18)], color, 2.2, 1);
    } else {
      scene.line([point(x - 0.2, y, z + 0.13), point(x + 0.23, y, z), point(x - 0.2, y, z - 0.16), point(x - 0.06, y, z), point(x - 0.2, y, z + 0.13)], IVORY, 1.5, 0.9);
    }
  }

  function drawWorkflow(scene) {
    scene.platform(6.6, 2.95);
    const modules = [
      { x: -2.44, z: -0.2, y: 0.35, icon: 'database' },
      { x: -0.87, z: -0.47, y: 0.69, icon: 'network' },
      { x: 0.86, z: 0.28, y: 0.51, icon: 'check' },
      { x: 2.44, z: -0.22, y: 0.83, icon: 'delivery' },
    ];
    modules.forEach((module, i) => {
      const lift = Math.sin(scene.time * 0.7 + i * 0.8) * 0.07;
      scene.box(module.x, 0.02, module.z, 1.03, 0.09, 1.03, 'sage');
      scene.box(module.x, 0.12, module.z, 0.84, 0.16, 0.84, 'dark');
      scene.box(module.x, module.y + lift, module.z, 1.0, 0.24, 1.0, i === 2 ? 'gold' : 'ivory');
      scene.box(module.x, module.y + lift + 0.24, module.z, 0.72, 0.025, 0.72, 'dark');
      for (const offset of [-0.36, 0.36]) scene.line([point(module.x + offset, 0.29, module.z + 0.36), point(module.x + offset, module.y + lift, module.z + 0.36)], IVORY, 0.8, 0.25);
      moduleIcon(scene, module.x, module.y + lift + 0.28, module.z, module.icon);
      scene.sphere(point(module.x + 0.32, module.y + lift + 0.27, module.z + 0.35), 0.034, LIME, 0.35);
      if (i < modules.length - 1) {
        const next = modules[i + 1];
        scene.trace([point(module.x + 0.55, 0.14, module.z), point((module.x + next.x) / 2, 0.14, module.z), point((module.x + next.x) / 2, 0.14, next.z), point(next.x - 0.55, 0.14, next.z)], i * 0.21, LIME, 0.19);
      }
    });
    const loop = [point(2.44, 0.08, 0.34), point(2.44, 0.08, 1.15), point(-0.86, 0.08, 1.15), point(-0.86, 0.08, 0.09)];
    scene.trace(loop, 0.3, GOLD, 0.11, 1.1);
    scene.ring(-0.86, 0.09, 0.1, 0.16, GOLD, 0.45, 24);
  }

  const DRAWERS = Object.freeze({
    grid: drawGrid,
    atlas: drawAtlas,
    rag: drawRag,
    neural: drawNeural,
    forecast: drawForecast,
    workflow: drawWorkflow,
  });

  /** Draw one complete frame. The caller owns the DPR transform and animation. */
  function draw(ctx, { type = 'grid', width, height, time = 0, pointerX = 0, pointerY = 0 }) {
    if (!ctx || !Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;
    const render = DRAWERS[type] || DRAWERS.grid;
    const t = Number.isFinite(time) ? time : 0;
    const px = Number.isFinite(pointerX) ? clamp(pointerX, -1, 1) : 0;
    const py = Number.isFinite(pointerY) ? clamp(pointerY, -1, 1) : 0;
    ctx.save();
    ctx.clearRect(0, 0, width, height);
    // A transparent studio light leaves the host's background visible.
    const light = ctx.createRadialGradient(width * 0.43, height * 0.34, 0, width * 0.43, height * 0.34, width * 0.58);
    light.addColorStop(0, 'rgba(156,195,140,.075)');
    light.addColorStop(1, 'rgba(156,195,140,0)');
    ctx.fillStyle = light;
    ctx.fillRect(0, 0, width, height);
    const scene = new Scene(ctx, width, height, t, px, py, type);
    render(scene);
    scene.render();
    ctx.restore();
  }

  window.EngineeringScenes = Object.freeze({ draw, types: Object.freeze(Object.keys(DRAWERS)) });
})();
