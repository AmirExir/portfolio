/* Shared drawing primitives for the portfolio's engineering story illustrations.
 * Coordinates and light intensities are artwork, not engineering measurements.
 * Renderers own their geometry; the portfolio controller owns time and sizing.
 */
(() => {
  'use strict';

  const colors = Object.freeze({ ice: '#91c8cd', gold: '#dfbb83', ivory: '#e8e5d5', white: '#fff2d4', muted: '#66848c', dark: '#101e25' });
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const lerp = (a, b, amount) => a + (b - a) * amount;
  const smooth = value => { const t = clamp(value, 0, 1); return t * t * (3 - 2 * t); };

  /** Make a shallow-perspective camera. World y points up, positive z forward. */
  function view(width, height, options = {}) {
    const { extent = 15, heightExtent = 6.5, centerX = .5, centerY = .63, pointerX = 0, pointerY = 0, depthX = .28, depthY = .42 } = options;
    const yaw = clamp(pointerX, -1, 1) * .065;
    return { width, height, cx: width * centerX, cy: height * centerY,
      scale: Math.min(width / extent, height / heightExtent),
      cos: Math.cos(yaw), sin: Math.sin(yaw), depthX, depthY: depthY + clamp(pointerY, -1, 1) * .035 };
  }

  function project(point, camera) {
    const [x, y, z = 0] = point;
    const rx = x * camera.cos + z * camera.sin;
    const rz = z * camera.cos - x * camera.sin;
    const perspective = 28 / (28 - rz * .22 - y * .06);
    return [camera.cx + (rx + rz * camera.depthX) * camera.scale * perspective,
      camera.cy + (rz * camera.depthY - y) * camera.scale * perspective];
  }

  function path(ctx, points, camera, close = false) {
    ctx.beginPath();
    points.forEach((point, index) => {
      const p = project(point, camera);
      if (index === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
    });
    if (close) ctx.closePath();
  }

  function line(ctx, points, camera, { color = colors.ice, alpha = .6, width = .75, dash = [], close = false } = {}) {
    ctx.save();
    try {
      path(ctx, points, camera, close);
      ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = width;
      ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.setLineDash(dash); ctx.stroke();
    } finally { ctx.restore(); }
  }

  function face(ctx, points, camera, { color = colors.dark, alpha = .8 } = {}) {
    ctx.save();
    try { path(ctx, points, camera, true); ctx.fillStyle = color; ctx.globalAlpha = alpha; ctx.fill(); }
    finally { ctx.restore(); }
  }

  function box(ctx, camera, { x = 0, y = 0, z = 0, w = 1, h = 1, d = 1, alpha = 1, color = colors.ice } = {}) {
    const x0 = x - w / 2, x1 = x + w / 2, z0 = z - d / 2, z1 = z + d / 2;
    const a = [x0, y, z1], b = [x1, y, z1], c = [x1, y + h, z1], e = [x0, y + h, z1];
    const f = [x0, y + h, z0], g = [x1, y + h, z0], j = [x1, y, z0];
    face(ctx, [a, b, c, e], camera, { color: '#263b42', alpha: alpha * .85 });
    face(ctx, [b, j, g, c], camera, { color: '#1a2d35', alpha: alpha * .85 });
    face(ctx, [e, c, g, f], camera, { color: '#657a79', alpha: alpha * .35 });
    line(ctx, [a, b, c, e, a], camera, { color, alpha: alpha * .55 });
    line(ctx, [e, f, g, c], camera, { color, alpha: alpha * .7 });
    line(ctx, [g, j, b], camera, { color, alpha: alpha * .4 });
  }

  function glow(ctx, point, camera, { color = colors.gold, alpha = 1, radius = 2 } = {}) {
    const p = project(point, camera);
    ctx.save();
    try {
      ctx.globalCompositeOperation = 'lighter'; ctx.fillStyle = color;
      for (const [size, opacity] of [[4, .045], [2, .12], [.55, 1]]) {
        ctx.globalAlpha = alpha * opacity;
        ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(.4, radius * size), 0, Math.PI * 2); ctx.fill();
      }
    } finally { ctx.restore(); }
  }

  function floor(ctx, camera, { extent = 7, depth = 4, spacing = .75, alpha = .13 } = {}) {
    for (let x = -extent; x <= extent; x += spacing) line(ctx, [[x, -.05, -depth], [x, -.05, depth]], camera, { color: colors.muted, alpha, width: .45 });
    for (let z = -depth; z <= depth; z += spacing) line(ctx, [[-extent, -.05, z], [extent, -.05, z]], camera, { color: colors.muted, alpha, width: .45 });
  }

  function backdrop(ctx, width, height) {
    ctx.save();
    try {
      const light = ctx.createRadialGradient(width * .53, height * .65, 0, width * .53, height * .65, width * .62);
      light.addColorStop(0, 'rgba(94,139,146,.095)'); light.addColorStop(.55, 'rgba(126,113,80,.035)'); light.addColorStop(1, 'rgba(50,74,83,0)');
      ctx.fillStyle = light; ctx.fillRect(0, 0, width, height);
    } finally { ctx.restore(); }
  }

  function label(ctx, point, camera, text, { color = colors.muted, alpha = 1, size = 10, align = 'center', offsetY = 0 } = {}) {
    const p = project(point, camera);
    ctx.save();
    try { ctx.font = `${size}px ui-sans-serif, system-ui, sans-serif`; ctx.fillStyle = color; ctx.globalAlpha = alpha; ctx.textAlign = align; ctx.fillText(text, p[0], p[1] + offsetY); }
    finally { ctx.restore(); }
  }

  window.ProjectSceneKit = Object.freeze({ colors, clamp, lerp, smooth, view, project, line, face, box, glow, floor, backdrop, label });
})();
