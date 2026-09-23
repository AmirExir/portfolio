/* A sculptural demand horizon: deterministic illustrative geometry, not data
 * from ERCOT or an application forecast. The alternative contours are not
 * statistical confidence bounds. Historical geometry never uses animation time.
 */
(() => {
  'use strict';
  const kit = window.ProjectSceneKit;
  if (!kit) return;
  const { colors, clamp, lerp, smooth } = kit;
  const labels = Object.freeze({ history: 'Historical patterns', horizon: 'Forecast origin', forecast: 'Forecast horizon' });
  const safeTime = time => Number.isFinite(time) ? Math.max(0, time) : 0;

  function getStage({ time = 0, state = 'auto' } = {}) {
    if (Object.hasOwn(labels, state)) return state;
    const cycle = safeTime(time) % 12;
    return cycle < 3 ? 'history' : cycle < 6 ? 'horizon' : 'forecast';
  }

  function progression(time, state) {
    if (Object.hasOwn(labels, state)) return {
      past: 1, plane: state === 'history' ? 0 : 1,
      future: state === 'forecast' ? 1 : state === 'horizon' ? .055 : 0,
    };
    const t = safeTime(time) % 12;
    return { past: 1, plane: smooth((t - 3) / 1.2),
      future: smooth((t - 6) / 3.8) * (1 - smooth((t - 11.3) / .7)) };
  }

  function demandShape(x, z) {
    const rhythm = Math.sin(x * 1.06 + z * .22);
    const morning = Math.exp(-Math.pow((x + 3.9 + z * .26) / 1.05, 2));
    const evening = Math.exp(-Math.pow((x - 3.45 + z * .19) / 1.2, 2));
    return .32 + rhythm * rhythm * .52 + morning * 1.54 + evening * 1.9
      + .24 * Math.cos(z * .82 + x * .27) + .12 * Math.sin(x * 2.1 + z * .65);
  }

  const meshes = new Map();
  function meshFor(mobile) {
    if (meshes.has(mobile)) return meshes.get(mobile);
    const rows = mobile ? 31 : 43, columns = 129;
    const points = [];
    for (let row = 0; row < rows; row++) {
      const z = lerp(mobile ? -3.35 : -2.45, mobile ? 3.35 : 2.45, row / (rows - 1));
      for (let col = 0; col < columns; col++) {
        const x = lerp(-6.8, 6.8, col / (columns - 1));
        points.push([x, demandShape(x, z) * (mobile ? 1.5 : 1), z]);
      }
    }
    const mesh = { rows, columns, points, screen: new Float32Array(rows * columns * 2) };
    meshes.set(mobile, mesh);
    return mesh;
  }

  function surface(ctx, mesh, width, progress, time) {
    const { rows, columns, points, screen } = mesh;
    const boundary = 64;
    const pastEnd = Math.min(boundary, Math.round(progress.past * boundary));
    const futureEnd = boundary + Math.round(progress.future * (columns - 1 - boundary));
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, '#48757e'); gradient.addColorStop(.3, '#a7d8d9');
    gradient.addColorStop(.5, '#e4e4cb'); gradient.addColorStop(.7, '#dfbd85'); gradient.addColorStop(1, '#876037');
    ctx.strokeStyle = gradient; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    const strokeRow = (row, start, end, alpha, lineWidth) => {
      if (end <= start) return;
      ctx.beginPath();
      for (let col = start; col <= end; col++) {
        const index = (row * columns + col) * 2;
        if (col === start) ctx.moveTo(screen[index], screen[index + 1]);
        else ctx.lineTo(screen[index], screen[index + 1]);
      }
      ctx.globalAlpha = alpha; ctx.lineWidth = lineWidth; ctx.stroke();
    };
    // Ghost contours provide spatial context while the narrative reveals itself.
    for (let row = 0; row < rows; row += 3) strokeRow(row, 0, columns - 1, .085, .45);
    for (let row = 0; row < rows; row++) {
      const edge = Math.sin(row / (rows - 1) * Math.PI);
      const alpha = .29 + edge * .44;
      if (row % 8 === 0) {
        strokeRow(row, 0, pastEnd, .035, 5);
        strokeRow(row, boundary, futureEnd, .045, 5);
      }
      strokeRow(row, 0, pastEnd, alpha, row % 7 === 0 ? .95 : .48);
      strokeRow(row, boundary, futureEnd, alpha * .94, row % 7 === 0 ? .95 : .48);
    }
    for (let col = 0; col < columns; col += 4) {
      const active = col <= pastEnd || (col >= boundary && col <= futureEnd);
      ctx.globalAlpha = active ? .16 : .025; ctx.lineWidth = .45; ctx.beginPath();
      for (let row = 0; row < rows; row++) {
        const index = (row * columns + col) * 2;
        if (row === 0) ctx.moveTo(screen[index], screen[index + 1]); else ctx.lineTo(screen[index], screen[index + 1]);
      }
      ctx.stroke();
    }
    for (let row = 1; row < rows; row += 3) {
      for (let col = 0; col < columns; col += 4) {
        if (col > pastEnd && (col < boundary || col > futureEnd)) continue;
        const index = (row * columns + col) * 2;
        ctx.fillStyle = col < boundary ? colors.ice : colors.gold;
        ctx.globalAlpha = .25 + (Math.sin(time * .6 + points[row * columns + col][0] * .7 + row) + 1) * .21;
        const size = row % 7 === 1 ? 1.45 : .85;
        ctx.fillRect(screen[index] - size / 2, screen[index + 1] - size / 2, size, size);
      }
    }
    ctx.globalAlpha = 1;
  }

  function originPlane(ctx, camera, progress, mobile, time) {
    const depth = mobile ? 3.55 : 2.65;
    const top = mobile ? 4.25 : 3.25;
    const plane = [[0, 0, -depth], [0, top, -depth], [0, top, depth], [0, 0, depth]];
    kit.face(ctx, plane, camera, { color: '#cfb87f', alpha: progress.plane * .04 });
    kit.line(ctx, plane, camera, { color: colors.gold, alpha: .12 + progress.plane * .48, width: .7, close: true });
    kit.line(ctx, [[0, 0, -depth], [0, 0, depth]], camera, { color: colors.gold, alpha: .2 + progress.plane * .6, dash: [2, 4] });
    if (progress.plane > .02) {
      const scan = ((time * .15) % 1 + 1) % 1;
      const z = lerp(-depth, depth, scan);
      kit.line(ctx, [[0, 0, z], [0, top, z]], camera, { color: colors.white, alpha: progress.plane * .32, width: 1 });
      for (let row = 0; row <= 28; row++) {
        const zi = lerp(-depth, depth, row / 28);
        kit.glow(ctx, [0, demandShape(0, zi) * (mobile ? 1.5 : 1), zi], camera, { color: colors.ivory, alpha: progress.plane * .5, radius: row % 7 === 0 ? 1.6 : .65 });
      }
    }
  }

  function outlook(ctx, camera, progress, mobile, time) {
    if (progress.future < .02) return;
    const stretch = mobile ? 1.5 : 1;
    for (let option = -1; option <= 1; option++) {
      const points = [];
      for (let step = 0; step <= 80 * progress.future; step++) {
        const x = step / 80 * 6.8;
        const y = (demandShape(x, .4) + option * .2 * Math.pow(x / 6.8, 1.3)) * stretch;
        points.push([x, y, .4 + option * .12]);
      }
      kit.line(ctx, points, camera, { color: option === 0 ? colors.white : colors.gold, alpha: option === 0 ? .9 : .38, width: option === 0 ? 1.1 : .7, dash: option === 0 ? [] : [3, 5] });
    }
    const x = (time * .075 % 1) * 6.8 * progress.future;
    kit.glow(ctx, [x, demandShape(x, .4) * stretch, .4], camera, { color: colors.white, alpha: .9, radius: 2 });
  }

  /** Draw one frame without changing layout, scheduling, or application data. */
  function draw(ctx, { width, height, time = 0, pointerX = 0, pointerY = 0, state = 'auto' } = {}) {
    if (!ctx || !Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;
    const t = safeTime(time), mobile = width <= 640;
    const camera = kit.view(width, height, { extent: mobile ? 15.8 : 17.2, heightExtent: 6.5,
      centerY: mobile ? .7 : .72, depthX: mobile ? .08 : .31, depthY: mobile ? .77 : .42,
      pointerX: Number.isFinite(pointerX) ? pointerX : 0, pointerY: Number.isFinite(pointerY) ? pointerY : 0 });
    const progress = progression(t, state);
    const mesh = meshFor(mobile);
    ctx.save();
    try {
      ctx.clearRect(0, 0, width, height);
      kit.backdrop(ctx, width, height);
      kit.floor(ctx, camera, { extent: 7.4, depth: mobile ? 3.5 : 2.8, spacing: .7, alpha: .12 });
      mesh.points.forEach((point, index) => {
        const screen = kit.project(point, camera);
        mesh.screen[index * 2] = screen[0]; mesh.screen[index * 2 + 1] = screen[1];
      });
      surface(ctx, mesh, width, progress, t);
      originPlane(ctx, camera, progress, mobile, t);
      outlook(ctx, camera, progress, mobile, t);
      const labelZ = mobile ? 3.7 : 3.25;
      kit.line(ctx, [[-6.8, -.08, labelZ], [6.8, -.08, labelZ]], camera, { color: colors.muted, alpha: .3, width: .6 });
      for (const x of [-6.8, 0, 6.8]) kit.line(ctx, [[x, -.08, labelZ], [x, -.08, labelZ + .16]], camera, { color: colors.ivory, alpha: .6 });
      kit.label(ctx, [-4.6, -.08, labelZ], camera, 'HISTORY', { color: colors.ice, size: mobile ? 8 : 9, offsetY: 18 });
      kit.label(ctx, [0, -.08, labelZ], camera, 'FORECAST ORIGIN', { color: colors.ivory, size: mobile ? 7 : 8, offsetY: 18 });
      kit.label(ctx, [4.6, -.08, labelZ], camera, 'FORECAST', { color: colors.gold, size: mobile ? 8 : 9, offsetY: 18 });
    } finally { ctx.restore(); }
  }

  window.ForecastScene = Object.freeze({ types: Object.freeze(['forecast']), labels, getStage, draw });
})();
