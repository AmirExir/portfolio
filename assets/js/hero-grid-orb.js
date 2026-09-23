(() => {
  'use strict';

  const host = document.querySelector('[data-grid-orb]');
  const canvas = host?.querySelector('canvas');
  const hero = host?.closest('.hero');
  if (!host || !canvas || !hero) return;

  const context = canvas.getContext('2d');
  if (!context) return;

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const palette = {
    brand: [23, 79, 70],
    brandStrong: [17, 60, 53],
    sage: [151, 176, 126],
    brass: [164, 132, 72],
    paper: [250, 249, 245],
  };
  const nodes = Array.from({ length: 22 }, (_, index) => {
    const y = 1 - (index / 21) * 2;
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const angle = index * Math.PI * (3 - Math.sqrt(5));
    return {
      x: Math.cos(angle) * radius,
      y,
      z: Math.sin(angle) * radius,
      isSubstation: index % 5 === 0,
    };
  });
  const edges = [];
  nodes.forEach((node, leftIndex) => {
    const nearest = nodes
      .map((candidate, rightIndex) => ({
        rightIndex,
        distance: Math.hypot(
          node.x - candidate.x,
          node.y - candidate.y,
          node.z - candidate.z,
        ),
      }))
      .filter(({ rightIndex }) => rightIndex !== leftIndex)
      .sort((left, right) => left.distance - right.distance)
      .slice(0, 3);
    nearest.forEach(({ rightIndex }) => {
      const edge = leftIndex < rightIndex
        ? [leftIndex, rightIndex]
        : [rightIndex, leftIndex];
      if (!edges.some(([start, end]) => start === edge[0] && end === edge[1])) {
        edges.push(edge);
      }
    });
  });

  let width = 0;
  let height = 0;
  let frame = 0;
  let visible = true;
  let pointerX = 0;
  let pointerY = 0;
  let targetPointerX = 0;
  let targetPointerY = 0;

  function rgba(color, alpha) {
    return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
  }

  function resize() {
    const bounds = host.getBoundingClientRect();
    width = Math.max(1, bounds.width);
    height = Math.max(1, bounds.height);
    const density = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * density);
    canvas.height = Math.round(height * density);
    context.setTransform(density, 0, 0, density, 0, 0);
    draw(performance.now());
  }

  function rotate(point, rotationX, rotationY, rotationZ = 0) {
    const cosX = Math.cos(rotationX);
    const sinX = Math.sin(rotationX);
    const cosY = Math.cos(rotationY);
    const sinY = Math.sin(rotationY);
    const cosZ = Math.cos(rotationZ);
    const sinZ = Math.sin(rotationZ);
    const y1 = point.y * cosX - point.z * sinX;
    const z1 = point.y * sinX + point.z * cosX;
    const x2 = point.x * cosY + z1 * sinY;
    const z2 = -point.x * sinY + z1 * cosY;
    return {
      x: x2 * cosZ - y1 * sinZ,
      y: x2 * sinZ + y1 * cosZ,
      z: z2,
    };
  }

  function project(point, radius) {
    const perspective = 3.6 / (4.4 - point.z);
    return {
      x: width / 2 + point.x * radius * perspective,
      y: height / 2 + point.y * radius * perspective,
      z: point.z,
      scale: perspective,
    };
  }

  function drawRing(radius, rotationX, rotationY, rotationZ, color, alpha) {
    context.beginPath();
    for (let index = 0; index <= 72; index += 1) {
      const angle = (index / 72) * Math.PI * 2;
      const point = rotate(
        { x: Math.cos(angle), y: Math.sin(angle), z: 0 },
        rotationX,
        rotationY,
        rotationZ,
      );
      const projected = project(point, radius);
      if (index === 0) context.moveTo(projected.x, projected.y);
      else context.lineTo(projected.x, projected.y);
    }
    context.strokeStyle = rgba(color, alpha);
    context.lineWidth = 1;
    context.stroke();
  }

  function drawTransformer(centerX, centerY, size, time) {
    const pulse = 0.72 + Math.sin(time * 0.0024) * 0.12;
    context.save();
    context.translate(centerX, centerY);
    context.fillStyle = rgba(palette.paper, 0.82);
    context.strokeStyle = rgba(palette.brandStrong, 0.68);
    context.lineWidth = 1;
    context.beginPath();
    if (typeof context.roundRect === 'function') {
      context.roundRect(-size * 0.34, -size * 0.25, size * 0.68, size * 0.5, size * 0.08);
    } else {
      context.rect(-size * 0.34, -size * 0.25, size * 0.68, size * 0.5);
    }
    context.fill();
    context.stroke();

    [-1, 1].forEach((side) => {
      context.strokeStyle = side < 0
        ? rgba(palette.brand, pulse)
        : rgba(palette.brass, pulse);
      context.lineWidth = Math.max(1.2, size * 0.025);
      for (let coil = -2; coil <= 2; coil += 1) {
        context.beginPath();
        context.arc(side * size * 0.105, coil * size * 0.04, size * 0.055, -Math.PI / 2, Math.PI / 2);
        context.stroke();
      }
    });
    context.restore();
  }

  function draw(time) {
    if (!width || !height) return;
    context.clearRect(0, 0, width, height);
    pointerX += (targetPointerX - pointerX) * 0.055;
    pointerY += (targetPointerY - pointerY) * 0.055;

    const radius = Math.min(width, height) * 0.39;
    const automaticTurn = reducedMotion.matches ? 0.48 : time * 0.00022;
    const rotationX = -0.32 + pointerY * 0.42;
    const rotationY = automaticTurn + pointerX * 0.55;
    const rotationZ = -0.08 + Math.sin(time * 0.00035) * 0.06;

    const glow = context.createRadialGradient(
      width * 0.43,
      height * 0.38,
      0,
      width / 2,
      height / 2,
      radius * 1.18,
    );
    glow.addColorStop(0, rgba(palette.paper, 0.88));
    glow.addColorStop(0.43, rgba(palette.sage, 0.14));
    glow.addColorStop(1, rgba(palette.paper, 0));
    context.fillStyle = glow;
    context.fillRect(0, 0, width, height);

    drawRing(radius * 1.03, rotationX + 1.16, rotationY, rotationZ, palette.brand, 0.33);
    drawRing(radius * 0.92, rotationX, rotationY + 0.78, rotationZ + 0.56, palette.brass, 0.46);
    drawRing(radius * 0.8, rotationX + 0.42, rotationY - 0.25, rotationZ - 0.62, palette.sage, 0.42);

    const projectedNodes = nodes.map((node) => {
      const rotated = rotate(node, rotationX, rotationY, rotationZ);
      return { ...project(rotated, radius), isSubstation: node.isSubstation };
    });

    edges
      .map(([start, end], index) => ({
        start: projectedNodes[start],
        end: projectedNodes[end],
        index,
      }))
      .sort((left, right) => (left.start.z + left.end.z) - (right.start.z + right.end.z))
      .forEach(({ start, end, index }) => {
        const depth = Math.max(0.12, Math.min(0.72, ((start.z + end.z) / 2 + 1.4) / 3));
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.strokeStyle = rgba(palette.brand, depth * 0.58);
        context.lineWidth = 0.75 + depth;
        context.stroke();

        if (index % 6 === 0 && !reducedMotion.matches) {
          const progress = (time * 0.00022 + index * 0.137) % 1;
          const x = start.x + (end.x - start.x) * progress;
          const y = start.y + (end.y - start.y) * progress;
          context.beginPath();
          context.arc(x, y, 1.7, 0, Math.PI * 2);
          context.fillStyle = rgba(palette.brass, 0.9);
          context.shadowColor = rgba(palette.brass, 0.8);
          context.shadowBlur = 7;
          context.fill();
          context.shadowBlur = 0;
        }
      });

    projectedNodes
      .slice()
      .sort((left, right) => left.z - right.z)
      .forEach((node) => {
        const depth = Math.max(0.18, Math.min(1, (node.z + 1.45) / 2.55));
        const nodeRadius = (node.isSubstation ? 3.5 : 2.25) * node.scale;
        context.beginPath();
        context.arc(node.x, node.y, nodeRadius + 1.2, 0, Math.PI * 2);
        context.fillStyle = rgba(palette.paper, 0.78);
        context.fill();
        context.beginPath();
        context.arc(node.x, node.y, nodeRadius, 0, Math.PI * 2);
        context.fillStyle = node.isSubstation
          ? rgba(palette.brass, 0.55 + depth * 0.4)
          : rgba(palette.brand, 0.4 + depth * 0.55);
        context.fill();
      });

    drawTransformer(width / 2, height / 2, radius * 0.72, time);
  }

  function animate(time) {
    draw(time);
    frame = visible && !document.hidden && !reducedMotion.matches
      ? window.requestAnimationFrame(animate)
      : 0;
  }

  function start() {
    if (!frame && visible && !document.hidden && !reducedMotion.matches) {
      frame = window.requestAnimationFrame(animate);
    } else if (reducedMotion.matches) {
      draw(performance.now());
    }
  }

  function stop() {
    if (frame) window.cancelAnimationFrame(frame);
    frame = 0;
  }

  hero.addEventListener('pointermove', (event) => {
    if (reducedMotion.matches || event.pointerType === 'touch') return;
    const bounds = hero.getBoundingClientRect();
    targetPointerX = ((event.clientX - bounds.left) / bounds.width - 0.5) * 2;
    targetPointerY = ((event.clientY - bounds.top) / bounds.height - 0.5) * 2;
    host.style.setProperty('--orb-x', `${targetPointerX * 10}px`);
    host.style.setProperty('--orb-y', `${targetPointerY * 8}px`);
  });
  hero.addEventListener('pointerleave', () => {
    targetPointerX = 0;
    targetPointerY = 0;
    host.style.setProperty('--orb-x', '0px');
    host.style.setProperty('--orb-y', '0px');
  });

  if ('IntersectionObserver' in window) {
    const visibilityObserver = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) start();
      else stop();
    }, { threshold: 0.05 });
    visibilityObserver.observe(hero);
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stop();
    else start();
  });
  const handleMotionPreference = () => {
    stop();
    start();
  };
  if (typeof reducedMotion.addEventListener === 'function') {
    reducedMotion.addEventListener('change', handleMotionPreference);
  } else if (typeof reducedMotion.addListener === 'function') {
    reducedMotion.addListener(handleMotionPreference);
  }

  if ('ResizeObserver' in window) new ResizeObserver(resize).observe(host);
  else window.addEventListener('resize', resize);
  resize();
  host.classList.add('is-ready');
  start();
})();
