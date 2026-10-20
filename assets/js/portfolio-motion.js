/* Shared, visibility-aware scheduler for decorative engineering scenes. */
(() => {
  'use strict';

  const renderers = [window.CinematicFields, window.EngineeringScenes]
    .filter((renderer) => renderer && typeof renderer.draw === 'function');
  // Reserve the project types: a missing module must not select an unrelated
  // legacy miniature that happens to advertise the same type (e.g. forecast).
  const projectRenderers = {
    fault: window.FaultScene,
    forecast: window.ForecastScene,
  };
  if (!renderers.length && !Object.values(projectRenderers).some(renderer => typeof renderer?.draw === 'function')) return;
  const media = window.matchMedia('(prefers-reduced-motion: reduce)');
  const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)');
  const root = document.documentElement;
  const toggle = document.querySelector('[data-motion-toggle]');
  const powerReplay = document.querySelector('[data-power-replay]');
  const powerPhase = document.querySelector('[data-power-phase]');
  const powerReplayLabel = document.querySelector('[data-power-replay-label]');
  const powerAnnouncement = document.querySelector('[data-power-announcement]');
  const powerTools = document.querySelector('.hero-grid-tools');
  const powerStateButtons = Array.from(document.querySelectorAll('[data-power-state]'));
  const powerContacts = Array.from(document.querySelectorAll('[data-power-contact]'))
    .map(contact => ({ contact, tip: contact.parentElement.querySelector('[data-power-contact-tip]') }));
  const powerLineStatus = document.querySelector('[data-power-detail-status="line"]');
  const powerSubstationStatus = document.querySelector('[data-power-detail-status="substation"]');
  const powerStates = ['auto', 'base', 'open', 'redistributed', 'substation'];
  const storageKey = 'portfolio-motion-paused';
  let paused = false;
  try { paused = window.localStorage.getItem(storageKey) === 'true'; } catch { /* Preferences are optional. */ }
  let frame = 0;
  let lastFrame = 0;
  let lastTick = 0;
  const scenes = [];
  const maximumBuffer = 2048;
  const modeDescriptions = {
    grid: 'A study in connection, energy, and flow.',
    knowledge: 'Many sources. Connected context. Traceable answers.',
    learning: 'Patterns emerge where connections meet.',
  };

  function updateHeroAnnotation(scene) {
    const mode = scene.host.dataset.fieldState || 'grid';
    const powerActive = mode === 'grid' && !scene.failed
      && scene.host.classList.contains('is-ready') && window.CinematicFields?.powerJourneyAvailable;
    const legend = document.querySelector('[data-power-legend]');
    const label = document.querySelector('[data-field-label]');
    const description = document.querySelector('[data-field-description]');
    if (legend) legend.hidden = !powerActive;
    if (label) label.hidden = powerActive;
    if (description) {
      description.textContent = powerActive
        ? 'Homes · Data centers · Crypto mining · Industrial · Commercial'
        : modeDescriptions[mode] || modeDescriptions.grid;
      description.classList.toggle('field-description--loads', Boolean(powerActive));
    }
  }

  const motionAllowed = () => !paused && !media.matches && !document.hidden;

  function scenePowerEvent(scene, reducedMotion = false) {
    return window.PowerJourney?.getEventState?.({
      time: scene.eventTime,
      state: scene.host.dataset.sceneState || 'auto',
      reducedMotion,
    });
  }

  function powerEventAvailable(scene) {
    return scene.type === 'field' && scene.host.dataset.fieldState === 'grid'
      && !scene.failed && scene.host.classList.contains('is-ready')
      && window.CinematicFields?.powerJourneyAvailable
      && typeof window.PowerJourney?.getEventState === 'function'
      && Boolean(window.PowerJourney.eventCues);
  }

  function updatePowerEvent(scene) {
    if (scene.type !== 'field' || !powerReplay) return;
    const available = Boolean(powerEventAvailable(scene));
    powerReplay.hidden = !available;
    if (powerTools) powerTools.hidden = !available;
    powerReplay.closest('.hero-bottom')?.classList.toggle('has-power-event', available);
    if (!available) {
      delete scene.host.dataset.powerEvent;
      return;
    }
    const still = paused || media.matches;
    const event = scenePowerEvent(scene, still);
    const isolated = event.stage === 'isolated' || event.stage === 'substation';
    const action = still ? (isolated ? 'Reset circuit' : 'Open circuit') : 'Replay contingency';
    scene.host.dataset.powerEvent = event.stage;
    if (powerPhase && powerPhase.textContent !== event.label) powerPhase.textContent = event.label;
    if (powerReplayLabel && powerReplayLabel.textContent !== action) powerReplayLabel.textContent = action;
    powerReplay.disabled = !still && scene.host.dataset.sceneState === 'auto' && !event.canReplay;
    powerReplay.setAttribute('aria-label', still ? `${action} in the illustration` : 'Replay contingency sequence');
    powerStateButtons.forEach(button => {
      const selected = String(button.dataset.powerState === (scene.host.dataset.sceneState || 'auto'));
      if (button.getAttribute('aria-pressed') !== selected) button.setAttribute('aria-pressed', selected);
    });
    // These enlarged contact details use the same opening factors as the grid.
    // The receiving-end breaker is also Substation A's HV isolation point.
    powerContacts.forEach(({ contact, tip }) => {
      const value = event[contact.dataset.powerContact];
      const opening = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
      const encoded = opening.toFixed(3);
      if (contact.dataset.opening === encoded) return;
      contact.dataset.opening = encoded;
      const endpoint = (38 - 26 * opening).toFixed(3);
      contact.setAttribute('x2', endpoint);
      if (tip) {
        tip.setAttribute('x1', endpoint);
        tip.setAttribute('x2', endpoint);
      }
    });
    const lineStatus = event.sendingOpen > .98 && event.receivingOpen > .98
      ? 'Both terminals open · circuit B in service'
      : event.sendingOpen > 0 || event.receivingOpen > 0
        ? 'A contacts open or moving · circuit B in service'
        : 'Both terminals closed · circuit B in service';
    const substationStatus = event.primaryFeederOpen > .98 && event.receivingOpen > .98
      ? 'HV + MV isolated · substation B supplies loads'
      : event.receivingOpen > .98
        ? 'HV open · MV connected · B supplies loads'
        : event.receivingOpen > 0
          ? 'HV contact moving · substation B in service'
          : 'HV + MV closed · substation B in service';
    if (powerLineStatus && powerLineStatus.textContent !== lineStatus) powerLineStatus.textContent = lineStatus;
    if (powerSubstationStatus && powerSubstationStatus.textContent !== substationStatus) powerSubstationStatus.textContent = substationStatus;
  }

  function updateStory(scene, time) {
    if (!scene.story) return;
    const ready = !scene.failed && scene.host.classList.contains('is-ready');
    const controls = scene.story.querySelector('.scene-controls');
    if (controls) controls.hidden = !ready;
    const label = scene.story.querySelector('[data-scene-phase]');
    if (!label) return;
    if (!ready) { label.textContent = scene.fallbackLabel; return; }
    const stage = scene.renderer.getStage({ time, state: scene.host.dataset.sceneState || 'auto' });
    if (stage !== scene.stage) {
      scene.stage = stage;
      label.textContent = scene.renderer.labels[stage] || scene.fallbackLabel;
    }
  }

  function draw(scene, immediateMode = false) {
    if (!scene.width || !scene.height || scene.failed) return;
    try {
      if (scene.strikeSuppressed && scenePowerEvent(scene)?.stage !== 'strike') {
        scene.strikeSuppressed = false;
      }
      const renderTime = immediateMode && scene.type === 'field' ? 0 : scene.time;
      scene.renderer.draw(scene.context, {
        type: scene.type, width: scene.width, height: scene.height,
        time: renderTime, pointerX: scene.pointerX, pointerY: scene.pointerY,
        mode: scene.host.dataset.fieldState || 'grid',
        state: scene.host.dataset.sceneState || 'auto',
        eventTime: scene.eventTime,
        reducedMotion: media.matches,
        suppressFlash: paused || Boolean(scene.strikeSuppressed),
      });
      const firstPaint = !scene.host.classList.contains('is-ready');
      scene.host.classList.add('is-ready');
      updateStory(scene, renderTime);
      if (scene.type === 'field' && (firstPaint || scene.powerAvailable !== window.CinematicFields?.powerJourneyAvailable)) {
        scene.powerAvailable = window.CinematicFields?.powerJourneyAvailable;
        updateHeroAnnotation(scene);
      }
      updatePowerEvent(scene);
    } catch (error) {
      // Restore the original project image if decorative rendering is unavailable.
      scene.failed = true;
      scene.host.classList.remove('is-ready');
      updateStory(scene, scene.time);
      if (scene.type === 'field') {
        document.querySelector('.field-modes')?.setAttribute('hidden', '');
        updateHeroAnnotation(scene);
        updatePowerEvent(scene);
      }
      console.warn(`Engineering illustration unavailable: ${scene.type}`, error);
    }
  }

  function resize(scene) {
    if (!scene.host.getClientRects().length || !scene.host.clientWidth) return;
    // Canvas bitmap dimensions must never determine the element's layout height.
    // This also bounds the display if a stale or missing stylesheet is delivered.
    const hostHeight = scene.host.clientHeight;
    // The small-screen generation district uses a deeper, vertically folded
    // composition. Other cinematic and project scenes retain their own bounds.
    const maximumHeight = scene.type === 'field' ? 1000 : scene.cinematic ? 900 : 360;
    if (hostHeight < 80 || hostHeight > maximumHeight) scene.host.style.height = scene.cinematic ? '580px' : '210px';
    const width = Math.min(2048, scene.host.clientWidth);
    const height = Math.min(maximumHeight, scene.host.clientHeight);
    if (!width || !height) return;
    const density = Math.min(window.devicePixelRatio || 1, scene.cinematic && width > 820 ? 1.5 : 2, maximumBuffer / width, maximumBuffer / height);
    const bufferWidth = Math.max(1, Math.round(width * density));
    const bufferHeight = Math.max(1, Math.round(height * density));
    if (scene.canvas.width === bufferWidth && scene.canvas.height === bufferHeight && scene.width === width && scene.height === height) return;
    scene.width = width;
    scene.height = height;
    scene.canvas.width = bufferWidth;
    scene.canvas.height = bufferHeight;
    scene.context.setTransform(bufferWidth / width, 0, 0, bufferHeight / height, 0, 0);
    draw(scene);
  }

  function animate(now) {
    frame = 0;
    if (!motionAllowed()) return;
    // A 30 fps ceiling limits work when several project cards are visible together.
    if (now - lastFrame >= 1000 / 30) {
      const delta = lastTick ? Math.min((now - lastTick) / 1000, .08) : 0;
      lastTick = now;
      lastFrame = now - ((now - lastFrame) % (1000 / 30));
      scenes.forEach((scene) => {
        if (!scene.visible || scene.failed || !scene.host.getClientRects().length) return;
        scene.time += delta;
        // The grid and its enlarged contact controls form one visible experience.
        // Hidden tabs and AI modes must not accumulate an unseen event backlog.
        if (powerEventAvailable(scene) && (scene.host.dataset.sceneState || 'auto') === 'auto') scene.eventTime += delta;
        scene.pointerX += (scene.targetX - scene.pointerX) * .11;
        scene.pointerY += (scene.targetY - scene.pointerY) * .11;
        draw(scene);
      });
    }
    if (scenes.some((scene) => scene.visible && !scene.failed)) frame = requestAnimationFrame(animate);
  }

  function start() {
    if (!frame && motionAllowed() && scenes.some((scene) => scene.visible && !scene.failed)) {
      lastTick = 0;
      frame = requestAnimationFrame(animate);
    }
  }

  function stop() {
    cancelAnimationFrame(frame);
    frame = 0;
    lastTick = 0;
  }

  const intersection = 'IntersectionObserver' in window ? new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const scene = scenes.find((item) => item.visibilityHost === entry.target);
      if (scene) {
        scene.visible = entry.isIntersecting;
        if (scene.visible) { resize(scene); draw(scene); }
      }
    });
    if (scenes.some((scene) => scene.visible && !scene.failed)) start();
    else stop();
  }, { threshold: .02 }) : null;

  const resizeObserver = 'ResizeObserver' in window ? new ResizeObserver((entries) => {
    entries.forEach((entry) => {
      const scene = scenes.find((item) => item.host === entry.target);
      if (scene) resize(scene);
    });
  }) : null;

  document.querySelectorAll('.motion-scene[data-scene]').forEach((host, index) => {
    const canvas = host.querySelector('canvas');
    const renderer = Object.hasOwn(projectRenderers, host.dataset.scene)
      ? projectRenderers[host.dataset.scene]
      : renderers.find((candidate) => candidate.types.includes(host.dataset.scene));
    if (!canvas || !renderer) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    // Defensive inline geometry breaks the old DPR/ResizeObserver feedback loop
    // even when the component stylesheet is unavailable or cached out of date.
    const cinematic = host.classList.contains('cinematic-scene');
    host.style.position = cinematic ? 'absolute' : 'relative';
    host.style.display = 'block';
    host.style.overflow = 'hidden';
    canvas.style.cssText = 'position:absolute;inset:0;display:block;width:100%;height:100%;pointer-events:none;';
    const scene = {
      host, canvas, context, renderer, cinematic, type: host.dataset.scene, visible: !intersection,
      visibilityHost: host.dataset.scene === 'field' ? host.closest('.hero') || host : host,
      width: 0, height: 0, time: 2.3 + index * .23, eventTime: 0,
      pointerX: 0, pointerY: 0, targetX: 0, targetY: 0, failed: false,
      story: host.closest('.project-story'),
    };
    scene.fallbackLabel = scene.story?.querySelector('[data-scene-phase]')?.textContent || '';
    if (scene.story) scene.time = 0;
    scenes.push(scene);
    resize(scene);
    if (intersection) intersection.observe(scene.visibilityHost);
    if (resizeObserver) resizeObserver.observe(host);
    const pointerArea = cinematic ? host.parentElement : host;
    pointerArea.addEventListener('pointermove', (event) => {
      if (!motionAllowed() || !finePointer.matches || event.pointerType === 'touch') return;
      const bounds = host.getBoundingClientRect();
      scene.targetX = Math.max(-1, Math.min(1, (event.clientX - bounds.left) / bounds.width * 2 - 1));
      scene.targetY = Math.max(-1, Math.min(1, (event.clientY - bounds.top) / bounds.height * 2 - 1));
    }, { passive: true });
    pointerArea.addEventListener('pointerleave', () => { scene.targetX = 0; scene.targetY = 0; });
  });

  scenes.filter(scene => scene.story).forEach(scene => {
    const buttons = Array.from(scene.story.querySelectorAll('.scene-controls [data-scene-state]'));
    buttons.forEach(button => button.addEventListener('click', () => {
      if (scene.failed) return;
      const state = button.dataset.sceneState;
      if (state !== 'auto' && !Object.hasOwn(scene.renderer.labels, state)) return;
      scene.host.dataset.sceneState = state;
      if (state === 'auto') scene.time = 0;
      buttons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      draw(scene);
      start();
    }));
  });

  // Mode controls alter decorative art only, not study data or project results.
  const heroScene = scenes.find((scene) => scene.type === 'field' && !scene.failed);
  const modes = document.querySelector('.field-modes');
  if (heroScene && modes) {
    const buttons = Array.from(modes.querySelectorAll('[data-field-mode]'));
    modes.hidden = false;
    buttons.forEach((button) => button.addEventListener('click', () => {
      const mode = button.dataset.fieldMode;
      if (!Object.hasOwn(modeDescriptions, mode)) return;
      heroScene.host.dataset.fieldState = mode;
      buttons.forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
      // A deliberate user selection renders a still even if automatic motion is off.
      draw(heroScene, !motionAllowed());
      updateHeroAnnotation(heroScene);
      start();
    }));
  }

  powerStateButtons.forEach(button => button.addEventListener('click', () => {
    if (!heroScene || !powerEventAvailable(heroScene)) return;
    const state = button.dataset.powerState;
    if (!powerStates.includes(state)) return;
    heroScene.host.dataset.sceneState = state;
    if (state === 'auto') heroScene.eventTime = 0;
    draw(heroScene);
    if (powerAnnouncement) {
      const event = scenePowerEvent(heroScene, !motionAllowed());
      powerAnnouncement.textContent = `${event.label}. ${state === 'substation'
        ? 'Substation A is isolated on both voltage sides. Substation B supplies distribution and loads.'
        : 'Loads remain supplied through the available paths.'}${paused || media.matches ? ' Animation remains stopped.' : ''}`;
    }
    start();
  }));

  powerReplay?.addEventListener('click', () => {
    if (!heroScene || !powerEventAvailable(heroScene) || powerReplay.disabled) return;
    const cues = window.PowerJourney.eventCues;
    if (paused || media.matches) {
      const current = scenePowerEvent(heroScene, true);
      const isolated = current.stage === 'isolated' || current.stage === 'substation';
      heroScene.eventTime = isolated ? cues.restored : cues.isolated;
      if (powerAnnouncement) powerAnnouncement.textContent = isolated
        ? 'Both transmission circuits closed. Loads remain supplied. Animation remains stopped.'
        : 'Circuit A isolated. The alternate circuit supplies distribution and loads. Animation remains stopped.';
    } else {
      heroScene.eventTime = cues.replay;
      if (powerAnnouncement) powerAnnouncement.textContent = 'Contingency sequence started. The alternate transmission circuit keeps loads supplied.';
    }
    heroScene.host.dataset.sceneState = 'auto';
    draw(heroScene);
    start();
  });

  function updatePreference() {
    stop();
    root.dataset.motion = media.matches ? 'reduced' : paused ? 'paused' : 'running';
    if (toggle) {
      toggle.hidden = scenes.length === 0;
      toggle.disabled = media.matches;
      toggle.setAttribute('aria-pressed', String(paused || media.matches));
      toggle.querySelector('[data-motion-label]').textContent = media.matches ? 'Reduced motion' : paused ? 'Play animations' : 'Pause animations';
      toggle.querySelector('[aria-hidden]').textContent = paused || media.matches ? '▷' : 'Ⅱ';
    }
    // Clear an in-progress bolt immediately when motion is disabled, including
    // while a grid scene is offscreen. Stopping RAF alone would freeze the flash.
    scenes.forEach(scene => {
      if (scene.type !== 'field') return;
      // Once interrupted, consume this bolt's remaining flash. Rapid toggling
      // of Pause/Play must not repeatedly reveal the same frozen bright frame.
      if ((paused || media.matches) && scenePowerEvent(scene)?.stage === 'strike') {
        scene.strikeSuppressed = true;
      }
      draw(scene);
    });
    start();
  }

  toggle?.addEventListener('click', () => {
    paused = !paused;
    try { window.localStorage.setItem(storageKey, String(paused)); } catch { /* Continue without persistence. */ }
    updatePreference();
  });
  if (typeof media.addEventListener === 'function') media.addEventListener('change', updatePreference);
  else if (typeof media.addListener === 'function') media.addListener(updatePreference);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); else start(); });
  window.addEventListener('resize', () => scenes.forEach(resize), { passive: true });
  updatePreference();
})();
