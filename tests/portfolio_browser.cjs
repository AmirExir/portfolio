// Optional offline browser regression checks; setup is documented in docs/portfolio.md.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PORTFOLIO_PLAYWRIGHT_MODULE || 'playwright');

const root = path.resolve(__dirname, '..');
const realFeed = JSON.parse(fs.readFileSync(path.join(root, 'ERCOTAPI/latest_ercot_updates.json'), 'utf8'));
const sceneTypes = ['field', 'contingency', 'atlas', 'evidence', 'rag', 'learning', 'forecast', 'fault', 'workflow'];
const cinematicTypes = ['field', 'contingency', 'evidence', 'learning', 'forecast', 'fault'];
const maximumSceneHeight = type => type === 'field' ? 1000 : cinematicTypes.includes(type) ? 900 : 360;
const powerLoadLabels = ['Homes', 'Data centers', 'Crypto mining', 'Industrial', 'Commercial'];
const powerSourceLabels = ['Nuclear', 'Gas', 'Hydro', 'Wind', 'Solar', 'Battery storage'];
const projectStories = [
  { type: 'contingency', id: 'aelab-story', module: 'contingency-scene.js', renderer: 'ContingencyScene', states: ['base', 'open', 'redistributed'] },
  { type: 'fault', id: 'fault-story', module: 'fault-scene.js', renderer: 'FaultScene', states: ['signals', 'features', 'classified'] },
  { type: 'forecast', id: 'forecast-story', module: 'forecast-scene.js', renderer: 'ForecastScene', states: ['history', 'horizon', 'forecast'] },
];

async function waitForPaint(scene) {
  await scene.scrollIntoViewIfNeeded();
  await scene.evaluate(element => new Promise((resolve, reject) => {
    const deadline = performance.now() + 5000;
    const check = () => {
      const canvas = element.querySelector('canvas');
      if (element.classList.contains('is-ready') && canvas.width && canvas.height) {
        const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        for (let index = 3; index < pixels.length; index += 4) {
          if (pixels[index]) return resolve();
        }
      }
      if (performance.now() > deadline) return reject(new Error(`Scene ${element.dataset.scene} did not paint`));
      requestAnimationFrame(check);
    };
    check();
  }));
}

const frame = scene => scene.locator('canvas').evaluate(canvas => canvas.toDataURL());

async function sceneBounds(page) {
  return page.locator('.motion-scene').evaluateAll(scenes => scenes.map(scene => {
    const box = scene.getBoundingClientRect();
    const canvas = scene.querySelector('canvas');
    const drawing = canvas.getBoundingClientRect();
    return {
      scene: scene.dataset.scene, width: box.width, height: box.height,
      canvasWidth: drawing.width, canvasHeight: drawing.height,
      backingWidth: canvas.width, backingHeight: canvas.height,
      ready: scene.classList.contains('is-ready'),
    };
  }));
}

async function assertBoundedScenes(page, label) {
  const before = await sceneBounds(page);
  await page.waitForTimeout(450);
  const after = await sceneBounds(page);
  assert.equal(before.length, after.length);
  for (let index = 0; index < after.length; index += 1) {
    const box = after[index];
    if (!box.width || !box.height) continue; // A category filter can hide a scene's card.
    const maximumHeight = maximumSceneHeight(box.scene);
    assert.ok(box.height <= maximumHeight, `${label}: ${box.scene} must remain within its ${maximumHeight}px layout limit`);
    assert.ok(Math.abs(box.height - before[index].height) <= 1, `${label}: canvas painting must not change ${box.scene} layout height`);
    assert.ok(box.canvasWidth <= box.width + 2 && box.canvasHeight <= box.height + 2, `${label}: ${box.scene} canvas must fit its scene`);
    if (!box.ready) continue; // Offscreen scenes may defer their initial allocation.
    assert.ok(box.backingWidth <= box.canvasWidth * 2 + 2, `${label}: ${box.scene} pixel width must be capped at 2x`);
    assert.ok(box.backingHeight <= box.canvasHeight * 2 + 2, `${label}: ${box.scene} pixel height must be capped at 2x`);
    assert.ok(box.backingWidth >= box.canvasWidth - 2 && box.backingHeight >= box.canvasHeight - 2, `${label}: ${box.scene} must paint at least at CSS resolution`);
  }
}

async function serveLocalFiles(context, { omitMotionStyles = false, omitPowerJourney = false, omitModules = [] } = {}) {
  await context.route('**/*', (route) => {
    const url = new URL(route.request().url());
    if (url.hostname !== 'portfolio.test') return route.abort();
    if (omitMotionStyles && ['/portfolio-motion.css', '/portfolio-cinematic.css', '/project-stories.css'].some(file => url.pathname.endsWith(file))) return route.abort();
    if (omitPowerJourney && url.pathname.endsWith('/power-journey.js')) return route.abort();
    if (omitModules.some(file => url.pathname.endsWith(`/${file}`))) return route.abort();
    const file = path.join(root, decodeURIComponent(url.pathname === '/' ? '/index.html' : url.pathname));
    return fs.existsSync(file) ? route.fulfill({ path: file }) : route.fulfill({ status: 404, body: '' });
  });
}

async function assertProjectFallback(page, label, projects = projectStories) {
  for (const project of projects) {
    const story = page.locator(`#${project.id}`);
    assert.equal(await story.locator('.scene-controls').isVisible(), false, `${label}: unavailable ${project.type} controls should remain hidden`);
    assert.equal(await story.locator('.motion-scene').evaluate(scene => scene.classList.contains('is-ready')), false, `${label}: ${project.type} must not advertise a successfully rendered scene`);
    assert.ok(await story.locator('h2, h3').first().isVisible(), `${label}: ${project.type} story heading must remain readable`);
    assert.equal(await story.locator('img.scene-fallback').isVisible(), true, `${label}: ${project.type} must preserve its actual screenshot fallback`);
    assert.ok(await story.locator('img.scene-fallback').evaluate(image => Number(getComputedStyle(image).opacity) > 0), `${label}: ${project.type} fallback must not be transparent`);
    assert.ok(await story.evaluate(element => {
      const section = element.parentElement.closest('section');
      return Array.from(section.querySelectorAll('.card-actions a[href]')).some(link => link.getClientRects().length && getComputedStyle(link).visibility !== 'hidden');
    }), `${label}: the ${project.type} project section must retain working destinations`);
  }
}

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.PORTFOLIO_BROWSER_EXECUTABLE || undefined,
    headless: true,
    args: ['--disable-gpu', '--no-first-run'],
  });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  let feed = realFeed;
  let feedStatus = 200;
  await context.route('**/*', (route) => {
    const url = new URL(route.request().url());
    if (url.hostname !== 'portfolio.test') return route.abort();
    if (url.pathname.endsWith('latest_ercot_updates.json')) {
      return route.fulfill({ status: feedStatus, contentType: 'application/json', body: JSON.stringify(feed) });
    }
    const relative = decodeURIComponent(url.pathname === '/' ? '/index.html' : url.pathname);
    const file = path.join(root, relative);
    if (!fs.existsSync(file)) return route.fulfill({ status: 404, body: '' });
    return route.fulfill({ path: file });
  });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.goto('http://portfolio.test/');
  await page.waitForFunction(() => document.querySelector('#ercotUpdatesList').getAttribute('aria-busy') === 'false');
  assert.equal(await page.locator('.hero-photo').getAttribute('src'), 'AmirinSubstation.jpeg');
  const heroScene = page.locator('.hero .motion-scene[data-scene="field"]');
  const motionToggle = page.locator('[data-motion-toggle]').first();
  const fieldModes = page.locator('[data-field-mode]');
  const powerLegend = page.locator('[data-power-legend]');
  const powerSources = page.locator('[data-power-sources]');
  const powerReplay = page.locator('[data-power-replay]');
  const powerPhase = page.locator('[data-power-phase]');
  const fieldDescription = page.locator('[data-field-description]');
  assert.equal(await fieldModes.count(), 3, 'The hero should expose its grid, knowledge, and learning modes');
  assert.equal(await page.locator('[data-project-card] .work-image img:visible').count(), 6, 'Selected work should show actual project screenshots');
  assert.ok(await page.locator('[data-project-card] .work-image img').evaluateAll(images => images.every(image => Number(getComputedStyle(image).opacity) > 0)), 'Project screenshots must remain visibly painted when scripts run');
  const sceneFrames = [];
  for (const type of sceneTypes) {
    const scene = page.locator(`.motion-scene[data-scene="${type}"]`).first();
    await waitForPaint(scene);
    sceneFrames.push(await frame(scene));
  }
  assert.equal(new Set(sceneFrames).size, sceneTypes.length, 'Project areas should use distinct rendered scenes');
  await waitForPaint(heroScene);
  assert.equal(await powerLegend.isVisible(), true, 'The power journey should identify its conceptual stages');
  const journeyStages = powerLegend.locator('ol > li');
  const stageLabels = (await journeyStages.allTextContents()).map(text => text.replace(/\s+/g, ' ').trim());
  assert.deepEqual(stageLabels, ['Generation + storage', 'Step-up', 'Transmission', 'Step-down substation', 'Distribution', 'Loads'], 'The power journey must distinguish storage, step-down and distribution before the load stage');
  assert.deepEqual((await powerSources.locator('span').allTextContents()).map(text => text.trim()), powerSourceLabels, 'Power mode should name all six illustrated generation and storage technologies');
  assert.equal(await powerSources.isVisible(), true);
  assert.equal(await powerReplay.isVisible(), true);
  assert.equal(await powerReplay.getAttribute('aria-controls'), 'heroField');
  assert.equal(await powerReplay.getAttribute('aria-describedby'), await powerPhase.getAttribute('id'));
  assert.equal(await powerPhase.getAttribute('aria-live'), 'off', 'Automatic event phases must not repeatedly announce themselves');
  const powerDescription = await fieldDescription.textContent();
  for (const label of powerLoadLabels) {
    assert.ok(powerDescription.includes(label), `Power mode should identify the ${label.toLowerCase()} load type`);
  }
  assert.doesNotMatch(await page.locator('.hero').textContent(), /\b(?:simulation|electrons?|trajector(?:y|ies))\b/i, 'Engineering caveats belong in the documentation, not the hero caption');
  const movingFrame = await frame(heroScene);
  await page.waitForTimeout(250);
  assert.notEqual(
    await frame(heroScene), movingFrame,
    'The hero scene should animate when motion is allowed',
  );
  assert.equal(await page.locator('html').getAttribute('data-motion'), 'running');
  await page.evaluate(() => {
    const actual = window.PowerJourney;
    window.portfolioTestEventTimes = [];
    window.PowerJourney = Object.freeze({ ...actual, draw(ctx, options) {
      actual.draw(ctx, options);
      window.portfolioTestEventTime = options.eventTime;
      window.portfolioTestEventTimes.push(options.eventTime);
    } });
  });
  await powerReplay.click();
  await page.waitForFunction(() => document.querySelector('[data-power-replay]').disabled);
  assert.ok(await page.evaluate(() => window.portfolioTestEventTimes.includes(window.PowerJourney.eventCues.replay)), 'Replay must restart the dedicated event clock at its cue');
  assert.match(await powerReplay.locator('[data-power-replay-label]').textContent(), /Replay strike/i);
  const replayAnnouncement = await page.locator('[data-power-announcement]').textContent();
  assert.ok(replayAnnouncement.trim(), 'A deliberate replay should receive an accessible confirmation');
  await page.waitForTimeout(100);
  assert.equal(await page.locator('[data-power-announcement]').textContent(), replayAnnouncement, 'Automatic event phases must not rewrite the manual-action announcement');
  await motionToggle.click();
  assert.equal(await motionToggle.getAttribute('aria-pressed'), 'true');
  assert.match(await motionToggle.textContent(), /Play animations/i);
  assert.equal(await page.locator('html').getAttribute('data-motion'), 'paused');
  await waitForPaint(heroScene);
  await page.waitForTimeout(150);
  const manuallyPausedFrame = await frame(heroScene);
  await page.waitForTimeout(250);
  assert.equal(await frame(heroScene), manuallyPausedFrame, 'The global pause button should stop animation');
  assert.equal(await powerReplay.isDisabled(), false, 'Paused users should be able to inspect static outage states');
  // The pause may land partway through tripping. Normalize to powered before
  // checking the two deliberate static states, independently of wall-clock load.
  if (/Restore power/i.test(await powerReplay.locator('[data-power-replay-label]').textContent())) await powerReplay.click();
  assert.match(await powerReplay.locator('[data-power-replay-label]').textContent(), /Show outage/i);
  const pausedPoweredFrame = await frame(heroScene);
  await powerReplay.click();
  const outageLabel = await page.evaluate(() => window.PowerJourney.getEventState({ time: window.PowerJourney.eventCues.isolated }).label);
  assert.equal(await page.evaluate(() => window.portfolioTestEventTime), 10);
  assert.equal((await powerPhase.textContent()).trim(), outageLabel);
  assert.match(await powerReplay.locator('[data-power-replay-label]').textContent(), /Restore power/i);
  const pausedOutageFrame = await frame(heroScene);
  assert.notEqual(pausedOutageFrame, pausedPoweredFrame, 'Show outage should visibly change the static power illustration');
  await page.waitForTimeout(150);
  assert.equal(await frame(heroScene), pausedOutageFrame, 'An outage inspection must not restart paused motion');
  assert.equal(await page.locator('html').getAttribute('data-motion'), 'paused');
  await powerReplay.click();
  const restoredLabel = await page.evaluate(() => window.PowerJourney.getEventState({ time: window.PowerJourney.eventCues.restored }).label);
  assert.equal(await page.evaluate(() => window.portfolioTestEventTime), 19);
  assert.equal((await powerPhase.textContent()).trim(), restoredLabel);
  assert.match(await powerReplay.locator('[data-power-replay-label]').textContent(), /Show outage/i);
  const pausedRestoredFrame = await frame(heroScene);
  assert.notEqual(pausedRestoredFrame, pausedOutageFrame, 'Restore power should relight the static scene');
  await page.waitForTimeout(150);
  assert.equal(await frame(heroScene), pausedRestoredFrame);
  const modeFrames = [manuallyPausedFrame];
  for (const [mode, key] of [['knowledge', 'Enter'], ['learning', 'Space']]) {
    const button = page.locator(`[data-field-mode="${mode}"]`);
    assert.equal(await button.getAttribute('aria-controls'), 'heroField');
    await button.focus();
    await page.keyboard.press(key);
    assert.equal(await button.getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('[data-field-mode][aria-pressed="true"]').count(), 1, 'Exactly one hero mode should be selected');
    assert.equal(await heroScene.getAttribute('data-field-state'), mode);
    assert.equal(await powerLegend.isVisible(), false, 'Power-stage labels must be hidden in AI modes');
    assert.equal(await powerSources.isVisible(), false, 'Generation/storage labels must be hidden in AI modes');
    assert.equal(await powerReplay.isVisible(), false, 'Power-event controls must be hidden in AI modes');
    assert.equal(await page.locator('[data-field-label]').isVisible(), true, 'AI modes should retain their general illustration label');
    assert.notEqual(await fieldDescription.textContent(), powerDescription, 'AI modes must replace the power-load caption with their own description');
    await page.waitForTimeout(150);
    const selectedFrame = await frame(heroScene);
    assert.notEqual(selectedFrame, modeFrames.at(-1), 'Selecting a mode should redraw even while automatic motion is paused');
    await page.waitForTimeout(150);
    assert.equal(await frame(heroScene), selectedFrame, 'Mode selection must not resume paused animation');
    modeFrames.push(selectedFrame);
  }
  assert.equal(new Set(modeFrames).size, 3, 'Grid, knowledge, and learning must produce distinct hero views');
  await page.locator('[data-field-mode="grid"]').click();
  assert.equal(await heroScene.getAttribute('data-field-state'), 'grid');
  assert.equal(await powerLegend.isVisible(), true);
  assert.equal(await powerSources.isVisible(), true);
  assert.equal(await powerReplay.isVisible(), true);
  assert.equal(await fieldDescription.textContent(), powerDescription, 'Returning to power mode should restore all load labels');
  for (const type of sceneTypes) {
    const scene = page.locator(`.motion-scene[data-scene="${type}"]`).first();
    await waitForPaint(scene);
    const pausedFrame = await frame(scene);
    await page.waitForTimeout(150);
    assert.equal(await frame(scene), pausedFrame, `Global pause must apply to the ${type} renderer`);
  }
  for (const project of projectStories) {
    const story = page.locator(`#${project.id}`);
    const scene = story.locator(`.motion-scene[data-scene="${project.type}"]`);
    await waitForPaint(scene);
    const controls = story.locator('.scene-controls');
    const phase = story.locator('[data-scene-phase]');
    assert.equal(await controls.isVisible(), true, `${project.type}: painted stories should enable their controls`);
    assert.equal(await controls.locator('button[data-scene-state]').count(), 4, `${project.type}: each story needs Auto and three manual stages`);
    assert.equal(await phase.getAttribute('aria-live'), 'off', 'Automatic visual phases must not repeatedly interrupt assistive technology');
    const selectedFrames = [];
    for (const [index, state] of project.states.entries()) {
      const button = controls.locator(`button[data-scene-state="${state}"]`);
      assert.equal(await button.getAttribute('aria-controls'), await scene.getAttribute('id'));
      await button.focus();
      await page.keyboard.press(index % 2 ? 'Space' : 'Enter');
      assert.equal(await button.getAttribute('aria-pressed'), 'true', `${project.type}: keyboard selection should set ${state}`);
      assert.equal(await controls.locator('[aria-pressed="true"]').count(), 1, `${project.type}: exactly one state should be selected`);
      assert.equal(await scene.getAttribute('data-scene-state'), state);
      const expectedLabel = await page.evaluate(({ renderer, state }) => window[renderer].labels[state], { renderer: project.renderer, state });
      assert.equal((await phase.textContent()).trim(), expectedLabel, `${project.type}: the phase label must describe the selected stage`);
      await page.waitForTimeout(100);
      const selectedFrame = await frame(scene);
      await page.waitForTimeout(150);
      assert.equal(await frame(scene), selectedFrame, `${project.type}: choosing ${state} must not resume paused animation`);
      assert.equal(await page.locator('html').getAttribute('data-motion'), 'paused');
      selectedFrames.push(selectedFrame);
    }
    assert.equal(new Set(selectedFrames).size, project.states.length, `${project.type}: manual stages need visibly distinct artwork`);
    const autoButton = controls.locator('button[data-scene-state="auto"]');
    await autoButton.focus();
    await page.keyboard.press('Enter');
    assert.equal(await scene.getAttribute('data-scene-state'), 'auto');
    assert.equal(await autoButton.getAttribute('aria-pressed'), 'true');
    const autoPausedFrame = await frame(scene);
    await page.waitForTimeout(150);
    assert.equal(await frame(scene), autoPausedFrame, `${project.type}: selecting Auto must preserve the global pause`);
  }
  await motionToggle.click();
  assert.equal(await motionToggle.getAttribute('aria-pressed'), 'false');
  assert.match(await motionToggle.textContent(), /Pause animations/i);
  await waitForPaint(heroScene);
  const resumedFrame = await frame(heroScene);
  await page.waitForTimeout(250);
  assert.notEqual(await frame(heroScene), resumedFrame, 'Play should resume animation');
  for (const project of projectStories) {
    const scene = page.locator(`#${project.id} .motion-scene`);
    await waitForPaint(scene);
    const playingFrame = await frame(scene);
    await page.waitForTimeout(250);
    assert.notEqual(await frame(scene), playingFrame, `${project.type}: Auto should animate after the global control resumes motion`);
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(150);
    const outsideFrame = await frame(scene);
    await page.waitForTimeout(150);
    assert.equal(await frame(scene), outsideFrame, `${project.type}: project artwork should stop advancing offscreen`);
  }
  await waitForPaint(heroScene);
  // Exercise the visibility-change handler without relying on the window
  // manager to background a headless browser tab.
  await page.evaluate(() => {
    Object.defineProperty(document, 'hidden', { configurable: true, value: true });
    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' });
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await page.waitForTimeout(150);
  const hiddenFrame = await frame(heroScene);
  await page.waitForTimeout(250);
  assert.equal(await frame(heroScene), hiddenFrame, 'The visibility handler should suspend rendering in a hidden tab');
  await page.evaluate(() => {
    delete document.hidden;
    delete document.visibilityState;
    document.dispatchEvent(new Event('visibilitychange'));
  });
  const visibleAgainFrame = await frame(heroScene);
  await page.waitForTimeout(250);
  assert.notEqual(await frame(heroScene), visibleAgainFrame, 'The visibility handler should restart rendering for a visible tab');
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(300);
  const offscreenFrame = await frame(heroScene);
  await page.waitForTimeout(250);
  assert.equal(
    await frame(heroScene), offscreenFrame,
    'A scene should stop rendering when it is off screen',
  );
  await assertBoundedScenes(page, 'Desktop 1x');
  await page.evaluate(() => window.scrollTo(0, 0));

  const education = page.locator('#education');
  const educationCards = education.locator('article.experience-card');
  assert.equal(await education.getAttribute('aria-labelledby'), 'educationTitle');
  assert.equal(await educationCards.count(), 3, 'The restored education timeline should include all three institutions');
  const expectedEducation = [
    [/M\.S\. in Artificial Intelligence/, /University of Texas at Austin.*Aug 2024.*Present/],
    [/M\.Eng\. in Electrical.*Computer Engineering/, /Lamar University.*Jan 2019.*May 2020/],
    [/B\.S\. in Electrical.*Computer Engineering/, /Shahid Beheshti University.*Oct 2012.*Jul 2017/],
  ];
  for (const [index, [degree, institutionAndDates]] of expectedEducation.entries()) {
    assert.match(await educationCards.nth(index).locator('h3').textContent(), degree);
    assert.match(await educationCards.nth(index).locator('.meta').textContent(), institutionAndDates);
  }
  assert.match(await educationCards.first().locator('.study-status').textContent(), /Degree candidate.*Graduating/, 'Current AI study must not be presented as a conferred degree');
  const educationLink = page.locator('a[href="#education"]').first();
  await educationLink.focus();
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => location.hash === '#education' && document.getElementById('educationTitle').getBoundingClientRect().top < innerHeight);
  assert.equal(await education.locator('h2').isVisible(), true, 'The education section must be reachable through its native anchor');

  const visibleCards = () => page.locator('[data-project-card]:not([hidden])').count();
  assert.equal(await visibleCards(), 6);
  const originalCardMediaHeights = await page.locator('[data-project-card] .work-image').evaluateAll(images => images.map(image => image.clientHeight));
  assert.equal(await page.locator('[data-project-tools][hidden]').count(), 0);
  await page.locator('[data-project-filter="engineering"]').click();
  assert.equal(await visibleCards(), 2);
  await page.locator('#searchInput').fill('GNN');
  assert.equal(await visibleCards(), 0);
  assert.equal(await page.locator('#projectEmpty').isVisible(), true);
  await page.locator('#clearSearch').click();
  assert.equal(await visibleCards(), 6);
  assert.equal(await page.locator('#searchInput').inputValue(), '');
  assert.equal(await page.locator('[data-project-filter="all"]').getAttribute('aria-pressed'), 'true');
  assert.equal(await page.locator('#projectEmpty').isVisible(), false);
  await page.locator('#searchInput').fill('  GNN voltage  ');
  assert.equal(await visibleCards(), 1);
  assert.match(await page.locator('[data-project-card]:not([hidden])').textContent(), /GNN predictor/);
  await page.locator('#searchInput').fill('');
  await page.waitForTimeout(150);
  assert.deepEqual(
    await page.locator('[data-project-card] .work-image').evaluateAll(images => images.map(image => image.clientHeight)),
    originalCardMediaHeights,
    'Filtering and restoring hidden cards must preserve their project-image layout',
  );
  const scrollPositions = await page.evaluate(() => {
    window.scrollTo(0, 1000);
    const before = window.scrollY;
    const input = document.getElementById('searchInput');
    input.value = 'ERCOT';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    return [before, window.scrollY];
  });
  assert.equal(scrollPositions[0], scrollPositions[1], 'Searching should not force page scrolling');

  const carousel = page.locator('[data-carousel]').first();
  const slides = carousel.locator('.carousel-slide');
  const slideCount = await slides.count();
  assert.equal(await slides.evaluateAll((elements) => elements.filter((element) => !element.inert).length), 1);
  await carousel.locator('[data-next]').click();
  assert.equal(await carousel.locator('.carousel-status').textContent(), `Slide 2 of ${slideCount}`);
  assert.equal(await slides.nth(0).getAttribute('aria-hidden'), 'true');
  assert.equal(await slides.nth(1).getAttribute('aria-hidden'), 'false');
  await carousel.focus();
  await page.keyboard.press('ArrowLeft');
  assert.equal(await carousel.locator('.carousel-status').textContent(), `Slide 1 of ${slideCount}`);
  await carousel.locator('[data-prev]').click();
  assert.equal(await carousel.locator('.carousel-status').textContent(), `Slide ${slideCount} of ${slideCount}`);
  await carousel.locator('.carousel-dot').nth(0).click();
  await page.waitForTimeout(5500);
  assert.equal(await carousel.locator('.carousel-status').textContent(), `Slide 1 of ${slideCount}`, 'Galleries must remain manual');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('#menuToggle').click();
  assert.equal(await page.locator('#menuToggle').getAttribute('aria-expanded'), 'true');
  assert.equal(await page.locator('#navLinks').isVisible(), true);
  assert.equal(await page.locator('#menuToggle').getAttribute('aria-label'), 'Close navigation');
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#menuToggle').getAttribute('aria-expanded'), 'false');
  await page.locator('#menuToggle').click();
  const menuBounds = await page.locator('#navLinks').boundingBox();
  await page.mouse.click(380, menuBounds.y + menuBounds.height + 30);
  assert.equal(await page.locator('#menuToggle').getAttribute('aria-expanded'), 'false');

  await page.locator('[data-copy-email]').click();
  await page.waitForFunction(() => document.getElementById('copyEmailStatus').textContent.includes('Copy is unavailable'));
  assert.match(await page.locator('#copyEmailStatus').textContent(), /Select and copy: contact@amirexirpe.com/);
  const expectedDocuments = Math.min(12, realFeed.items.length);
  assert.equal(await page.locator('#ercotUpdatesList > li').count(), expectedDocuments);
  const generationDate = new Date(realFeed.generated_at);
  const expectedSnapshot = Number.isNaN(generationDate.getTime())
    ? 'Snapshot generation date unavailable.'
    : `Snapshot generated ${generationDate.toLocaleDateString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC',
    })} (UTC).`;
  assert.ok((await page.locator('#ercotUpdatesStatus').textContent()).startsWith(expectedSnapshot));
  assert.equal(await page.locator('#ercotUpdatesList .update-effectiveness').count(), expectedDocuments);

  feed = { generated_at: 'bad-date', items: [{ title: '<img src=x onerror=alert(1)>', url: 'javascript:alert(1)', status: 'Pending', effectiveness_note: 'Pending proposal.' }] };
  await page.reload();
  await page.waitForFunction(() => document.querySelector('#ercotUpdatesList').getAttribute('aria-busy') === 'false');
  assert.equal(await page.locator('#ercotUpdatesList a').count(), 0);
  assert.equal(await page.locator('#ercotUpdatesList img').count(), 0);
  assert.match(await page.locator('#ercotUpdatesStatus').textContent(), /generation date unavailable/);
  assert.match(await page.locator('#ercotUpdatesList strong').textContent(), /<img src=x/);
  assert.match(await page.locator('#ercotUpdatesList .update-meta').textContent(), /Status: Pending/);
  assert.equal(await page.locator('#ercotUpdatesList .update-effectiveness').textContent(), 'Pending proposal.');

  feed = { generated_at: '2026-01-01', items: [null] };
  await page.reload();
  await page.waitForFunction(() => document.querySelector('#ercotUpdatesList').getAttribute('aria-busy') === 'false');
  assert.match(await page.locator('#ercotUpdatesStatus').textContent(), /could not be loaded/);
  assert.equal(await page.locator('#ercotUpdatesList > li').count(), 0);

  feed = realFeed;
  feedStatus = 503;
  await page.reload();
  await page.waitForFunction(() => document.querySelector('#ercotUpdatesList').getAttribute('aria-busy') === 'false');
  assert.match(await page.locator('#ercotUpdatesStatus').textContent(), /could not be loaded/);
  assert.equal(errors.length, 0, errors.join('\n'));
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true, 'Mobile page must not overflow horizontally');
  // Layout regressions: presentation changes must preserve filtering,
  // readable mobile inputs, usable targets, and the no-JavaScript fallback.
  await page.emulateMedia({ reducedMotion: 'reduce' });
  let reducedPowerJourneyFrame;
  const layoutViewports = [
    { width: 1440, height: 900 }, { width: 1440, height: 800 },
    ...[1024, 820, 768, 700, 641, 640, 481, 480, 390, 320].map(width => ({ width, height: 900 })),
  ];
  for (const { width, height } of layoutViewports) {
    const viewportLabel = `${width}×${height}px`;
    await page.setViewportSize({ width, height });
    await page.reload();
    await page.waitForFunction(() => document.documentElement.classList.contains('js'));
    assert.equal(await visibleCards(), 6);
    if (width === 1440 && height === 900) {
      await waitForPaint(heroScene);
      assert.equal(await page.locator('html').getAttribute('data-motion'), 'reduced');
      assert.equal(await motionToggle.isDisabled(), true, 'The system reduced-motion preference must take priority');
      const staticFrame = await frame(heroScene);
      reducedPowerJourneyFrame = staticFrame;
      await page.waitForTimeout(250);
      assert.equal(
        await frame(heroScene), staticFrame,
        'Reduced motion should retain a static painted scene',
      );
      assert.equal(await powerReplay.isDisabled(), false, 'Reduced motion should still allow static outage inspection');
      await powerReplay.click();
      assert.equal((await powerPhase.textContent()).trim(), outageLabel);
      const reducedOutageFrame = await frame(heroScene);
      assert.notEqual(reducedOutageFrame, staticFrame);
      await page.waitForTimeout(150);
      assert.equal(await frame(heroScene), reducedOutageFrame, 'Reduced-motion outage selection must remain still');
      assert.equal(await page.locator('html').getAttribute('data-motion'), 'reduced');
      await powerReplay.click();
      assert.equal((await powerPhase.textContent()).trim(), restoredLabel);
      assert.notEqual(await frame(heroScene), reducedOutageFrame);
      for (const type of sceneTypes) {
        const scene = page.locator(`.motion-scene[data-scene="${type}"]`).first();
        await waitForPaint(scene);
        const reducedFrame = await frame(scene);
        await page.waitForTimeout(150);
        assert.equal(await frame(scene), reducedFrame, `Reduced motion must apply to the ${type} renderer`);
      }
      for (const project of projectStories) {
        const story = page.locator(`#${project.id}`);
        const scene = story.locator('.motion-scene');
        await waitForPaint(scene);
        const initialFrame = await frame(scene);
        const lastState = project.states.at(-1);
        await story.locator(`.scene-controls button[data-scene-state="${lastState}"]`).click();
        assert.equal(await scene.getAttribute('data-scene-state'), lastState, 'Reduced motion must still permit deliberate state changes');
        const selectedFrame = await frame(scene);
        assert.notEqual(selectedFrame, initialFrame, `${project.type}: reduced-motion state selection should redraw`);
        await page.waitForTimeout(150);
        assert.equal(await frame(scene), selectedFrame, `${project.type}: state selection must retain reduced-motion stillness`);
        await story.locator('.scene-controls button[data-scene-state="auto"]').click();
      }
    }
    await assertBoundedScenes(page, `${viewportLabel} 1x`);
    const educationBounds = await educationCards.evaluateAll(cards => cards.map(card => {
      const { left, right, height } = card.getBoundingClientRect();
      return { left, right, height };
    }));
    assert.equal(educationBounds.length, 3);
    assert.ok(educationBounds.every(box => box.left >= 0 && box.right <= width + 1 && box.height > 0), `${viewportLabel}: all education entries must remain readable within the viewport`);
    const legendBounds = await powerLegend.boundingBox();
    assert.ok(legendBounds && legendBounds.x >= 0 && legendBounds.x + legendBounds.width <= width + 1, `${viewportLabel}: power journey labels must remain inside the viewport`);
    const sourceBounds = await powerSources.locator('span').evaluateAll(labels => labels.map(element => {
      const { left, right, height } = element.getBoundingClientRect();
      return { left, right, height, clipped: element.scrollWidth > element.clientWidth + 1 || element.scrollHeight > element.clientHeight + 1 };
    }));
    assert.equal(sourceBounds.length, powerSourceLabels.length);
    assert.ok(sourceBounds.every(box => box.left >= 0 && box.right <= width + 1 && box.height > 0 && !box.clipped), `${viewportLabel}: all six generation/storage labels must fit without clipping`);
    const replayBounds = await powerReplay.boundingBox();
    assert.ok(replayBounds && replayBounds.height >= 44 && replayBounds.width >= 44, `${viewportLabel}: the event control needs a 44px hit area`);
    assert.ok(replayBounds.x >= 0 && replayBounds.x + replayBounds.width <= width + 1, `${viewportLabel}: the event control must fit the viewport`);
    const descriptionBounds = await fieldDescription.evaluate(description => {
      const { left, right } = description.getBoundingClientRect();
      return { left, right, clipped: description.scrollWidth > description.clientWidth + 1 || description.scrollHeight > description.clientHeight + 1 };
    });
    assert.ok(descriptionBounds.left >= 0 && descriptionBounds.right <= width + 1 && !descriptionBounds.clipped, `${viewportLabel}: all load labels must fit without clipping`);
    const bounds = await page.locator('[data-project-card]').evaluateAll(cards => cards.map(card => {
      const { width, top } = card.getBoundingClientRect();
      return { width, top };
    }));
    if (width > 1100) {
      assert.ok(bounds[0].width > bounds[2].width * 1.5, 'Lead projects should be visibly featured');
      assert.equal(bounds[0].top, bounds[1].top, 'Both featured projects share one row');
      assert.ok(bounds[2].top > bounds[0].top, 'Supporting work follows the featured row');
    }
    for (const category of ['engineering', 'ai', 'data', 'all']) {
      await page.locator(`[data-project-filter="${category}"]`).click();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `${viewportLabel} overflow for ${category}`);
      const visibleBounds = await page.locator('[data-project-card]:not([hidden])').evaluateAll(cards => cards.map(card => {
        const { left, right } = card.getBoundingClientRect();
        return { left, right };
      }));
      assert.ok(visibleBounds.every(box => box.left >= 0 && box.right <= width + 1), 'Visible cards stay within the viewport');
      if (category !== 'all' && width > 1100) {
        assert.equal(await page.locator('[data-project-card]:not([hidden])').first().evaluate(card => getComputedStyle(card).gridColumnEnd), 'auto', 'Filtering restores regular card spans');
      }
    }
    assert.equal(await page.locator('#menuToggle').isVisible(), width <= 820);
    for (const mode of ['grid', 'knowledge', 'learning']) {
      const target = await page.locator(`[data-field-mode="${mode}"]`).boundingBox();
      assert.ok(target.height >= 44 && target.width >= 44, `${viewportLabel}: the ${mode} mode needs a 44px hit area`);
      assert.ok(target.x >= 0 && target.x + target.width <= width + 1, `${viewportLabel}: hero mode controls must remain inside the viewport`);
    }
    for (const project of projectStories) {
      const story = page.locator(`#${project.id}`);
      await waitForPaint(story.locator('.motion-scene'));
      for (const state of ['auto', ...project.states]) {
        const target = await story.locator(`.scene-controls button[data-scene-state="${state}"]`).boundingBox();
        assert.ok(target && target.height >= 44 && target.width >= 44, `${viewportLabel}: ${project.type}/${state} needs a 44px hit area`);
        assert.ok(target.x >= 0 && target.x + target.width <= width + 1, `${viewportLabel}: ${project.type} controls must fit the viewport`);
      }
      const phase = await story.locator('[data-scene-phase]').evaluate(element => {
        const { left, right } = element.getBoundingClientRect();
        return { left, right, clipped: element.scrollWidth > element.clientWidth + 1 || element.scrollHeight > element.clientHeight + 1 };
      });
      assert.ok(phase.left >= 0 && phase.right <= width + 1 && !phase.clipped, `${viewportLabel}: the ${project.type} phase should remain legible`);
    }
    if (width <= 820) {
      await page.locator('#menuToggle').click();
      assert.equal(await page.locator('#navLinks').isVisible(), true);
      await page.keyboard.press('Escape');
      assert.equal(await page.locator('#navLinks').isVisible(), false);
      assert.equal(await page.locator('#menuToggle').evaluate(toggle => toggle === document.activeElement), true);
    }
    const firstGallery = page.locator('[data-carousel]').first();
    for (const selector of ['[data-next]', '.carousel-dot']) {
      const target = await firstGallery.locator(selector).first().boundingBox();
      assert.ok(target.width >= 44 && target.height >= 44, `${selector} must have a 44px hit area`);
    }
    const galleryBounds = await page.locator('[data-carousel]').evaluateAll(galleries => galleries.map(gallery => {
      const track = gallery.querySelector('.carousel-track').getBoundingClientRect();
      const dots = gallery.querySelector('[data-dots]').getBoundingClientRect();
      const controls = Array.from(gallery.querySelectorAll('.carousel-control:not([hidden])')).map(control => {
        const { top, bottom } = control.getBoundingClientRect();
        return { top, bottom };
      });
      return { id: gallery.id, trackTop: track.top, trackBottom: track.bottom, dotsTop: dots.top, controls };
    }));
    for (const gallery of galleryBounds) {
      for (const control of gallery.controls) {
        assert.ok(control.top >= gallery.trackTop - 1 && control.bottom <= gallery.trackBottom + 1, `${gallery.id} arrows must stay inside the image at ${viewportLabel}`);
        assert.ok(control.bottom <= gallery.dotsTop + 1, `${gallery.id} arrows must not overlap pagination at ${viewportLabel}`);
      }
    }
    if (width <= 820) {
      assert.ok(await page.locator('#searchInput').evaluate(input => parseFloat(getComputedStyle(input).fontSize) >= 16), `${viewportLabel}: mobile search should not trigger iOS text zoom`);
    }
  }
  const keyboardLink = page.locator('.work-card h3 a').first();
  await keyboardLink.focus();
  assert.equal(await keyboardLink.evaluate(link => link.closest('.work-card').matches(':focus-within')), true);
  assert.notEqual(await keyboardLink.evaluate(link => getComputedStyle(link).outlineStyle), 'none');
  assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).scrollBehavior), 'auto', 'Reduced motion disables smooth scrolling');

  // Canvas pixel dimensions must never feed back into CSS layout, including
  // Retina screens where the backing bitmap is twice the displayed size.
  const retinaContext = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 2, reducedMotion: 'no-preference' });
  await serveLocalFiles(retinaContext);
  const retinaPage = await retinaContext.newPage();
  retinaPage.on('pageerror', error => errors.push(error.message));
  await retinaPage.goto('http://portfolio.test/');
  for (const width of [1280, 640, 390, 320]) {
    await retinaPage.setViewportSize({ width, height: 900 });
    for (const type of sceneTypes) await waitForPaint(retinaPage.locator(`.motion-scene[data-scene="${type}"]`).first());
    await assertBoundedScenes(retinaPage, `${width}px 2x`);
    assert.equal(await retinaPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `${width}px Retina layout must not overflow`);
  }
  await retinaContext.close();

  // Each project illustration is optional. In particular, a missing new
  // ForecastScene must not silently resolve to the older compact forecast art.
  for (const omitModules of [projectStories.map(project => project.module), ['project-scene-kit.js']]) {
    const missingStoriesContext = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: 'reduce' });
    await serveLocalFiles(missingStoriesContext, { omitModules });
    const missingStoriesPage = await missingStoriesContext.newPage();
    missingStoriesPage.on('pageerror', error => errors.push(error.message));
    await missingStoriesPage.goto('http://portfolio.test/');
    await waitForPaint(missingStoriesPage.locator('#heroField'));
    // Scrolling is necessary if missing dependencies are discovered at first draw.
    for (const project of projectStories) {
      await missingStoriesPage.locator(`#${project.id}`).scrollIntoViewIfNeeded();
      await missingStoriesPage.waitForTimeout(100);
    }
    await assertProjectFallback(missingStoriesPage, `Missing ${omitModules.join(', ')}`);
    assert.equal(await missingStoriesPage.locator('[data-project-card] .work-image img:visible').count(), 6, 'Missing project artwork must retain actual screenshots');
    assert.equal(await missingStoriesPage.locator('.hero-photo').isVisible(), true);
    assert.equal(await missingStoriesPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await missingStoriesContext.close();
  }

  // Project renderers must also work independently of the original families.
  const standaloneStoriesContext = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: 'reduce' });
  await serveLocalFiles(standaloneStoriesContext, { omitModules: ['cinematic-fields.js', 'engineering-scenes.js'] });
  const standaloneStoriesPage = await standaloneStoriesContext.newPage();
  standaloneStoriesPage.on('pageerror', error => errors.push(error.message));
  await standaloneStoriesPage.goto('http://portfolio.test/');
  for (const project of projectStories) {
    const story = standaloneStoriesPage.locator(`#${project.id}`);
    const scene = story.locator('.motion-scene');
    await waitForPaint(scene);
    const initial = await frame(scene);
    const lastState = project.states.at(-1);
    const button = story.locator(`.scene-controls button[data-scene-state="${lastState}"]`);
    await button.click();
    assert.equal(await button.getAttribute('aria-pressed'), 'true');
    assert.equal(await scene.getAttribute('data-scene-state'), lastState);
    assert.notEqual(await frame(scene), initial, `${project.type}: controls must work without the older rendering families`);
  }
  assert.equal(await standaloneStoriesPage.locator('.motion-scene.is-ready').count(), 3);
  assert.equal(await standaloneStoriesPage.locator('#heroField .field-static').isVisible(), true, 'The hero should retain its static fallback when its renderer is unavailable');
  assert.equal(await standaloneStoriesPage.locator('.hero h1').isVisible(), true);
  assert.equal(await standaloneStoriesPage.locator('.field-modes').isVisible(), false);
  assert.equal(await standaloneStoriesPage.locator('[data-power-legend]').isVisible(), false);
  assert.equal(await standaloneStoriesPage.locator('.hero-photo').isVisible(), true);
  await standaloneStoriesContext.close();

  const failedStoryContext = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: 'no-preference' });
  await serveLocalFiles(failedStoryContext);
  await failedStoryContext.addInitScript(() => {
    let renderer;
    Object.defineProperty(window, 'FaultScene', {
      configurable: true,
      get: () => renderer,
      set: value => {
        renderer = { ...value, draw(...args) {
          if (window.portfolioTestFaultFailure) throw new Error('PORTFOLIO_TEST_FAULT_FAILURE');
          return value.draw(...args);
        } };
      },
    });
  });
  const failedStoryPage = await failedStoryContext.newPage();
  const storyWarnings = [];
  failedStoryPage.on('pageerror', error => errors.push(error.message));
  failedStoryPage.on('console', message => {
    if (message.type() === 'warning' && message.text().includes('PORTFOLIO_TEST_FAULT_FAILURE')) storyWarnings.push(message.text());
  });
  await failedStoryPage.goto('http://portfolio.test/');
  await waitForPaint(failedStoryPage.locator('#fault-story .motion-scene'));
  assert.equal(await failedStoryPage.locator('#fault-story .scene-controls').isVisible(), true);
  await failedStoryPage.evaluate(() => { window.portfolioTestFaultFailure = true; });
  await failedStoryPage.waitForFunction(() => !document.querySelector('#fault-story .motion-scene').classList.contains('is-ready'));
  await assertProjectFallback(failedStoryPage, 'Late project renderer failure', projectStories.filter(project => project.type === 'fault'));
  await failedStoryPage.waitForTimeout(200);
  assert.equal(storyWarnings.length, 1, 'A project draw failure should be reported once and disable its controls');
  for (const project of projectStories.filter(project => project.type !== 'fault')) {
    await waitForPaint(failedStoryPage.locator(`#${project.id} .motion-scene`));
    assert.equal(await failedStoryPage.locator(`#${project.id} .scene-controls`).isVisible(), true, 'A failed project renderer must not disable its healthy neighbors');
  }
  await waitForPaint(failedStoryPage.locator('#heroField'));
  await failedStoryContext.close();

  // Loading the optional power artwork must change the grid-mode illustration;
  // losing that asset should restore the original field without breaking AI modes.
  const noJourneyContext = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  await serveLocalFiles(noJourneyContext, { omitPowerJourney: true });
  const noJourneyPage = await noJourneyContext.newPage();
  noJourneyPage.on('pageerror', error => errors.push(error.message));
  await noJourneyPage.goto('http://portfolio.test/');
  const fallbackField = noJourneyPage.locator('#heroField');
  await waitForPaint(fallbackField);
  assert.equal(await noJourneyPage.evaluate(() => typeof window.PowerJourney), 'undefined');
  const fallbackFieldFrame = await frame(fallbackField);
  assert.ok(reducedPowerJourneyFrame, 'The normal power journey must be captured before testing its fallback');
  assert.notEqual(fallbackFieldFrame, reducedPowerJourneyFrame, 'The loaded power renderer should replace the original abstract grid field');
  assert.equal(await noJourneyPage.locator('[data-power-legend]').isVisible(), false, 'A fallback field must not advertise absent power-stage objects');
  assert.equal(await noJourneyPage.locator('[data-power-sources]').isVisible(), false, 'A fallback field must not advertise missing generation or storage artwork');
  assert.equal(await noJourneyPage.locator('[data-power-replay]').isVisible(), false, 'A fallback field must not offer an unavailable event');
  assert.equal(await noJourneyPage.locator('[data-field-label]').isVisible(), true);
  assert.notEqual(await noJourneyPage.locator('[data-field-description]').textContent(), powerDescription, 'The fallback should not advertise absent load artwork');
  await noJourneyPage.waitForTimeout(150);
  assert.equal(await frame(fallbackField), fallbackFieldFrame, 'Fallback artwork must still respect reduced motion');
  for (const mode of ['knowledge', 'learning']) {
    await noJourneyPage.locator(`[data-field-mode="${mode}"]`).click();
    assert.equal(await fallbackField.getAttribute('data-field-state'), mode);
    assert.notEqual(await frame(fallbackField), fallbackFieldFrame, 'Missing power artwork must not disable AI illustrations');
    assert.equal(await noJourneyPage.locator('[data-power-legend]').isVisible(), false);
  }
  await noJourneyContext.close();

  // A renderer can also fail after it has already painted successfully. This
  // optional artwork must not take the healthy AI illustrations down with it.
  const failedJourneyContext = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: 'no-preference' });
  await serveLocalFiles(failedJourneyContext);
  const failedJourneyPage = await failedJourneyContext.newPage();
  const journeyWarnings = [];
  failedJourneyPage.on('pageerror', error => errors.push(error.message));
  failedJourneyPage.on('console', message => {
    if (message.type() === 'warning' && message.text().includes('PORTFOLIO_TEST_POWER_FAILURE')) journeyWarnings.push(message.text());
  });
  await failedJourneyPage.goto('http://portfolio.test/');
  const recoveringField = failedJourneyPage.locator('#heroField');
  await waitForPaint(recoveringField);
  assert.equal(await failedJourneyPage.locator('[data-power-legend]').isVisible(), true, 'The real journey must paint before the injected late failure');
  await failedJourneyPage.evaluate(() => {
    window.PowerJourney = Object.freeze({ draw() { throw new Error('PORTFOLIO_TEST_POWER_FAILURE'); } });
  });
  await failedJourneyPage.waitForFunction(() => document.querySelector('#heroField').classList.contains('is-ready') && document.querySelector('[data-power-legend]').hidden);
  assert.equal(await failedJourneyPage.locator('[data-field-label]').isVisible(), true);
  assert.equal(await failedJourneyPage.locator('[data-power-sources]').isVisible(), false, 'A failed journey must hide its generation/storage annotations');
  assert.equal(await failedJourneyPage.locator('[data-power-replay]').isVisible(), false, 'A failed journey must hide its event control');
  assert.notEqual(await failedJourneyPage.locator('[data-field-description]').textContent(), powerDescription, 'A failed power renderer must also replace its load caption');
  assert.equal(await failedJourneyPage.locator('.field-modes').isVisible(), true, 'An optional renderer failure must preserve working hero controls');
  const recoveredFrame = await frame(recoveringField);
  await failedJourneyPage.waitForTimeout(200);
  assert.notEqual(await frame(recoveringField), recoveredFrame, 'The original field should continue animating after journey failure');
  await failedJourneyPage.emulateMedia({ reducedMotion: 'reduce' });
  const recoveredAiFrames = [];
  for (const mode of ['knowledge', 'learning']) {
    await failedJourneyPage.locator(`[data-field-mode="${mode}"]`).click();
    assert.equal(await recoveringField.getAttribute('data-field-state'), mode);
    assert.equal(await failedJourneyPage.locator(`[data-field-mode="${mode}"]`).getAttribute('aria-pressed'), 'true');
    assert.equal(await recoveringField.evaluate(scene => scene.classList.contains('is-ready')), true);
    assert.equal(await failedJourneyPage.locator('[data-power-legend]').isVisible(), false);
    recoveredAiFrames.push(await frame(recoveringField));
  }
  assert.equal(new Set(recoveredAiFrames).size, 2, 'Both AI illustrations should still render their distinct content after journey failure');
  assert.equal(journeyWarnings.length, 1, 'An optional renderer failure should be reported once, not silently swallowed or logged every frame');
  await failedJourneyContext.close();

  // A preference change must clear an already painted flash, even when the
  // visibility scheduler has stopped because the hero is offscreen.
  const flashPreferenceContext = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: 'no-preference' });
  await serveLocalFiles(flashPreferenceContext);
  const flashPreferencePage = await flashPreferenceContext.newPage();
  flashPreferencePage.on('pageerror', error => errors.push(error.message));
  await flashPreferencePage.goto('http://portfolio.test/');
  const flashField = flashPreferencePage.locator('#heroField');
  await waitForPaint(flashField);
  await flashPreferencePage.evaluate(() => {
    const actual = window.PowerJourney;
    window.portfolioTestForcedEventTime = 8.15;
    window.PowerJourney = Object.freeze({ ...actual,
    getEventState(options = {}) { return actual.getEventState({ ...options, time: window.portfolioTestForcedEventTime }); },
    draw(ctx, options) {
      const parameters = { ...options, eventTime: window.portfolioTestForcedEventTime };
      actual.draw(ctx, parameters);
      const event = actual.getEventState({ time: parameters.eventTime, reducedMotion: parameters.reducedMotion || parameters.suppressFlash });
      window.portfolioTestStrike = {
        reduced: parameters.reducedMotion,
        suppressed: parameters.suppressFlash,
        stage: event.stage,
        opacity: event.strikeOpacity,
      };
    } });
  });
  await flashPreferencePage.waitForFunction(() => window.portfolioTestStrike?.opacity > 0);
  for (let toggle = 0; toggle < 2; toggle += 1) {
    await flashPreferencePage.locator('[data-motion-toggle]').click();
    assert.equal(await flashPreferencePage.locator('html').getAttribute('data-motion'), 'paused');
    assert.equal(await flashPreferencePage.evaluate(() => window.portfolioTestStrike.opacity), 0, 'Pause should clear the current lightning effect');
    await flashPreferencePage.locator('[data-motion-toggle]').click();
    assert.equal(await flashPreferencePage.locator('html').getAttribute('data-motion'), 'running');
    assert.equal(await flashPreferencePage.evaluate(() => window.portfolioTestStrike.suppressed), true, 'Resuming within the same strike must retain the flash-suppression latch');
    assert.equal(await flashPreferencePage.evaluate(() => window.portfolioTestStrike.opacity), 0, 'Rapid Pause/Play must not replay the same bright frame');
  }
  await flashPreferencePage.evaluate(() => { window.portfolioTestForcedEventTime = 9.1; });
  await flashPreferencePage.waitForFunction(() => window.portfolioTestStrike?.stage === 'isolated' && window.portfolioTestStrike.suppressed === false);
  await flashPreferencePage.evaluate(() => { window.portfolioTestForcedEventTime = 8.15; });
  await flashPreferencePage.waitForFunction(() => window.portfolioTestStrike?.opacity > 0);
  await flashPreferencePage.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await flashPreferencePage.waitForTimeout(150);
  const visibleStrikeFrame = await frame(flashField);
  await flashPreferencePage.emulateMedia({ reducedMotion: 'reduce' });
  await flashPreferencePage.waitForFunction(() => window.portfolioTestStrike?.reduced === true && window.portfolioTestStrike.opacity === 0);
  const clearedStrikeFrame = await frame(flashField);
  assert.notEqual(clearedStrikeFrame, visibleStrikeFrame, 'Enabling reduced motion must repaint and remove an existing lightning effect');
  await flashPreferencePage.waitForTimeout(200);
  assert.equal(await frame(flashField), clearedStrikeFrame, 'The cleared reduced-motion frame must remain still');
  await flashPreferenceContext.close();

  // Regression for the previous orb: missing/stale component CSS allowed
  // a Retina canvas's bitmap dimensions to grow its parent on every resize.
  const missingCssContext = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 2, reducedMotion: 'no-preference' });
  await serveLocalFiles(missingCssContext, { omitMotionStyles: true });
  const missingCssPage = await missingCssContext.newPage();
  missingCssPage.on('pageerror', error => errors.push(error.message));
  await missingCssPage.goto('http://portfolio.test/');
  for (const width of [1280, 390, 1024]) {
    await missingCssPage.setViewportSize({ width, height: 900 });
    await missingCssPage.waitForTimeout(350);
    const dimensions = await missingCssPage.locator('.motion-scene').evaluateAll(scenes => ({
      documentHeight: document.documentElement.scrollHeight,
      scenes: scenes.map(scene => {
        const canvas = scene.querySelector('canvas');
        return { type: scene.dataset.scene, height: scene.getBoundingClientRect().height, width: canvas.width, bitmapHeight: canvas.height };
      }),
    }));
    assert.ok(dimensions.documentHeight < 50000, `${width}px: missing motion CSS must not inflate the document`);
    for (const scene of dimensions.scenes) {
      const maximumHeight = maximumSceneHeight(scene.type);
      assert.ok(scene.height <= maximumHeight, `${width}px: missing motion CSS must leave each scene bounded`);
      assert.ok(scene.width <= 2048 && scene.bitmapHeight <= 2048, `${width}px: canvas allocations must remain bounded without component CSS`);
    }
  }
  await missingCssContext.close();

  // Decorative rendering can be unavailable (for example, resource pressure
  // prevents a 2D context). The portfolio must still be readable and navigable.
  const noCanvasContext = await browser.newContext({ viewport: { width: 390, height: 844 } });
  await serveLocalFiles(noCanvasContext);
  await noCanvasContext.addInitScript(() => {
    const getContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, ...args) {
      if (type === '2d' && this.closest('.motion-scene')) return null;
      return getContext.call(this, type, ...args);
    };
  });
  const fallbackPage = await noCanvasContext.newPage();
  fallbackPage.on('pageerror', error => errors.push(error.message));
  await fallbackPage.goto('http://portfolio.test/');
  await fallbackPage.waitForFunction(() => document.querySelector('#ercotUpdatesList').getAttribute('aria-busy') === 'false');
  assert.equal(await fallbackPage.locator('.motion-scene.is-ready').count(), 0);
  assert.equal(await fallbackPage.locator('.field-modes').isVisible(), false, 'Unavailable canvas controls should not be offered when rendering cannot initialize');
  assert.equal(await fallbackPage.locator('[data-power-legend]').isVisible(), false);
  assert.equal(await fallbackPage.locator('[data-power-sources]').isVisible(), false);
  assert.equal(await fallbackPage.locator('[data-power-replay]').isVisible(), false);
  await assertProjectFallback(fallbackPage, 'Canvas context failure');
  assert.equal(await fallbackPage.locator('.hero-photo').isVisible(), true, 'Canvas failure must preserve the original portrait');
  assert.equal(await fallbackPage.locator('[data-project-card] .work-image img:visible').count(), 6, 'Canvas failure must preserve project screenshots');
  assert.ok(await fallbackPage.locator('[data-project-card] .work-image img').evaluateAll(images => images.every(image => Number(getComputedStyle(image).opacity) > 0)), 'Project screenshots must remain visibly painted without canvas contexts');
  await fallbackPage.locator('#menuToggle').click();
  assert.equal(await fallbackPage.locator('#navLinks').isVisible(), true, 'Navigation must work if rendering cannot initialize');
  await fallbackPage.keyboard.press('Escape');
  await fallbackPage.locator('[data-project-filter="engineering"]').click();
  assert.equal(await fallbackPage.locator('[data-project-card]:not([hidden])').count(), 2, 'Project filtering must work independently of decorative rendering');
  assert.equal(await fallbackPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await noCanvasContext.close();

  // Use a separate context because JavaScript is a context-level setting.
  const noJsContext = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 390, height: 844 } });
  await serveLocalFiles(noJsContext);
  const staticPage = await noJsContext.newPage();
  await staticPage.goto('http://portfolio.test/');
  assert.equal(await staticPage.locator('#navLinks').isVisible(), true);
  assert.equal(await staticPage.locator('.field-modes').isVisible(), false, 'Mode controls should be hidden without their interaction script');
  assert.equal(await staticPage.locator('[data-power-legend]').isVisible(), false);
  assert.equal(await staticPage.locator('[data-power-sources]').isVisible(), false);
  assert.equal(await staticPage.locator('[data-power-replay]').isVisible(), false);
  await assertProjectFallback(staticPage, 'JavaScript disabled');
  assert.equal(await staticPage.locator('#education article.experience-card:visible').count(), 3, 'Education must remain available without JavaScript');
  await staticPage.locator('a[href="#education"]').first().click();
  assert.equal(new URL(staticPage.url()).hash, '#education', 'Education navigation must use a native fragment link');
  assert.equal(await staticPage.locator('[data-project-card]:visible').count(), 6);
  assert.equal(await staticPage.locator('.hero-photo').isVisible(), true, 'The original portrait must remain visible without scripts');
  assert.equal(await staticPage.locator('[data-project-card] .work-image img:visible').count(), 6, 'Project screenshots must remain available when animation cannot initialize');
  assert.equal(await staticPage.locator('.motion-scene.is-ready').count(), 0);
  assert.equal(await staticPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await noJsContext.close();
  assert.equal(errors.length, 0, errors.join('\n'));
  console.log('PASS: project search/category/reset/no-scroll; carousel controls, keyboard, inert slides and no autoplay; navigation; clipboard fallback; feed date/status/shape/error/URL safety.');
  console.log('PASS: responsive layouts 320–1440px; featured/filtered grids; 44px gallery targets; mobile navigation/search; keyboard focus; reduced motion; no-JavaScript fallback.');
  console.log('PASS: nine distinct scene types; keyboard hero and project-stage controls; global pause/resume and reduced motion across all renderers; offscreen/visibility pause; bounded 1x/2x canvas layout, including missing component CSS; context failure and screenshot fallbacks.');
  console.log('PASS: contingency, fault and forecast manual states redraw while paused; Auto and phase labels; 44px responsive project controls; missing project renderers/shared kit preserve content without legacy forecast substitution.');
  console.log('PASS: restored education content, degree status, native anchors and responsive/no-JavaScript access; power-stage legend, six source types and five load labels; missing/late-failing optional renderer fallback with AI modes preserved and one failure warning.');
  console.log('PASS: power-event replay, static outage/restoration while paused or reduced, hidden unavailable event controls, same-strike Pause/Play flash suppression, and immediate reduced-motion flash removal while offscreen.');
  await browser.close();
})().catch((error) => { console.error(error); process.exit(1); });
