// Optional offline browser regression checks; setup is documented in docs/portfolio.md.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PORTFOLIO_PLAYWRIGHT_MODULE || 'playwright');

const root = path.resolve(__dirname, '..');
const realFeed = JSON.parse(fs.readFileSync(path.join(root, 'ERCOTAPI/latest_ercot_updates.json'), 'utf8'));
const sceneTypes = ['grid', 'atlas', 'rag', 'neural', 'forecast', 'workflow'];

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
    assert.ok(box.height <= 640, `${label}: ${box.scene} must not consume an unbounded vertical column`);
    assert.ok(Math.abs(box.height - before[index].height) <= 1, `${label}: canvas painting must not change ${box.scene} layout height`);
    assert.ok(box.canvasWidth <= box.width + 2 && box.canvasHeight <= box.height + 2, `${label}: ${box.scene} canvas must fit its scene`);
    if (!box.ready) continue; // Offscreen scenes may defer their initial allocation.
    assert.ok(box.backingWidth <= box.canvasWidth * 2 + 2, `${label}: ${box.scene} pixel width must be capped at 2x`);
    assert.ok(box.backingHeight <= box.canvasHeight * 2 + 2, `${label}: ${box.scene} pixel height must be capped at 2x`);
    assert.ok(box.backingWidth >= box.canvasWidth - 2 && box.backingHeight >= box.canvasHeight - 2, `${label}: ${box.scene} must paint at least at CSS resolution`);
  }
}

async function serveLocalFiles(context, { omitMotionStyles = false } = {}) {
  await context.route('**/*', (route) => {
    const url = new URL(route.request().url());
    if (url.hostname !== 'portfolio.test') return route.abort();
    if (omitMotionStyles && url.pathname.endsWith('/portfolio-motion.css')) return route.abort();
    const file = path.join(root, decodeURIComponent(url.pathname === '/' ? '/index.html' : url.pathname));
    return fs.existsSync(file) ? route.fulfill({ path: file }) : route.fulfill({ status: 404, body: '' });
  });
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
  const heroScene = page.locator('.hero-scene');
  const motionToggle = page.locator('[data-motion-toggle]').first();
  assert.ok(await page.locator('.motion-scene').count() >= 10, 'Motion should be distributed across the portfolio');
  const sceneFrames = [];
  for (const type of sceneTypes) {
    const scene = page.locator(`.motion-scene[data-scene="${type}"]`).first();
    await waitForPaint(scene);
    sceneFrames.push(await frame(scene));
  }
  assert.equal(new Set(sceneFrames).size, sceneTypes.length, 'Project areas should use distinct rendered scenes');
  await waitForPaint(heroScene);
  const movingFrame = await frame(heroScene);
  await page.waitForTimeout(250);
  assert.notEqual(
    await frame(heroScene), movingFrame,
    'The hero scene should animate when motion is allowed',
  );
  assert.equal(await page.locator('html').getAttribute('data-motion'), 'running');
  await motionToggle.click();
  assert.equal(await motionToggle.getAttribute('aria-pressed'), 'true');
  assert.match(await motionToggle.textContent(), /Play animations/i);
  assert.equal(await page.locator('html').getAttribute('data-motion'), 'paused');
  await waitForPaint(heroScene);
  await page.waitForTimeout(150);
  const manuallyPausedFrame = await frame(heroScene);
  await page.waitForTimeout(250);
  assert.equal(await frame(heroScene), manuallyPausedFrame, 'The global pause button should stop animation');
  await motionToggle.click();
  assert.equal(await motionToggle.getAttribute('aria-pressed'), 'false');
  assert.match(await motionToggle.textContent(), /Pause animations/i);
  await waitForPaint(heroScene);
  const resumedFrame = await frame(heroScene);
  await page.waitForTimeout(250);
  assert.notEqual(await frame(heroScene), resumedFrame, 'Play should resume animation');
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

  const visibleCards = () => page.locator('[data-project-card]:not([hidden])').count();
  assert.equal(await visibleCards(), 6);
  const originalCardSceneHeights = await page.locator('[data-project-card] .motion-scene').evaluateAll(scenes => scenes.map(scene => scene.clientHeight));
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
    await page.locator('[data-project-card] .motion-scene').evaluateAll(scenes => scenes.map(scene => scene.clientHeight)),
    originalCardSceneHeights,
    'Filtering and restoring hidden cards must preserve their original scene heights',
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
  for (const width of [1440, 1024, 820, 768, 700, 641, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await page.reload();
    await page.waitForFunction(() => document.documentElement.classList.contains('js'));
    assert.equal(await visibleCards(), 6);
    if (width === 1440) {
      await waitForPaint(heroScene);
      assert.equal(await page.locator('html').getAttribute('data-motion'), 'reduced');
      assert.equal(await motionToggle.isDisabled(), true, 'The system reduced-motion preference must take priority');
      const staticFrame = await frame(heroScene);
      await page.waitForTimeout(250);
      assert.equal(
        await frame(heroScene), staticFrame,
        'Reduced motion should retain a static painted scene',
      );
    }
    await assertBoundedScenes(page, `${width}px 1x`);
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
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `${width}px overflow for ${category}`);
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
        assert.ok(control.top >= gallery.trackTop - 1 && control.bottom <= gallery.trackBottom + 1, `${gallery.id} arrows must stay inside the image at ${width}px`);
        assert.ok(control.bottom <= gallery.dotsTop + 1, `${gallery.id} arrows must not overlap pagination at ${width}px`);
      }
    }
    if (width <= 820) {
      assert.ok(await page.locator('#searchInput').evaluate(input => parseFloat(getComputedStyle(input).fontSize) >= 16), 'Mobile search should not trigger iOS text zoom');
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
  for (const width of [1280, 390, 320]) {
    await retinaPage.setViewportSize({ width, height: 900 });
    for (const type of sceneTypes) await waitForPaint(retinaPage.locator(`.motion-scene[data-scene="${type}"]`).first());
    await assertBoundedScenes(retinaPage, `${width}px 2x`);
    assert.equal(await retinaPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `${width}px Retina layout must not overflow`);
  }
  await retinaContext.close();

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
        return { height: scene.getBoundingClientRect().height, width: canvas.width, bitmapHeight: canvas.height };
      }),
    }));
    assert.ok(dimensions.documentHeight < 50000, `${width}px: missing motion CSS must not inflate the document`);
    for (const scene of dimensions.scenes) {
      assert.ok(scene.height <= 360, `${width}px: missing motion CSS must leave each scene bounded`);
      assert.ok(scene.width <= 2048 && scene.bitmapHeight <= 2048, `${width}px: canvas allocations must remain bounded without component CSS`);
    }
  }
  await missingCssContext.close();

  // Use a separate context because JavaScript is a context-level setting.
  const noJsContext = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 390, height: 844 } });
  await serveLocalFiles(noJsContext);
  const staticPage = await noJsContext.newPage();
  await staticPage.goto('http://portfolio.test/');
  assert.equal(await staticPage.locator('#navLinks').isVisible(), true);
  assert.equal(await staticPage.locator('[data-project-card]:visible').count(), 6);
  assert.equal(await staticPage.locator('.hero-photo').isVisible(), true, 'The original portrait must remain visible without scripts');
  assert.equal(await staticPage.locator('[data-project-card] .work-image img:visible').count(), 6, 'Project screenshots must remain available when animation cannot initialize');
  assert.equal(await staticPage.locator('.motion-scene.is-ready').count(), 0);
  assert.equal(await staticPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await noJsContext.close();
  assert.equal(errors.length, 0, errors.join('\n'));
  console.log('PASS: project search/category/reset/no-scroll; carousel controls, keyboard, inert slides and no autoplay; navigation; clipboard fallback; feed date/status/shape/error/URL safety.');
  console.log('PASS: responsive layouts 320–1440px; featured/filtered grids; 44px gallery targets; mobile navigation/search; keyboard focus; reduced motion; no-JavaScript fallback.');
  console.log('PASS: six distinct animated scenes; global pause/resume; offscreen/visibility pause; bounded 1x/2x canvas layout, including missing component CSS; reduced motion and screenshot fallbacks.');
  await browser.close();
})().catch((error) => { console.error(error); process.exit(1); });
