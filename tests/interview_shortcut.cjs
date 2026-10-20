// Dependency-free checks: node --test tests/interview_shortcut.cjs
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const source = readFileSync(path.join(root, 'assets/js/portfolio.js'), 'utf8');
const html = readFileSync(path.join(root, 'index.html'), 'utf8');
const css = readFileSync(path.join(root, 'assets/css/portfolio.css'), 'utf8');
const preferenceKey = 'portfolio.interview-shortcut';

function visit({ address = 'https://portfolio.test/?view=work', saved = new Map(), denied = [] } = {}) {
  const listeners = new Map();
  const clicks = new Map();
  const focus = [];
  const historyUpdates = [];
  const link = { focus: () => focus.push('shortcut') };
  const hide = { addEventListener: (name, handler) => clicks.set(name, handler) };
  const backToTop = { focus: () => focus.push('back-to-top') };
  const shortcut = {
    hidden: true,
    querySelector: selector => selector === 'a' ? link : hide,
  };
  const storage = {
    getItem(key) {
      if (denied.includes('read')) throw new Error('Storage read denied');
      return saved.get(key) ?? null;
    },
    setItem(key, value) {
      if (denied.includes('write')) throw new Error('Storage write denied');
      saved.set(key, value);
    },
    removeItem(key) {
      if (denied.includes('remove')) throw new Error('Storage removal denied');
      saved.delete(key);
    },
  };
  const window = {
    location: new URL(address),
    history: {
      state: { existing: 'state' },
      replaceState(state, title, destination) {
        historyUpdates.push({ state, destination });
        window.location = new URL(destination, window.location);
      },
    },
    get localStorage() {
      if (denied.includes('access')) throw new Error('Storage access denied');
      return storage;
    },
    addEventListener: (name, handler) => listeners.set(name, handler),
  };
  const document = {
    getElementById: id => id === 'interviewShortcut' ? shortcut : null,
    querySelector: selector => selector === '.site-footer a[href="#top"]' ? backToTop : null,
    querySelectorAll: () => [],
    documentElement: { classList: { add() {} } },
  };
  vm.runInNewContext(source, { window, document });
  return {
    shortcut, saved, focus, window, historyUpdates,
    hide: () => clicks.get('click')(),
    navigateHash(hash) {
      window.location.hash = hash;
      listeners.get('hashchange')();
    },
  };
}

test('markup hides the shortcut before JavaScript and keeps the public assistant separate', () => {
  assert.match(html, /<span\b[^>]*id="interviewShortcut"[^>]*\bhidden>/);
  assert.match(css, /\[hidden\]\s*\{\s*display:\s*none\s*!important;/);
  const footer = html.slice(html.indexOf('<footer'));
  assert.match(footer, /InterviewBot<span class="sr-only"> \(opens in a new tab\)<\/span>/);
  assert.match(footer, /<button type="button" data-hide-interview-shortcut aria-controls="interviewShortcut">Hide shortcut<\/button>/);
  assert.doesNotMatch(html.slice(0, html.indexOf('<footer')), /AI Interview Assistant|interview preparation|portfolio-rs4eamgdjpptbcpbeci6nu/);
  assert.equal((html.match(/portfolio-nvngbm55k5prquuhngg7el\.streamlit\.app\//g) || []).length, 2);
});

test('a new visitor sees no shortcut and unrelated fragments do not enable it', () => {
  const page = visit();
  assert.equal(page.shortcut.hidden, true);
  page.navigateHash('#contact');
  assert.equal(page.shortcut.hidden, true);
  assert.equal(page.window.location.hash, '#contact');
  assert.equal(page.saved.size, 0);
  assert.deepEqual(page.focus, []);
});

test('activation remembers the shortcut, removes only the fragment, and focuses its link', () => {
  const page = visit({ address: 'https://portfolio.test/portfolio/?view=work#my-interview-bot' });
  assert.equal(page.shortcut.hidden, false);
  assert.equal(page.saved.get(preferenceKey), 'enabled');
  assert.equal(page.window.location.href, 'https://portfolio.test/portfolio/?view=work');
  assert.deepEqual(page.historyUpdates, [{ state: { existing: 'state' }, destination: '/portfolio/?view=work' }]);
  assert.deepEqual(page.focus, ['shortcut']);
  const laterVisit = visit({ saved: page.saved });
  assert.equal(laterVisit.shortcut.hidden, false);
  assert.deepEqual(laterVisit.focus, []);
});

test('activation also works after the page has loaded', () => {
  const page = visit();
  page.navigateHash('#my-interview-bot');
  assert.equal(page.shortcut.hidden, false);
  assert.equal(page.window.location.hash, '');
  assert.equal(page.saved.get(preferenceKey), 'enabled');
});

test('hiding clears the preference and restores focus to a visible footer link', () => {
  const page = visit({ saved: new Map([[preferenceKey, 'enabled']]) });
  page.hide();
  assert.equal(page.shortcut.hidden, true);
  assert.equal(page.saved.has(preferenceKey), false);
  assert.deepEqual(page.focus, ['back-to-top']);
  assert.equal(visit({ saved: page.saved }).shortcut.hidden, true);
});

for (const denied of [['access'], ['read', 'write', 'remove']]) {
  test(`blocked storage (${denied.join(', ')}) preserves activation for this visit only`, () => {
    const page = visit({ denied });
    assert.equal(page.shortcut.hidden, true);
    page.navigateHash('#my-interview-bot');
    assert.equal(page.shortcut.hidden, false);
    assert.equal(page.window.location.hash, '');
    assert.equal(page.saved.size, 0);
    page.hide();
    assert.equal(page.shortcut.hidden, true);
    assert.equal(visit({ saved: page.saved, denied }).shortcut.hidden, true);
  });
}

test('failed preference removal still hides the current page', () => {
  const page = visit({ saved: new Map([[preferenceKey, 'enabled']]), denied: ['remove'] });
  page.hide();
  assert.equal(page.shortcut.hidden, true);
  assert.deepEqual(page.focus, ['back-to-top']);
});
