// Pure illustration-state tests. These do not validate engineering calculations.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const sourceRoot = path.resolve(__dirname, '../assets/js');
const sandbox = vm.createContext({ window: {} });
function load(filename) {
  vm.runInContext(fs.readFileSync(path.join(sourceRoot, filename), 'utf8'), sandbox, { filename });
}
load('project-scene-kit.js');

const scenarios = [
  {
    file: 'contingency-scene.js', global: 'ContingencyScene', type: 'contingency',
    manual: ['base', 'open', 'redistributed'],
    boundaries: [[0, 'base'], [2.999, 'base'], [3, 'open'], [4.499, 'open'], [4.5, 'redistributed'], [8.999, 'redistributed'], [9, 'restoring'], [11.999, 'restoring'], [12, 'base'], [15, 'open']],
  },
  {
    file: 'fault-scene.js', global: 'FaultScene', type: 'fault',
    manual: ['signals', 'features', 'classified'],
    boundaries: [[0, 'signals'], [3.999, 'signals'], [4, 'features'], [7.999, 'features'], [8, 'classified'], [11.999, 'classified'], [12, 'signals'], [16, 'features']],
  },
  {
    file: 'forecast-scene.js', global: 'ForecastScene', type: 'forecast',
    manual: ['history', 'horizon', 'forecast'],
    boundaries: [[0, 'history'], [2.999, 'history'], [3, 'horizon'], [5.999, 'horizon'], [6, 'forecast'], [11.999, 'forecast'], [12, 'history'], [15, 'horizon']],
  },
];

for (const scenario of scenarios) {
  load(scenario.file);
  const renderer = sandbox.window[scenario.global];
  assert.ok(renderer, `${scenario.file} must expose ${scenario.global}`);
  assert.equal(typeof renderer.draw, 'function');
  assert.deepEqual(Array.from(renderer.types), [scenario.type]);
  assert.equal(renderer.getStage(), scenario.manual[0], `${scenario.type}: omitted options should select the initial stage`);
  for (const [time, expected] of scenario.boundaries) {
    assert.equal(renderer.getStage({ time, state: 'auto' }), expected, `${scenario.type}: Auto at ${time}s`);
    assert.equal(renderer.getStage({ time, state: 'not-a-stage' }), expected, `${scenario.type}: invalid state at ${time}s should fall back to Auto`);
    assert.equal(renderer.getStage({ time }), expected, `${scenario.type}: omitted state at ${time}s should use Auto`);
    assert.equal(typeof renderer.labels[expected], 'string', `${scenario.type}: ${expected} needs a readable phase label`);
    assert.ok(renderer.labels[expected].trim());
  }
  for (const state of scenario.manual) {
    for (const time of [0, 3, 6, 10, 19, 24]) {
      assert.equal(renderer.getStage({ time, state }), state, `${scenario.type}: manual ${state} should not advance with time`);
    }
  }
  for (const time of [-1, -12, NaN, Infinity, -Infinity]) {
    assert.equal(renderer.getStage({ time, state: 'auto' }), scenario.manual[0], `${scenario.type}: invalid time should normalize to zero`);
  }
  console.log(`PASS: ${scenario.type} deterministic stages, cycle boundaries, labels, manual selection, and invalid inputs.`);
}
