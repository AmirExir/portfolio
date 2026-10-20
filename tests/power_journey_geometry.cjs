// Canvas-command regression checks for decorative artwork, not grid calculations.
// A recording context keeps these checks deterministic and dependency-free.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const sandbox = vm.createContext({ window: {} });
const source = fs.readFileSync(path.resolve(__dirname, '../assets/js/power-journey.js'), 'utf8');
const sourceBuilders = ['nuclearPlant', 'gasPlant', 'hydroDam', 'windFarm', 'solarFarm', 'batteryStorage'];
// White-box instrumentation is inserted only into this VM copy; tests observe
// the actual builders used by draw() without adding production-only exports.
const exportMarker = 'window.PowerJourney = Object.freeze(';
assert.equal(source.split(exportMarker).length, 2, 'Update test instrumentation if the renderer export changes');
const instrumentation = `
  const sourceBuildRecords = [];
  const networkBuildRecords = [];
  let activeTestLayout = null;
  const originalTestBuild = build;
  build = (...args) => {
    const [mobile, narrowPhone] = args;
    activeTestLayout = mobile ? (narrowPhone ? 'phone' : 'mobile') : 'desktop';
    try { return originalTestBuild(...args); } finally { activeTestLayout = null; }
  };
  const recordSourceBuild = (name, builder) => (group, ...args) => {
    const start = group.g.items.length;
    const routeStart = group.g.routes.length;
    const result = builder(group, ...args);
    sourceBuildRecords.push({ name, layout: activeTestLayout,
      terminals: Array.from(result, point => Array.from(point)),
      routes: group.g.routes.slice(routeStart),
      items: group.g.items.slice(start).map(item => ({ kind: item.kind, points: item.points.map(point => Array.from(point)) })) });
    return result;
  };
  ${sourceBuilders.map(name => `${name} = recordSourceBuild('${name}', ${name});`).join('\n  ')}
  const recordNetworkBuild = (name, builder) => (group, ...args) => {
    const result = builder(group, ...args);
    networkBuildRecords.push({ name, layout: activeTestLayout, terminals: result });
    return result;
  };
  ${['distributionPole', 'serviceTransformer', 'tower'].map(name => `${name} = recordNetworkBuild('${name}', ${name});`).join('\n  ')}
  window.powerJourneyTestInternals = { sourceBuildRecords, networkBuildRecords, storageState, drawRoutes, routePower, drawDetails, drawBreakers, drawContactDetails, drawLightning, models };
`;
vm.runInContext(source.replace(exportMarker, `${instrumentation}\n${exportMarker}`), sandbox, { filename: 'power-journey.js' });
const renderer = sandbox.window.PowerJourney;
assert.equal(typeof renderer?.draw, 'function');

function recordingContext({ throwOn } = {}) {
  const operations = [];
  const paints = [];
  const initialState = { globalAlpha: 1, globalCompositeOperation: 'source-over', lineWidth: 1 };
  let state = { ...initialState };
  const stack = [];
  let gradientIndex = 0;
  let injected = false;
  const record = (method, args) => {
    for (const value of args) {
      if (typeof value === 'number') assert.ok(Number.isFinite(value), `${method} received a nonfinite coordinate`);
    }
    operations.push([method, ...args]);
    if (['fill', 'stroke', 'fillRect', 'strokeRect'].includes(method)) paints.push({ method, alpha: state.globalAlpha });
    if (method === throwOn && !injected) {
      injected = true;
      throw new Error('INJECTED_CANVAS_FAILURE');
    }
  };
  const methods = {};
  for (const method of ['beginPath', 'closePath', 'moveTo', 'lineTo', 'bezierCurveTo', 'quadraticCurveTo', 'arc', 'ellipse', 'rect', 'stroke', 'fill', 'fillRect', 'strokeRect', 'clearRect', 'translate', 'scale', 'rotate', 'transform', 'setTransform', 'clip', 'fillText', 'strokeText', 'setLineDash']) {
    methods[method] = (...args) => record(method, args);
  }
  methods.save = () => { record('save', []); stack.push({ ...state }); };
  methods.restore = () => {
    record('restore', []);
    assert.ok(stack.length, 'Canvas state must not be restored without a matching save');
    state = stack.pop();
  };
  for (const method of ['createLinearGradient', 'createRadialGradient']) {
    methods[method] = (...args) => {
      record(method, args);
      const id = `gradient-${gradientIndex++}`;
      return { id, addColorStop: (offset, color) => record('colorStop', [id, offset, color]) };
    };
  }
  methods.measureText = text => ({ width: String(text).length * 6 });
  const context = new Proxy({}, {
    get: (_, name) => Object.hasOwn(methods, name) ? methods[name] : state[name],
    set: (_, name, value) => {
      if (name === 'globalAlpha') assert.ok(Number.isFinite(value) && value >= 0 && value <= 1, 'Drawing opacity must remain valid');
      if (name === 'lineWidth') assert.ok(Number.isFinite(value) && value >= 0, 'Drawing width must remain valid');
      state[name] = value;
      record(`set:${String(name)}`, [value?.id || value]);
      return true;
    },
  });
  return {
    context, operations, paints,
    assertRestored() {
      assert.equal(stack.length, 0, 'The renderer must balance all canvas saves');
      assert.deepEqual(state, initialState, 'The renderer must restore caller-owned drawing state');
    },
    digest() { return crypto.createHash('sha256').update(JSON.stringify(operations)).digest('hex'); },
  };
}

function trace(options) {
  const recorder = recordingContext();
  renderer.draw(recorder.context, options);
  recorder.assertRestored();
  assert.ok(recorder.operations.filter(operation => operation[0] === 'lineTo').length > 100, 'The power journey must paint actual equipment geometry');
  assert.deepEqual(recorder.operations.find(operation => operation[0] === 'clearRect'), ['clearRect', 0, 0, options.width, options.height]);
  return recorder;
}

for (const [width, height] of [[1440, 900], [640, 980], [390, 980], [320, 980]]) {
  const options = { width, height, time: 2, pointerX: 0, pointerY: 0 };
  const first = trace(options);
  assert.equal(trace(options).digest(), first.digest(), `${width}px: drawing the same frame must be deterministic`);
  assert.notEqual(trace({ ...options, time: 2.25 }).digest(), first.digest(), `${width}px: energy-flow artwork must advance with time`);
  for (const pointer of [-1, 1]) trace({ ...options, pointerX: pointer, pointerY: pointer });
}

function normalizedGeometryDigest(items) {
  const points = items.flatMap(item => Array.from(item.points));
  assert.ok(points.length > 20, 'A source must construct equipment geometry, not just a label');
  const minimum = [0, 1, 2].map(axis => Math.min(...points.map(point => point[axis])));
  const maximum = [0, 1, 2].map(axis => Math.max(...points.map(point => point[axis])));
  const normalized = Array.from(items, item => [item.kind, Array.from(item.points, point => Array.from(point, (value, axis) => {
    assert.ok(Number.isFinite(value), 'Equipment geometry must contain finite world coordinates');
    const extent = maximum[axis] - minimum[axis];
    return Number((extent ? (value - minimum[axis]) / extent : 0).toFixed(5));
  }))]);
  return crypto.createHash('sha256').update(JSON.stringify(normalized)).digest('hex');
}

const internals = sandbox.window.powerJourneyTestInternals;
for (const [time, mode, direction] of [[0, 'discharge', 1], [8.199, 'discharge', 1], [8.2, 'idle', 0], [9.199, 'idle', 0], [9.2, 'charge', -1], [17.399, 'charge', -1], [17.4, 'idle', 0], [18.399, 'idle', 0], [18.4, 'discharge', 1]]) {
  const state = internals.storageState(time);
  assert.equal(state.mode, mode, `Storage mode at ${time}s`);
  assert.equal(state.direction, direction, `Storage flow direction at ${time}s`);
  assert.ok(Number.isFinite(state.level) && state.level >= 0 && state.level <= 1, 'The illustrative storage-level indicator must remain bounded');
}
assert.ok(internals.storageState(7).level < internals.storageState(1).level, 'The storage indicator must decrease during discharge');
assert.ok(internals.storageState(16).level > internals.storageState(10).level, 'The storage indicator must increase during charging');
for (const time of [-1, NaN, Infinity, -Infinity]) {
  assert.equal(internals.storageState(time).mode, 'discharge', 'Invalid storage times should normalize to the first phase');
  assert.equal(internals.storageState(time).level, internals.storageState(0).level);
}

for (const layout of ['desktop', 'mobile', 'phone']) {
  const records = Array.from(internals.sourceBuildRecords).filter(record => record.layout === layout);
  assert.deepEqual([...new Set(records.map(record => record.name))].sort(), [...sourceBuilders].sort(), `${layout}: draw() must actually construct all six source/storage technologies`);
  for (const record of records) {
    assert.equal(record.terminals.length, 3, `${layout}/${record.name}: the source should provide three distinct AC terminals`);
    assert.equal(new Set(Array.from(record.terminals, terminal => JSON.stringify(terminal))).size, 3, `${layout}/${record.name}: phase terminals must not collapse to one point`);
  }
  const shapes = sourceBuilders.map(name => normalizedGeometryDigest(records.find(record => record.name === name).items));
  assert.equal(new Set(shapes).size, sourceBuilders.length, `${layout}: all six equipment shapes must differ beyond translation, scale, color, or captions`);
  for (const name of ['solarFarm', 'batteryStorage']) {
    const routes = records.find(record => record.name === name).routes;
    assert.equal(routes.length, 2, `${layout}/${name}: DC connections should use separate positive/negative routes into the converter`);
    assert.notEqual(JSON.stringify(routes[0].points), JSON.stringify(routes[1].points), `${layout}/${name}: the DC pair must not collapse into one route`);
    assert.equal(routes.every(route => Boolean(route.storage) === (name === 'batteryStorage')), true, `${layout}/${name}: only storage connections should reverse with the battery state`);
  }
  const model = internals.models.get(layout);
  assert.equal(model.routes.filter(route => route.storage).length, 5, `${layout}: battery direction must control both DC routes and all three AC phases`);

  const network = Array.from(internals.networkBuildRecords).filter(record => record.layout === layout);
  const towers = network.filter(record => record.name === 'tower');
  assert.ok(towers.length >= 4, `${layout}: the two circuits must have distinct transmission structures`);
  assert.equal(new Set(towers.map(record => JSON.stringify(record.terminals))).size, towers.length, `${layout}: transmission structures must not overlap at identical coordinates`);
  const poles = network.filter(record => record.name === 'distributionPole');
  assert.ok(poles.length >= 2, `${layout}: distribution should include a visible overhead feeder span`);
  const downstream = model.routes.filter(route => route.zone === 'downstream');
  const pointKey = point => JSON.stringify(Array.from(point));
  const downstreamEndpoints = new Set(downstream.flatMap(route => [pointKey(route.points[0]), pointKey(route.points.at(-1))]));
  for (const pole of poles) {
    assert.equal(pole.terminals.length, 3, `${layout}: each distribution pole needs three phase terminals`);
    assert.equal(new Set(Array.from(pole.terminals, pointKey)).size, 3, `${layout}: distribution phases must not collapse into a shared terminal`);
    for (const terminal of pole.terminals) {
      assert.ok(downstreamEndpoints.has(pointKey(terminal)), `${layout}: each overhead phase must connect to a downstream conductor`);
    }
  }
  const services = network.filter(record => record.name === 'serviceTransformer');
  assert.ok(services.length > 0, `${layout}: distribution feeders need service transformation before the load branches`);
  for (const { terminals: service } of services) {
    for (const winding of ['primary', 'secondary']) {
      assert.equal(service[winding].length, 3, `${layout}: service transformer ${winding} needs distinct phase terminals`);
      assert.equal(new Set(Array.from(service[winding], pointKey)).size, 3);
      assert.ok(service[winding].some(terminal => downstreamEndpoints.has(pointKey(terminal))), `${layout}: the service transformer ${winding} must actually connect to the downstream network`);
    }
    const primary = new Set(Array.from(service.primary, pointKey));
    const secondary = new Set(Array.from(service.secondary, pointKey));
    assert.ok([...primary].every(terminal => !secondary.has(terminal)), `${layout}: primary and secondary terminals must not coincide`);
    // A transformer exchanges energy through its windings, never through a
    // wire drawn directly across the primary and secondary voltage stages.
    assert.ok(downstream.every(route => {
      const endpoints = [pointKey(route.points[0]), pointKey(route.points.at(-1))];
      return !(endpoints.some(terminal => primary.has(terminal)) && endpoints.some(terminal => secondary.has(terminal)));
    }), `${layout}: service transformer windings must not be bypassed by a conductor`);
  }
}

function routeFrame(time, storage = true, actualRoute) {
  const recorder = recordingContext();
  const route = actualRoute || { points: [[0, 0, 0], [10, 0, 0]], phase: 0, start: 0, duration: 1, color: '#ffffff', storage };
  internals.drawRoutes(recorder.context, { routes: [route] }, { cx: 0, ground: 0, scale: 1, depthX: 0, depthY: 0, mobile: false }, time);
  return {
    head: recorder.operations.find(operation => operation[0] === 'arc')?.[1],
    tail: recorder.operations.find(operation => operation[0] === 'moveTo')?.[1],
    arcs: recorder.operations.filter(operation => operation[0] === 'arc').length,
  };
}
const discharge = routeFrame(1);
const charge = routeFrame(10.2);
assert.ok(routeFrame(1.2).head > discharge.head, 'Discharge packets must travel from storage toward the grid');
assert.ok(routeFrame(10.4).head < charge.head, 'Charging packets must reverse toward storage');
assert.ok(discharge.tail < discharge.head, 'Discharge light trails must follow behind their packet');
assert.ok(charge.tail > charge.head, 'Charge light trails must reverse with their packet');
assert.equal(routeFrame(8.7).arcs, 0, 'Storage idle must stop its transfer packets');
assert.ok(routeFrame(8.7, false).arcs > 0, 'Storage idle must not stop ordinary grid-transfer artwork');

function firstTransferPulse(route, from, to) {
  for (let step = 0; step <= Math.round((to - from) / 0.025); step += 1) {
    const time = from + step * 0.025;
    if (routeFrame(time, true, route).arcs) return time;
  }
  assert.fail(`A storage connection did not animate in its ${from}–${to}s transfer window`);
}
for (const layout of ['desktop', 'mobile', 'phone']) {
  const battery = Array.from(internals.sourceBuildRecords).find(record => record.layout === layout && record.name === 'batteryStorage');
  const model = internals.models.get(layout);
  const dc = battery.routes.find(route => route.phase === 0);
  const ac = model.routes.find(route => route.storage && route.phase === 0 && !battery.routes.includes(route));
  assert.ok(ac, `${layout}: storage needs a converter-to-collector connection`);
  // Observe rendered pulses on real model segments; do not duplicate the
  // renderer's mirrored scheduling formula in the test's expected result.
  assert.ok(firstTransferPulse(dc, 1, 3) < firstTransferPulse(ac, 1, 3), `${layout}: discharge must leave the DC storage side before entering the AC collector`);
  assert.ok(firstTransferPulse(ac, 10, 12) < firstTransferPulse(dc, 10, 12), `${layout}: charging must enter from the AC collector before reaching DC storage`);
}

assert.equal(renderer.cycleDuration, 26);
assert.deepEqual({ ...renderer.eventCues }, { replay: 7.6, isolated: 10, restored: 19 });
for (const [time, stage] of [[0, 'normal'], [7.999, 'normal'], [8, 'strike'], [8.299, 'strike'], [8.3, 'tripping'], [9.099, 'tripping'], [9.1, 'isolated'], [13.799, 'isolated'], [13.8, 'reclosing'], [15.199, 'reclosing'], [15.2, 'restoring'], [18.499, 'restoring'], [18.5, 'restored'], [25.999, 'restored'], [26, 'normal']]) {
  const event = renderer.getEventState({ time });
  assert.equal(event.stage, stage, `Power-event stage at ${time}s`);
  assert.equal(event.label, renderer.labels[stage]);
  assert.equal(event.cycleDuration, 26);
  assert.equal(event.canReplay, time % 26 < 7.6 || time % 26 >= 18.5, 'Replay should remain locked during an active sequence');
  assert.equal(renderer.getEventState({ time, reducedMotion: true }).strikeOpacity, 0, 'Reduced motion must never produce a bolt');
}
for (const time of [-1, NaN, Infinity, -Infinity]) assert.equal(renderer.getEventState({ time }).stage, 'normal');
assert.equal(renderer.getEventState().stage, 'normal');
assert.equal(renderer.getEventState({ time: 7.6 }).canReplay, false);
const normalEvent = renderer.getEventState({ time: 0 });
const isolatedEvent = renderer.getEventState({ time: 10 });
const sendingClosedEvent = renderer.getEventState({ time: 14.5 });
assert.equal(isolatedEvent.sendingOpen, 1);
assert.equal(isolatedEvent.receivingOpen, 1);
assert.equal(isolatedEvent.linePower, 0, 'Both primary terminals must isolate the faulted circuit');
assert.equal(isolatedEvent.alternatePower, 1, 'The second circuit must remain energized');
assert.equal(isolatedEvent.downstreamPower, 1, 'The healthy circuit must continue supplying the distribution network');
assert.equal(sendingClosedEvent.sendingOpen, 0);
assert.equal(sendingClosedEvent.receivingOpen, 1);
assert.equal(sendingClosedEvent.linePower, 1);
assert.equal(sendingClosedEvent.downstreamPower, 1, 'Reclosing the primary circuit must not interrupt the alternate supply');
for (const state of ['base', 'open', 'redistributed']) {
  const initial = renderer.getEventState({ time: 0, state });
  for (const time of [2, 8.15, 10, 14.5, 19, 26]) {
    const event = renderer.getEventState({ time, state });
    assert.equal(event.stage, state === 'base' ? 'normal' : 'isolated', `${state}: manual switching state must not advance with time`);
    for (const field of ['sendingOpen', 'receivingOpen', 'linePower', 'alternatePower', 'downstreamPower', 'redistributionGain']) {
      assert.equal(event[field], initial[field], `${state}: manual ${field} must remain fixed`);
    }
    assert.equal(event.strikeOpacity, 0, 'Manual state inspection must not flash lightning');
  }
  assert.equal(initial.sendingOpen, state === 'base' ? 0 : 1);
  assert.equal(initial.receivingOpen, state === 'base' ? 0 : 1);
  assert.equal(initial.linePower, state === 'base' ? 1 : 0);
  assert.equal(initial.alternatePower, 1);
  assert.equal(initial.downstreamPower, 1);
  assert.equal(initial.redistributionGain, state === 'redistributed' ? 1 : 0, 'Manual redistribution changes illustrative emphasis, not supply continuity');
}
assert.deepEqual(
  { ...renderer.getEventState({ time: 10, state: 'invalid' }) },
  { ...isolatedEvent },
  'An invalid manual state should use the automatic event sequence',
);
let flashRuns = 0;
let previousFlash = false;
for (let sample = 0; sample <= 2600; sample += 1) {
  const event = renderer.getEventState({ time: sample / 100 });
  for (const field of ['sendingOpen', 'receivingOpen', 'linePower', 'alternatePower', 'downstreamPower', 'redistributionGain', 'restorationProgress']) assert.ok(Number.isFinite(event[field]) && event[field] >= 0 && event[field] <= 1, `${field} must remain bounded`);
  assert.ok(event.strikeOpacity >= 0 && event.strikeOpacity <= 0.85 + 1e-12, 'Lightning intensity must remain bounded');
  if (event.linePower > 0) assert.ok(event.sendingOpen <= 1e-12, 'The line must not energize before the sending contacts close');
  assert.equal(event.alternatePower, 1, 'The healthy parallel circuit remains energized throughout the illustrative sequence');
  assert.equal(event.downstreamPower, 1, 'The single-circuit contingency must not become a total load outage');
  const flashing = event.strikeOpacity > 1e-12;
  if (flashing && !previousFlash) flashRuns += 1;
  previousFlash = flashing;
}
assert.equal(flashRuns, 1, 'One event cycle should contain one restrained pulse, not repeated strobing');
assert.ok(renderer.getEventState({ time: 17 }).redistributionGain < renderer.getEventState({ time: 16 }).redistributionGain, 'Redistribution emphasis should recede as both circuits return to service');

const testView = { cx: 500, ground: 450, scale: 35, depthX: 0.43, depthY: 0.29, mobile: false };
const emptyDetails = { wind: [], water: [], storage: [], rotors: [], flux: [], lights: [] };
for (const layout of ['desktop', 'mobile', 'phone']) {
  const model = internals.models.get(layout);
  assert.equal(model.breakers.length, 12, `${layout}: both circuits need three breaker poles at both ends`);
  for (const circuit of ['primary', 'alternate']) {
    for (const role of ['sending', 'receiving']) {
      assert.deepEqual(Array.from(model.breakers.filter(breaker => breaker.circuit === circuit && breaker.role === role), breaker => breaker.phase).sort(), [0, 1, 2], `${layout}/${circuit}/${role}: phases must remain separate`);
    }
  }
  for (const zone of ['source', 'line', 'alternate', 'downstream']) assert.ok(model.routes.some(route => route.zone === zone), `${layout}: ${zone} routes must be represented`);
  for (const route of model.routes) {
    assert.ok(['source', 'line', 'alternate', 'downstream'].includes(route.zone));
    const primaryContact = route.breakerRole && route.circuit === 'primary';
    assert.equal(internals.routePower(route, isolatedEvent), route.zone === 'line' || primaryContact ? 0 : 1, `${layout}: only the isolated circuit and its open contacts should lose transfer packets`);
  }
  const isolatedRoutes = recordingContext();
  internals.drawRoutes(isolatedRoutes.context, { routes: model.routes.filter(route => route.zone === 'line' || (route.breakerRole && route.circuit === 'primary')) }, testView, 2, isolatedEvent);
  assert.equal(isolatedRoutes.paints.length, 0, 'The isolated primary circuit and its open contacts must show no transfer packets');
  for (const zone of ['source', 'alternate', 'downstream']) {
    const liveRoutes = recordingContext();
    internals.drawRoutes(liveRoutes.context, { routes: model.routes.filter(route => route.zone === zone) }, testView, 2, isolatedEvent);
    assert.ok(liveRoutes.paints.length > 0, `${layout}: ${zone} transfer artwork must remain active with the primary circuit isolated`);
  }
  const loadDetails = { ...emptyDetails, lights: model.lights, flux: model.flux.filter(flux => flux.zone === 'downstream') };
  const energizedLoads = recordingContext();
  internals.drawDetails(energizedLoads.context, loadDetails, testView, 2, normalEvent);
  assert.ok(energizedLoads.paints.length > 0, 'Normal operation should paint load windows and receiving-transformer effects');
  const suppliedLoads = recordingContext();
  internals.drawDetails(suppliedLoads.context, loadDetails, testView, 2, isolatedEvent);
  assert.equal(suppliedLoads.digest(), energizedLoads.digest(), 'Load windows and receiving-transformer effects must retain supply through the healthy circuit');
  const sourceDetails = { ...model, lights: [], flux: model.flux.filter(flux => flux.zone === 'source') };
  const liveSources = recordingContext(), isolatedSources = recordingContext(), laterSources = recordingContext();
  internals.drawDetails(liveSources.context, sourceDetails, testView, 2, normalEvent);
  internals.drawDetails(isolatedSources.context, sourceDetails, testView, 2, isolatedEvent);
  internals.drawDetails(laterSources.context, sourceDetails, testView, 2.25, isolatedEvent);
  assert.equal(liveSources.digest(), isolatedSources.digest(), 'A single-circuit outage must not disable upstream generation/storage visuals');
  assert.notEqual(isolatedSources.digest(), laterSources.digest(), 'Source-side rotors, water and storage artwork must continue animating');
  for (const breaker of model.breakers) {
    const gap = event => {
      const recorder = recordingContext();
      internals.drawBreakers(recorder.context, { breakers: [breaker] }, testView, event);
      const moves = recorder.operations.filter(operation => operation[0] === 'moveTo');
      const lines = recorder.operations.filter(operation => operation[0] === 'lineTo');
      return Math.hypot(lines[0][1] - moves[1][1], lines[0][2] - moves[1][2]);
    };
    assert.equal(gap(normalEvent), 0, 'Closed contacts must meet in the rendered chamber');
    if (breaker.circuit === 'primary') {
      assert.ok(gap(isolatedEvent) > 1, 'Both ends of the primary circuit must show a visible contact gap');
    } else {
      assert.equal(gap(isolatedEvent), 0, 'Both ends of the healthy circuit must remain visibly closed');
    }
    assert.equal(gap(sendingClosedEvent) > 1, breaker.circuit === 'primary' && breaker.role === 'receiving', 'Only the primary receiving contacts should remain open during ordered restoration');
  }
}

// Exercise the actual base-face paint loop with only emissive items retained.
const desktopModel = internals.models.get('desktop');
assert.ok(desktopModel.items.some(item => item.emission), 'Load windows need explicitly gated base emission');
for (const [width, height, layout] of [[1240, 490, 'desktop'], [640, 560, 'mobile'], [390, 560, 'phone'], [320, 560, 'phone']]) {
  const model = internals.models.get(layout);
  const closed = recordingContext(), opened = recordingContext(), resetting = recordingContext();
  internals.drawContactDetails(closed.context, model, width, height, normalEvent);
  internals.drawContactDetails(opened.context, model, width, height, isolatedEvent);
  internals.drawContactDetails(resetting.context, model, width, height, sendingClosedEvent);
  closed.assertRestored(); opened.assertRestored(); resetting.assertRestored();
  const labels = recorder => recorder.operations.filter(operation => operation[0] === 'fillText').map(operation => operation[1]);
  assert.equal(labels(closed).filter(label => label === 'A CLOSED').length, 2, `${width}px: both primary terminals must be shown closed in the contact detail`);
  assert.equal(labels(opened).filter(label => label === 'A OPEN').length, 2, `${width}px: both primary terminals must be shown open in the contact detail`);
  assert.equal(labels(opened).filter(label => label === 'B CLOSED').length, 2, `${width}px: both healthy terminals must remain closed in the contact detail`);
  assert.equal(labels(resetting).filter(label => label === 'A OPEN').length, 1, `${width}px: reset detail must preserve the primary receiving-end gap`);
  const gaps = recorder => {
    const moves = recorder.operations.filter(operation => operation[0] === 'moveTo');
    const lines = recorder.operations.filter(operation => operation[0] === 'lineTo');
    // Each enlarged contact contributes two contact segments and two end caps.
    assert.equal(moves.length, 16, 'The two terminal panels should each enlarge one pole of both circuits');
    return [0, 4, 8, 12].map(index => Math.hypot(lines[index][1] - moves[index + 1][1], lines[index][2] - moves[index + 1][2]));
  };
  assert.deepEqual(gaps(closed), [0, 0, 0, 0], `${width}px: the actual enlarged closed contacts must touch`);
  const openGaps = gaps(opened);
  assert.ok(openGaps[0] >= 20 && openGaps[2] >= 20, `${width}px: enlarged primary gaps must be plainly visible`);
  assert.equal(openGaps[1], 0); assert.equal(openGaps[3], 0);
  for (const operation of opened.operations.filter(operation => ['moveTo', 'lineTo'].includes(operation[0]))) {
    assert.ok(operation[1] >= 0 && operation[1] <= width && operation[2] >= 0 && operation[2] <= height, `${width}px: enlarged contacts must remain inside the story canvas`);
  }
  const options = { width, height, presentation: 'story', state: 'open', time: 2, reducedMotion: true };
  const first = trace(options);
  assert.equal(trace({ ...options, time: 19 }).digest(), first.digest(), `${width}px: reduced-motion manual story inspection must remain still`);
  assert.notEqual(trace({ ...options, state: 'base' }).digest(), first.digest(), `${width}px: the shared story must visibly distinguish closed and isolated circuits`);
  assert.notEqual(trace({ ...options, state: 'redistributed' }).digest(), first.digest(), `${width}px: the shared story must visibly distinguish open contacts from redistribution emphasis`);
}
try {
  internals.models.set('desktop', { ...desktopModel, ...emptyDetails, items: desktopModel.items.filter(item => item.emission), routes: [], breakers: [], strike: null });
  const normalFaces = trace({ width: 1440, height: 900, time: 2, eventTime: 0 });
  const suppliedFaces = trace({ width: 1440, height: 900, time: 2, eventTime: 10 });
  assert.ok(suppliedFaces.paints.some(paint => paint.method === 'fill' && paint.alpha > 0), 'Window faces must remain illuminated through the alternate supply');
  assert.deepEqual(suppliedFaces.paints, normalFaces.paints, 'Base load-window emission must remain unchanged when one circuit opens');
} finally { internals.models.set('desktop', desktopModel); }
const boltEvent = renderer.getEventState({ time: 8.15 });
const bolt = recordingContext(), repeatedBolt = recordingContext(), quietBolt = recordingContext();
internals.drawLightning(bolt.context, desktopModel, testView, boltEvent);
internals.drawLightning(repeatedBolt.context, desktopModel, testView, boltEvent);
internals.drawLightning(quietBolt.context, desktopModel, testView, renderer.getEventState({ time: 8.15, reducedMotion: true }));
bolt.assertRestored(); repeatedBolt.assertRestored(); quietBolt.assertRestored();
assert.ok(bolt.paints.length > 0);
assert.equal(bolt.digest(), repeatedBolt.digest(), 'Strike geometry must be deterministic, without random flicker');
assert.equal(quietBolt.paints.length, 0, 'Reduced motion must paint no lightning');
assert.ok(bolt.operations.every(operation => !['fillRect', 'rect'].includes(operation[0])), 'The bolt must not flash the whole canvas');
const boltCoordinates = bolt.operations.filter(operation => ['moveTo', 'lineTo', 'arc'].includes(operation[0]));
for (const axis of [1, 2]) assert.ok(Math.max(...boltCoordinates.map(operation => operation[axis])) - Math.min(...boltCoordinates.map(operation => operation[axis])) <= 100, 'Lightning must remain localized around the strike point');
const eventOptions = { width: 390, height: 980, time: 2, eventTime: 8.15 };
assert.equal(trace({ ...eventOptions, suppressFlash: true }).digest(), trace({ ...eventOptions, eventTime: 0 }).digest(), 'Suppressing flash must preserve ambient motion and otherwise-powered equipment');
assert.equal(trace({ ...eventOptions, reducedMotion: true }).digest(), trace({ ...eventOptions, time: 24, reducedMotion: true }).digest(), 'Reduced motion must freeze ambient effects as well as suppress lightning');

for (const options of [{ width: 0, height: 900 }, { width: 390, height: -1 }, { width: NaN, height: 900 }, { width: 1440, height: Infinity }]) {
  const invalid = recordingContext();
  renderer.draw(invalid.context, options);
  assert.equal(invalid.operations.length, 0, 'Invalid canvas dimensions must not produce drawing commands');
  invalid.assertRestored();
}

const failed = recordingContext({ throwOn: 'lineTo' });
assert.throws(() => renderer.draw(failed.context, { width: 1440, height: 900, time: 2 }), /INJECTED_CANVAS_FAILURE/);
failed.assertRestored();
console.log('PASS: six genuinely distinct source/storage geometries in desktop/mobile/phone layouts; distinct AC terminals and DC pairs; storage cycle, bidirectional packets/trails, ordered AC/DC transfer, and idle behavior.');
console.log('PASS: power journey paints deterministic finite geometry at desktop/mobile widths, animates with time, handles pointer extremes, and restores canvas state even on failure.');
console.log('PASS: shared manual/automatic event states; both ends isolate only the primary circuit while alternate and downstream packets/windows/flux remain live; localized deterministic bolt, flash suppression and static reduced motion.');
console.log('PASS: connected three-phase distribution poles and separate service-transformer windings; visible enlarged contact gaps and distinct manual story frames through 320px, with reduced-motion stillness.');
