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
const sourceConnections = {
  nuclearPlant: { id: 'nuclear', circuit: 'primary', bay: 0 },
  hydroDam: { id: 'hydro', circuit: 'primary', bay: 1 },
  windFarm: { id: 'wind', circuit: 'primary', bay: 2 },
  gasPlant: { id: 'gas', circuit: 'alternate', bay: 0 },
  solarFarm: { id: 'solar', circuit: 'alternate', bay: 1 },
  batteryStorage: { id: 'battery', circuit: 'alternate', bay: 2 },
};
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
    networkBuildRecords.push({ name, layout: activeTestLayout, args, terminals: result });
    return result;
  };
  ${['distributionPole', 'serviceTransformer', 'tower', 'transformer', 'distributionCabinet', 'feederBreaker', 'sendingBus'].map(name => `${name} = recordNetworkBuild('${name}', ${name});`).join('\n  ')}
  window.powerJourneyTestInternals = { sourceBuildRecords, networkBuildRecords, storageState, drawRoutes, routePower, zonePower, drawDetails, drawBreakers, drawBreakerCues, drawAnnotations, projection, drawLightning, models };
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

function paintedPaths(recorder, paint = 'stroke') {
  const paths = [];
  let commands = [];
  for (const operation of recorder.operations) {
    if (operation[0] === 'beginPath') commands = [];
    else if (['moveTo', 'lineTo'].includes(operation[0])) commands.push(operation);
    else if (operation[0] === paint) paths.push(commands.slice());
  }
  return paths;
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

// This graph checks the actual drawing's physical connections. A conductor is
// retained even when transfer packets stop; only an open modeled contact cuts
// it. Transformer winding links are explicit, separate from conductor edges.
const terminalKey = terminal => Array.from(terminal, coordinate => coordinate.toFixed(7)).join(',');
function reachableTerminals(model, transformers, starts, event, includeWindings = true) {
  const adjacency = new Map();
  const connect = (from, to) => {
    const a = terminalKey(from), b = terminalKey(to);
    if (!adjacency.has(a)) adjacency.set(a, new Set());
    if (!adjacency.has(b)) adjacency.set(b, new Set());
    adjacency.get(a).add(b); adjacency.get(b).add(a);
  };
  for (const route of model.routes) {
    if (route.breakerRole && route.circuit === 'primary') {
      const open = route.breakerRole === 'sending' ? event.sendingOpen
        : route.breakerRole === 'receiving' ? event.receivingOpen : event.primaryFeederOpen;
      if (open > 0) continue;
    }
    if (route.breakerRole === 'feeder' && route.circuit === 'alternate' && event.alternateFeederOpen > 0) continue;
    for (let index = 1; index < route.points.length; index += 1) connect(route.points[index - 1], route.points[index]);
  }
  if (includeWindings) {
    for (const { terminals } of transformers) {
      const high = terminals.high || terminals.primary;
      const low = terminals.low || terminals.secondary;
      high.forEach((terminal, phase) => connect(terminal, low[phase]));
    }
  }
  const seen = new Set(starts.map(terminalKey));
  const pending = [...seen];
  for (const current of pending) {
    for (const next of adjacency.get(current) || []) {
      if (!seen.has(next)) { seen.add(next); pending.push(next); }
    }
  }
  return seen;
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
  assert.equal(poles.length, 6, `${layout}: the two distribution feeders require separate three-pole routes`);
  assert.equal(new Set(poles.flatMap(record => record.terminals.map(terminalKey))).size, 18, `${layout}: the two distribution feeders must not share pole terminals`);
  const downstream = model.routes.filter(route => route.zone === 'downstream' || route.zone.startsWith('distribution-'));
  const pointKey = terminalKey;
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
  assert.ok(ac, `${layout}: storage needs its own converter-to-substation tie`);
  // Observe rendered pulses on real model segments; do not duplicate the
  // renderer's mirrored scheduling formula in the test's expected result.
  assert.ok(firstTransferPulse(dc, 1, 3) < firstTransferPulse(ac, 1, 3), `${layout}: discharge must leave the DC storage side before entering its AC tie`);
  assert.ok(firstTransferPulse(ac, 10, 12) < firstTransferPulse(dc, 10, 12), `${layout}: charging must enter from the AC tie before reaching DC storage`);
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
const substationEvent = renderer.getEventState({ state: 'substation' });
assert.equal(internals.zonePower('line', isolatedEvent), 0, 'Opening both ends must de-energize circuit A');
assert.equal(internals.zonePower('line', substationEvent), 1, 'A closed sending breaker must keep its open-ended line visibly energized');
assert.equal(internals.zonePower('station-primary', isolatedEvent), 1, 'The common MV bus must keep station A energized during a line-only outage');
assert.equal(internals.zonePower('station-primary', substationEvent), 0, 'A station isolated at HV and MV must be visibly de-energized');
assert.equal(isolatedEvent.sendingOpen, 1);
assert.equal(isolatedEvent.receivingOpen, 1);
assert.equal(isolatedEvent.linePower, 0, 'Both primary terminals must isolate the faulted circuit');
assert.equal(isolatedEvent.alternatePower, 1, 'The second circuit must remain energized');
assert.equal(isolatedEvent.downstreamPower, 1, 'The healthy circuit must continue supplying the distribution network');
assert.equal(isolatedEvent.primaryStationPower, 1, 'The connected MV bus can energize station A during a line-only outage');
assert.equal(isolatedEvent.primaryStationFlow, 0, 'An isolated incoming circuit must not imply real power transfer through station A');
assert.equal(isolatedEvent.primaryFeederOpen, 0, 'A line-only outage must leave the station MV feeder connected');
assert.equal(sendingClosedEvent.sendingOpen, 0);
assert.equal(sendingClosedEvent.receivingOpen, 1);
assert.equal(sendingClosedEvent.linePower, 1);
assert.equal(sendingClosedEvent.downstreamPower, 1, 'Reclosing the primary circuit must not interrupt the alternate supply');
const switchingTransitions = [
  [8.3, 1, 1, 'Both contacts still closed at the start of the trip'],
  [8.325, 1, 0, 'Sending contacts separating while the receiving side backfeeds the line'],
  [8.35, 1, 0, 'Receiving contacts still closed at the start of their motion'],
  [8.351, 0, 0, 'Both contacts separated'],
  [8.6, 0, 0, 'Both contacts opening'],
  [9.1, 0, 0, 'Both contacts fully open'],
  [14.4, 0, 0, 'Sending contacts approaching closure'],
  [14.45, 1, 0, 'Sending contacts closed with receiving contacts open'],
  [14.5, 1, 0, 'Line energized from the sending end only'],
  [15.1, 1, 0, 'Receiving contacts approaching closure'],
  [15.2, 1, 1, 'Both terminal contacts closed again'],
].map(([time, linePower, lineFlow, description]) => {
  const event = renderer.getEventState({ time });
  assert.equal(event.linePower, linePower, `${time}s: ${description} must determine line energization`);
  assert.equal(event.lineFlow, lineFlow, `${time}s: ${description} must determine through-transfer`);
  return { time, event };
});
for (const state of ['base', 'open', 'redistributed', 'substation']) {
  const initial = renderer.getEventState({ time: 0, state });
  for (const time of [2, 8.15, 10, 14.5, 19, 26]) {
    const event = renderer.getEventState({ time, state });
    assert.equal(event.stage, initial.stage, `${state}: manual switching state must not advance with time`);
    for (const field of ['sendingOpen', 'receivingOpen', 'linePower', 'lineFlow', 'alternatePower', 'downstreamPower', 'redistributionGain', 'primaryStationPower', 'alternateStationPower', 'primaryStationFlow', 'alternateStationFlow', 'primaryFeederOpen', 'alternateFeederOpen']) {
      assert.equal(event[field], initial[field], `${state}: manual ${field} must remain fixed`);
    }
    assert.equal(event.strikeOpacity, 0, 'Manual state inspection must not flash lightning');
  }
  assert.equal(initial.sendingOpen, state === 'base' || state === 'substation' ? 0 : 1);
  assert.equal(initial.receivingOpen, state === 'base' ? 0 : 1);
  assert.equal(initial.linePower, state === 'base' || state === 'substation' ? 1 : 0);
  assert.equal(initial.lineFlow, state === 'base' ? 1 : 0, 'An open-ended line can stay energized without through-transfer packets');
  assert.equal(initial.alternatePower, 1);
  assert.equal(initial.downstreamPower, 1);
  assert.equal(initial.primaryStationPower, state === 'substation' ? 0 : 1);
  assert.equal(initial.alternateStationPower, 1);
  assert.equal(initial.primaryStationFlow, state === 'base' ? 1 : 0);
  assert.equal(initial.alternateStationFlow, 1);
  assert.equal(initial.primaryFeederOpen, state === 'substation' ? 1 : 0, 'A station outage must open its MV contact to prevent backfeed');
  assert.equal(initial.alternateFeederOpen, 0);
  assert.equal(initial.redistributionGain, ['redistributed', 'substation'].includes(state) ? 1 : 0, 'Manual redistribution changes illustrative emphasis, not supply continuity');
  assert.equal(typeof initial.label, 'string');
  assert.ok(initial.label.trim());
}
assert.notEqual(substationEvent.label, isolatedEvent.label, 'Station and line outages must have distinct status descriptions');
assert.deepEqual(
  { ...renderer.getEventState({ time: 10, state: 'invalid' }) },
  { ...isolatedEvent },
  'An invalid manual state should use the automatic event sequence',
);
let flashRuns = 0;
let previousFlash = false;
for (let sample = 0; sample <= 2600; sample += 1) {
  const event = renderer.getEventState({ time: sample / 100 });
  for (const field of ['sendingOpen', 'receivingOpen', 'linePower', 'lineFlow', 'alternatePower', 'downstreamPower', 'redistributionGain', 'restorationProgress', 'primaryStationPower', 'alternateStationPower', 'primaryStationFlow', 'alternateStationFlow', 'primaryFeederOpen', 'alternateFeederOpen']) assert.ok(Number.isFinite(event[field]) && event[field] >= 0 && event[field] <= 1, `${field} must remain bounded`);
  assert.ok(event.strikeOpacity >= 0 && event.strikeOpacity <= 0.85 + 1e-12, 'Lightning intensity must remain bounded');
  if (event.lineFlow > 0) assert.ok(event.sendingOpen === 0 && event.receivingOpen === 0, 'Through-transfer must stop as soon as either terminal contact separates');
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
  assert.equal(model.breakers.length, 18, `${layout}: both circuits need three-pole sending, receiving HV, and station MV breakers`);
  for (const circuit of ['primary', 'alternate']) {
    for (const role of ['sending', 'receiving', 'feeder']) {
      assert.deepEqual(Array.from(model.breakers.filter(breaker => breaker.circuit === circuit && breaker.role === role), breaker => breaker.phase).sort(), [0, 1, 2], `${layout}/${circuit}/${role}: phases must remain separate`);
    }
  }
  const network = Array.from(internals.networkBuildRecords).filter(record => record.layout === layout);
  const plants = Array.from(internals.sourceBuildRecords).filter(record => record.layout === layout);
  const generationTerminals = plants.flatMap(record => Array.from(record.terminals));
  const transformers = network.filter(record => ['transformer', 'serviceTransformer'].includes(record.name));
  const mainTransformers = transformers.filter(record => record.name === 'transformer');
  assert.equal(mainTransformers.length, 4, `${layout}: both paths need their own sending step-up and receiving step-down transformers`);
  const stepUp = mainTransformers.find(record => !record.args[0] && (record.args[1] || 'primary') === 'primary');
  const alternateStepUp = mainTransformers.find(record => !record.args[0] && record.args[1] === 'alternate');
  const stationA = mainTransformers.find(record => record.args[0] && (record.args[1] || 'primary') === 'primary');
  const stationB = mainTransformers.find(record => record.args[0] && record.args[1] === 'alternate');
  assert.ok(stepUp && alternateStepUp && stationA && stationB, `${layout}: transformer roles must identify both sending and both receiving stations`);
  assert.equal(new Set(mainTransformers.flatMap(record => [...record.terminals.high, ...record.terminals.low].map(terminalKey))).size, 24, `${layout}: all main transformer winding terminals must be physically distinct`);
  const sendingBuses = network.filter(record => record.name === 'sendingBus');
  assert.equal(sendingBuses.length, 2, `${layout}: the sending substations must have separate source buswork`);
  for (const bus of sendingBuses) {
    assert.equal(bus.terminals.bays.length, 3, `${layout}: each sending station needs three individual plant bays`);
    assert.equal(new Set(bus.terminals.bays.flatMap(bay => bay.map(terminalKey))).size, 9, `${layout}: each plant bay must have three distinct phase terminals`);
  }
  const plantTies = [];
  for (const plant of plants) {
    const { id, circuit, bay } = sourceConnections[plant.name];
    const ownStation = circuit === 'primary' ? stepUp : alternateStepUp;
    const otherStation = circuit === 'primary' ? alternateStepUp : stepUp;
    const bus = sendingBuses.find(record => record.args[0] === circuit);
    assert.ok(bus, `${layout}/${plant.name}: the assigned sending bus must exist`);
    for (const [phase, terminal] of Array.from(plant.terminals).entries()) {
      const ties = model.routes.filter(route => terminalKey(route.points[0]) === terminalKey(terminal));
      assert.equal(ties.length, 1, `${layout}/${plant.name}/${phase}: each plant AC phase needs its own direct tie line`);
      assert.equal(ties[0].sourceId, id, `${layout}/${plant.name}/${phase}: source metadata must identify the connected plant`);
      assert.equal(ties[0].circuit, circuit);
      assert.equal(ties[0].bayIndex, bay);
      assert.equal(ties[0].phase, phase);
      assert.equal(terminalKey(ties[0].points.at(-1)), terminalKey(bus.terminals.bays[bay][phase]), `${layout}/${plant.name}/${phase}: the dedicated tie must end at its own phase terminal in its own bay`);
      plantTies.push(ties[0]);
      const connected = reachableTerminals(model, transformers, [terminal], normalEvent, false);
      assert.ok(connected.has(terminalKey(ownStation.terminals.low[phase])), `${layout}/${plant.name}/${phase}: the tie must reach its assigned sending station on the matching phase`);
      for (const [otherPhase, stationTerminal] of Array.from(ownStation.terminals.low).entries()) {
        assert.equal(connected.has(terminalKey(stationTerminal)), otherPhase === phase, `${layout}/${plant.name}/${phase}: station buswork must never join different phases`);
      }
      for (const stationTerminal of otherStation.terminals.low) {
        assert.equal(connected.has(terminalKey(stationTerminal)), false, `${layout}/${plant.name}/${phase}: no shared source collector or direct tie may join the two sending substations`);
      }
    }
  }
  assert.equal(model.routes.filter(route => route.sourceId).length, 18, `${layout}: all six plant connections must retain three traceable phase ties`);
  assert.equal(new Set(plantTies.map(route => terminalKey(route.points.at(-1)))).size, 18, `${layout}: every plant phase must arrive at its own station-bay terminal`);
  const tiePointSets = plantTies.map(route => new Set(route.points.map(terminalKey)));
  assert.equal(
    new Set(tiePointSets.flatMap(points => [...points])).size,
    tiePointSets.reduce((count, points) => count + points.size, 0),
    `${layout}: dedicated plant tie lines must not merge into a shared conductor before their station bays`,
  );
  const primaryHV = reachableTerminals(model, transformers, stepUp.terminals.high, normalEvent, false);
  const alternateHV = reachableTerminals(model, transformers, alternateStepUp.terminals.high, normalEvent, false);
  for (const terminal of stationA.terminals.high) assert.ok(primaryHV.has(terminalKey(terminal)), `${layout}: sending A must have its own conductor path to receiving A`);
  for (const terminal of stationB.terminals.high) assert.ok(alternateHV.has(terminalKey(terminal)), `${layout}: sending B must have its own conductor path to receiving B`);
  assert.ok([...primaryHV].every(terminal => !alternateHV.has(terminal)), `${layout}: the sending and receiving HV paths must remain separate instead of sharing a bus`);
  for (const [circuit, sendingTransformer] of [['primary', stepUp], ['alternate', alternateStepUp]]) {
    const lowTerminals = new Set(sendingTransformer.terminals.low.map(terminalKey));
    const highTerminals = new Set(sendingTransformer.terminals.high.map(terminalKey));
    const leads = model.routes.filter(route => lowTerminals.has(terminalKey(route.points.at(-1))) || highTerminals.has(terminalKey(route.points[0])));
    assert.equal(leads.length, 6, `${layout}/${circuit}: both sending transformer winding sides need distinct phase leads`);
    for (const lead of leads) {
      const conductor = model.items.find(item => item.kind === 'line' && item.points === lead.points);
      assert.ok(conductor, `${layout}/${circuit}: the sending transformer lead must be drawn`);
      for (const event of [isolatedEvent, substationEvent]) {
        assert.equal(internals.zonePower(conductor.circuitZone, event), 1, `${layout}/${circuit}: each sending transformer stays energized by its connected plants`);
        assert.equal(internals.routePower(lead, event), circuit === 'primary' ? 0 : 1, `${layout}/${circuit}: an unloaded sending transformer must suppress transfer packets on both winding sides`);
      }
    }
  }
  const service = transformers.find(record => record.name === 'serviceTransformer');
  const feederRoutes = circuit => model.routes.filter(route => route.zone === `distribution-${circuit}`);
  const primaryFeederPoints = new Set(feederRoutes('primary').flatMap(route => route.points.map(terminalKey)));
  const alternateFeederPoints = new Set(feederRoutes('alternate').flatMap(route => route.points.map(terminalKey)));
  assert.deepEqual(
    [...primaryFeederPoints].filter(terminal => alternateFeederPoints.has(terminal)).sort(),
    Array.from(service.terminals.primary, terminalKey).sort(),
    `${layout}: the separate feeders must join only at the load-side service-transformer primary terminals`,
  );
  for (const circuit of ['primary', 'alternate']) {
    assert.ok(feederRoutes(circuit).length >= 12, `${layout}/${circuit}: all three phases must traverse a distinct multi-span feeder`);
    for (const route of feederRoutes(circuit)) {
      const conductor = model.items.find(item => item.kind === 'line' && item.points === route.points);
      assert.ok(conductor, `${layout}/${circuit}: each feeder span must draw an actual conductor`);
      assert.equal(internals.zonePower(conductor.circuitZone, substationEvent), 1, `${layout}/${circuit}: the common load-side bus must retain feeder energization after station A is isolated`);
    }
  }
  assert.equal(network.filter(record => record.name === 'distributionCabinet').length, 2, `${layout}: each substation requires its own distribution switchgear`);
  const feederA = network.find(record => record.name === 'feederBreaker' && record.args[0] === 'primary');
  assert.ok(feederA, `${layout}: station A must expose its actual MV contact terminals`);
  const outletKeys = new Set(feederA.terminals.outputs.map(terminalKey));
  const busStubs = model.routes.filter(route => outletKeys.has(terminalKey(route.points[0])));
  assert.equal(busStubs.length, 3, `${layout}: each MV breaker output must have its own conductor to the common distribution bus`);
  for (const route of busStubs) {
    const conductor = model.items.find(item => item.kind === 'line' && item.points === route.points);
    assert.ok(conductor, `${layout}: the MV output connection must draw an actual conductor`);
    assert.equal(internals.zonePower(conductor.circuitZone, substationEvent), 1, `${layout}: B must keep the bus-side conductor visibly energized after A's MV contact opens`);
    assert.equal(internals.routePower(route, substationEvent), 0, `${layout}: an energized MV stub must not show through-transfer across A's open contact`);
    assert.equal(route.breakerRole, undefined, `${layout}: the bus-side conductor must remain connected when only the internal breaker contact opens`);
  }
  for (const transformer of transformers) {
    const high = transformer.terminals.high || transformer.terminals.primary;
    const low = transformer.terminals.low || transformer.terminals.secondary;
    const conductors = reachableTerminals(model, transformers, high, normalEvent, false);
    assert.ok(low.every(terminal => !conductors.has(terminalKey(terminal))), `${layout}: a conductor path must never bypass a transformer winding`);
  }
  const loadBranches = model.routes.filter(route => route.zone === 'downstream' && route.start >= 0.94);
  assert.ok(loadBranches.length >= 14, `${layout}: continuity checks must cover campus and residential load branches`);
  for (const [remaining, removedStepUp] of [['primary', alternateStepUp], ['alternate', stepUp]]) {
    const unavailable = remaining === 'primary' ? 'alternate' : 'primary';
    const remainingModel = { routes: model.routes.filter(route => route.zone !== `distribution-${unavailable}`) };
    const remainingTransformers = transformers.filter(transformer => transformer !== removedStepUp);
    const remainingSources = plants.filter(record => sourceConnections[record.name].circuit === remaining).flatMap(record => Array.from(record.terminals));
    const reachable = reachableTerminals(remainingModel, remainingTransformers, remainingSources, normalEvent);
    for (const route of loadBranches) assert.ok(reachable.has(terminalKey(route.points.at(-1))), `${layout}/${remaining}: either sending/receiving/feeder path must independently reach every load after removing the other sending winding and feeder`);
  }
  for (const [name, event] of [['line', isolatedEvent], ['substation', substationEvent]]) {
    const energized = reachableTerminals(model, transformers, generationTerminals, event);
    for (const route of loadBranches) assert.ok(energized.has(terminalKey(route.points.at(-1))), `${layout}/${name}: each drawn load branch must retain an actual path through a transformer and the healthy station`);
    for (const terminal of [...stationB.terminals.high, ...stationB.terminals.low]) assert.ok(energized.has(terminalKey(terminal)), `${layout}/${name}: station B must remain connected at both voltage stages`);
    for (const terminal of [...stationA.terminals.high, ...stationA.terminals.low]) assert.equal(energized.has(terminalKey(terminal)), name === 'line', `${layout}/${name}: station A must be energized through the MV coupling only when its MV breaker remains closed`);
    for (const terminal of feederA.terminals.outputs) assert.ok(energized.has(terminalKey(terminal)), `${layout}/${name}: A's MV breaker bus side must remain physically connected to the healthy supply`);
    for (const terminal of feederA.terminals.inputs) assert.equal(energized.has(terminalKey(terminal)), name === 'line', `${layout}/${name}: A's MV breaker station side must follow the actual contact isolation`);
  }
  for (const { time, event } of switchingTransitions) {
    const energized = reachableTerminals(model, transformers, generationTerminals, event);
    const connectedHV = reachableTerminals(model, transformers, stepUp.terminals.high, event, false);
    const throughConnected = stationA.terminals.high.every(terminal => connectedHV.has(terminalKey(terminal)));
    for (const route of model.routes.filter(route => route.zone === 'line' && !route.breakerRole)) {
      for (const terminal of [route.points[0], route.points.at(-1)]) {
        assert.equal(Boolean(internals.zonePower('line', event)), energized.has(terminalKey(terminal)), `${layout}/${time}s: line energization must match actual source connectivity through its moving contacts`);
      }
      assert.equal(Boolean(internals.routePower(route, event)), throughConnected, `${layout}/${time}s: transmission pulses must match the physical connection between the two HV stations`);
    }
    for (const route of model.routes.filter(route => route.circuit === 'primary' && route.breakerRole)) {
      assert.equal(Boolean(internals.routePower(route, event)), throughConnected, `${layout}/${time}s/${route.breakerRole}: internal breaker pulses must stop with the circuit's through-transfer`);
    }
  }
  const zones = ['source', 'sending-primary', 'sending-alternate', 'line', 'alternate', 'station-primary', 'station-alternate', 'distribution-primary', 'distribution-alternate', 'downstream'];
  for (const zone of zones) assert.ok(model.routes.some(route => route.zone === zone), `${layout}: ${zone} routes must be represented`);
  for (const route of model.routes) {
    assert.ok(zones.includes(route.zone));
    const primaryLineContact = ['sending', 'receiving'].includes(route.breakerRole) && route.circuit === 'primary';
    const primaryContact = route.breakerRole && route.circuit === 'primary';
    assert.equal(internals.routePower(route, isolatedEvent), ['sending-primary', 'line', 'station-primary', 'distribution-primary'].includes(route.zone) || primaryLineContact ? 0 : 1, `${layout}: line isolation stops sending A, circuit A, receiving A, and feeder A through-transfer packets`);
    assert.equal(internals.routePower(route, substationEvent), ['sending-primary', 'line', 'station-primary', 'distribution-primary'].includes(route.zone) || primaryContact ? 0 : 1, `${layout}: station isolation stops A through-transfer packets while its sending line and backfed feeder stay energized`);
  }
  const isolatedRoutes = recordingContext();
  internals.drawRoutes(isolatedRoutes.context, { routes: model.routes.filter(route => ['sending-primary', 'line', 'station-primary', 'distribution-primary'].includes(route.zone)) }, testView, 2, isolatedEvent);
  assert.equal(isolatedRoutes.paints.length, 0, 'The isolated primary circuit and its unloaded transformers and feeder must show no through-transfer packets');
  const isolatedStation = recordingContext();
  internals.drawRoutes(isolatedStation.context, { routes: model.routes.filter(route => ['station-primary', 'distribution-primary'].includes(route.zone)) }, testView, 2, substationEvent);
  assert.equal(isolatedStation.paints.length, 0, 'An isolated station and its backfed feeder must show no through-transfer packets');
  for (const event of [isolatedEvent, substationEvent]) {
    for (const zone of ['source', 'sending-alternate', 'alternate', 'station-alternate', 'distribution-alternate', 'downstream']) {
      const liveRoutes = recordingContext();
      // Individual live segments may be between packets at any one instant.
      // Observe a full pulse cycle so removing feeder routes from the load zone
      // cannot turn a harmless phase offset into a false outage assertion.
      for (let sample = 0; sample < 24 && !liveRoutes.paints.length; sample += 1) {
        internals.drawRoutes(liveRoutes.context, { routes: model.routes.filter(route => route.zone === zone) }, testView, sample * 0.4, event);
      }
      assert.ok(liveRoutes.paints.length > 0, `${layout}: ${zone} transfer artwork must remain active through either outage`);
    }
  }
  const loadDetails = { ...emptyDetails, lights: model.lights.filter(light => light.zone === 'downstream'), flux: model.flux.filter(flux => flux.zone === 'downstream') };
  const energizedLoads = recordingContext();
  internals.drawDetails(energizedLoads.context, loadDetails, testView, 2, normalEvent);
  assert.ok(energizedLoads.paints.length > 0, 'Normal operation should paint load windows and receiving-transformer effects');
  const suppliedLoads = recordingContext();
  internals.drawDetails(suppliedLoads.context, loadDetails, testView, 2, isolatedEvent);
  assert.equal(suppliedLoads.digest(), energizedLoads.digest(), 'Load windows and receiving-transformer effects must retain supply through the healthy circuit');
  const suppliedStationOutage = recordingContext();
  internals.drawDetails(suppliedStationOutage.context, loadDetails, testView, 2, substationEvent);
  assert.equal(suppliedStationOutage.digest(), energizedLoads.digest(), 'Load windows must retain supply after station A is isolated');
  const primaryFlux = model.flux.filter(flux => flux.zone === 'station-primary');
  assert.equal(primaryFlux.length, 1, 'Station A must have its own gated transformer effect');
  const energizedStationFlux = recordingContext(), isolatedStationFlux = recordingContext();
  internals.drawDetails(energizedStationFlux.context, { ...emptyDetails, flux: primaryFlux }, testView, 2, isolatedEvent);
  internals.drawDetails(isolatedStationFlux.context, { ...emptyDetails, flux: primaryFlux }, testView, 2, substationEvent);
  assert.ok(energizedStationFlux.paints.length > 0, 'The connected MV bus must keep the unloaded station transformer energized after a line outage');
  assert.equal(isolatedStationFlux.paints.length, 0, 'Opening both station voltage sides must remove its transformer energization effect');
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
    if (breaker.circuit === 'primary' && breaker.role !== 'feeder') {
      assert.ok(gap(isolatedEvent) > 1, 'Both ends of the primary circuit must show a visible contact gap');
    } else {
      assert.equal(gap(isolatedEvent), 0, 'Healthy transmission and both station MV contacts must remain visibly closed during a line outage');
    }
    assert.equal(gap(sendingClosedEvent) > 1, breaker.circuit === 'primary' && breaker.role === 'receiving', 'Only the primary receiving contacts should remain open during ordered restoration');
    assert.equal(gap(substationEvent) > 1, breaker.circuit === 'primary' && ['receiving', 'feeder'].includes(breaker.role), 'Station A requires visible HV and MV gaps while the primary sending and all B contacts stay closed');
  }
  const cueView = { ...testView, mobile: layout !== 'desktop' };
  const projectedPath = points => Array.from(points, (point, index) => {
    const projected = new Float32Array(2);
    internals.projection(point, cueView, projected);
    return [index ? 'lineTo' : 'moveTo', ...projected];
  });
  const contactFrame = (breaker, event) => {
    const recorder = recordingContext();
    internals.drawBreakers(recorder.context, { breakers: [breaker] }, cueView, event, false);
    return recorder;
  };
  const cueFrame = (breakers, event) => {
    const recorder = recordingContext();
    internals.drawBreakerCues(recorder.context, { breakers }, cueView, event);
    recorder.assertRestored();
    return recorder;
  };
  const cueCases = [
    [normalEvent, []],
    [isolatedEvent, ['sending', 'receiving']],
    [substationEvent, ['receiving', 'feeder']],
    [sendingClosedEvent, ['receiving']],
  ];
  for (const [event, roles] of cueCases) {
    const frame = cueFrame(model.breakers, event);
    assert.equal(paintedPaths(frame, 'fill').length, roles.length * 3, `${layout}: all three physical poles of each operating device must remain visible`);
    assert.equal(frame.operations.some(operation => ['fillRect', 'strokeRect'].includes(operation[0])), false, `${layout}: no independently positioned switch box may cover the actual conductor geometry`);
    for (const breaker of model.breakers) {
      const single = cueFrame([breaker], event);
      const expected = breaker.circuit === 'primary' && roles.includes(breaker.role);
      assert.equal(single.paints.length > 0, expected, `${layout}/${breaker.circuit}/${breaker.role}/${breaker.phase}: only physically opening poles should be highlighted`);
      if (!expected) continue;
      const contactPaths = paintedPaths(contactFrame(breaker, event));
      const highlightedPaths = paintedPaths(single);
      assert.deepEqual(highlightedPaths.slice(0, 4), contactPaths, `${layout}/${breaker.role}/${breaker.phase}: highlighted contacts and tips must exactly coincide with the actual moving apparatus`);
      assert.deepEqual(highlightedPaths.slice(4), [projectedPath(breaker.inputLead), projectedPath(breaker.outputLead)], `${layout}/${breaker.role}/${breaker.phase}: highlights must follow the real conductors to both terminal bushings`);
      assert.deepEqual(paintedPaths(single, 'fill'), [projectedPath(breaker.window)], `${layout}/${breaker.role}/${breaker.phase}: the contact background must remain within the actual projected cutaway`);
    }
  }
  for (const breaker of model.breakers) {
    const route = model.routes.find(route => route.breakerRole === breaker.role && route.circuit === breaker.circuit && route.phase === breaker.phase);
    assert.ok(route, `${layout}/${breaker.role}/${breaker.phase}: each physical breaker needs its own transfer path`);
    for (const lead of [breaker.inputLead, breaker.outputLead]) {
      assert.ok(model.items.some(item => item.kind === 'line' && item.points === lead), `${layout}/${breaker.role}/${breaker.phase}: highlighted terminal leads must exist in the rendered apparatus`);
    }
    assert.ok(model.items.some(item => item.kind === 'face' && item.points === breaker.window), `${layout}/${breaker.role}/${breaker.phase}: highlights must use the actual cutaway face`);
    const pathSegments = path => path.slice(1).map((point, index) => [path[index].slice(1), point.slice(1)]);
    const physicalSegments = [
      ...pathSegments(projectedPath(breaker.inputLead)),
      ...paintedPaths(contactFrame(breaker, normalEvent)).slice(0, 2).flatMap(pathSegments),
      ...pathSegments(projectedPath(breaker.outputLead)),
    ];
    assert.deepEqual(pathSegments(projectedPath(route.points)), physicalSegments, `${layout}/${breaker.circuit}/${breaker.role}/${breaker.phase}: transfer packets must follow the drawn leads and closed contact, without diagonal shortcuts`);
    if (breaker.circuit !== 'primary') continue;
    const field = breaker.role === 'sending' ? 'sendingOpen' : breaker.role === 'receiving' ? 'receivingOpen' : 'primaryFeederOpen';
    for (const opening of [0.05, 0.25, 0.5, 0.75, 1]) {
      const event = { ...normalEvent, [field]: opening };
      assert.deepEqual(paintedPaths(cueFrame([breaker], event)).slice(0, 4), paintedPaths(contactFrame(breaker, event)), `${layout}/${breaker.role}/${breaker.phase}/${opening}: contact highlights must stay aligned throughout their full travel`);
    }
  }
  const annotationFrame = event => {
    const recorder = recordingContext();
    internals.drawAnnotations(recorder.context, model, cueView, event);
    recorder.assertRestored();
    return recorder;
  };
  const baseAnnotations = annotationFrame(normalEvent);
  const textOperations = recorder => recorder.operations.filter(operation => operation[0] === 'fillText');
  const baseLabelPositions = textOperations(baseAnnotations).map(operation => operation.slice(2));
  for (const [event, roles] of cueCases) {
    const annotations = annotationFrame(event);
    assert.deepEqual(textOperations(annotations).map(operation => operation.slice(2)), baseLabelPositions, `${layout}: operating-state labels must reuse the existing scene annotation positions`);
    const leaders = annotations.operations.filter(operation => operation[0] === 'lineTo');
    for (const role of roles) {
      const breaker = model.breakers.find(item => item.circuit === 'primary' && item.phase === 1 && item.role === role);
      const anchor = new Float32Array(2);
      internals.projection(breaker.group.p(breaker.x, breaker.y, breaker.z), cueView, anchor);
      assert.ok(leaders.some(operation => operation[1] === anchor[0] && operation[2] === anchor[1] - 2), `${layout}/${role}: the operating label must point to its actual opening device`);
    }
  }
  const annotationLabels = event => textOperations(annotationFrame(event)).map(operation => operation[1]);
  const sendingLabel = cueView.mobile ? 'A send' : 'Sending A';
  assert.ok(annotationLabels(isolatedEvent).includes(`${sendingLabel} · open`));
  assert.ok(annotationLabels(isolatedEvent).includes('A HV · open'));
  assert.ok(annotationLabels(isolatedEvent).includes('A · line isolated'));
  assert.ok(annotationLabels(isolatedEvent).includes(cueView.mobile ? 'B · supply' : 'B · supplying loads'));
  assert.ok(annotationLabels(substationEvent).includes('A HV/MV · open'));
  assert.ok(annotationLabels(renderer.getEventState({ time: 8.15, reducedMotion: true })).includes('A · fault'), 'Suppressing the bolt must preserve the readable fault state');
  assert.ok(annotationLabels(renderer.getEventState({ time: 8.6 })).includes(`${sendingLabel} · opening`));
  assert.ok(annotationLabels(renderer.getEventState({ time: 8.6 })).includes('A · opening'));
  assert.ok(annotationLabels(renderer.getEventState({ time: 14 })).includes(`${sendingLabel} · closing`));
  assert.ok(annotationLabels(renderer.getEventState({ time: 14 })).includes('A · isolated'));
  assert.ok(annotationLabels(sendingClosedEvent).includes('A · energized'));
}

// Exercise the actual base-face paint loop with only emissive items retained.
const desktopModel = internals.models.get('desktop');
assert.ok(desktopModel.items.some(item => item.emission), 'Load windows need explicitly gated base emission');
for (const [width, height] of [[1440, 900], [640, 980], [390, 980], [320, 980]]) {
  const options = { width, height, state: 'open', time: 2, reducedMotion: true };
  const first = trace(options);
  assert.equal(trace({ ...options, time: 19 }).digest(), first.digest(), `${width}px: reduced-motion manual inspection must remain still`);
  assert.notEqual(trace({ ...options, state: 'base' }).digest(), first.digest(), `${width}px: the combined journey must visibly distinguish closed and isolated circuits`);
  assert.notEqual(trace({ ...options, state: 'redistributed' }).digest(), first.digest(), `${width}px: the combined journey must distinguish open contacts from redistribution emphasis`);
  assert.notEqual(trace({ ...options, state: 'substation' }).digest(), first.digest(), `${width}px: the combined journey must distinguish station isolation from line isolation`);
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
const phoneModel = internals.models.get('phone');
try {
  // Fault text should remain readable even when its bolt is suppressed. Remove
  // only annotations here to compare the otherwise identical equipment frames.
  internals.models.set('phone', { ...phoneModel, annotations: [] });
  assert.equal(trace({ ...eventOptions, suppressFlash: true }).digest(), trace({ ...eventOptions, eventTime: 0 }).digest(), 'Suppressing flash must preserve ambient motion and otherwise-powered equipment');
} finally { internals.models.set('phone', phoneModel); }
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
console.log('PASS: manual/automatic event states; line and substation isolation use distinct HV/MV contact gaps, preserve alternate supply, and gate station energization separately from transfer packets.');
console.log('PASS: four main transformers, separate HV paths and six distribution poles; feeders join only at the load bus; either complete path independently reaches every load; no winding bypass; distinct manual frames through 320px.');
console.log('PASS: six dedicated three-phase plant ties connect distinct bays on separate sending buses, preserve phase separation, and supply independent paths from actual plant terminals.');
