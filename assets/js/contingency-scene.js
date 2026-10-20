/*
 * AELab's close-up presentation of the shared illustrative power journey.
 * The hero and story use the same topology, contact positions and event timing.
 * The controller owns canvas sizing, motion preferences and scheduling.
 */
(() => {
  'use strict';

  if (typeof window.PowerJourney?.draw !== 'function' || typeof window.PowerJourney?.getEventState !== 'function') return;
  const labels = Object.freeze({
    base: 'Both transmission circuits closed',
    open: 'Circuit A opens · alternate supply remains',
    redistributed: 'Alternate circuit supplies distribution and loads',
    restoring: 'Returning to two-circuit operation',
  });
  const manual = new Set(['base', 'open', 'redistributed']);
  const stages = Object.freeze({
    normal: 'base', strike: 'open', tripping: 'open', isolated: 'redistributed',
    reclosing: 'restoring', restoring: 'restoring', restored: 'base',
  });

  /** Resolve story labels from the shared event, preserving manual controls. */
  function getStage({ time = 0, state = 'auto' } = {}) {
    if (manual.has(state)) return state;
    return stages[window.PowerJourney.getEventState({ time, state }).stage];
  }

  /** Draw the common grid with enlarged terminal contact views below it. */
  function draw(ctx, options = {}) {
    window.PowerJourney.draw(ctx, { ...options, presentation: 'story', eventTime: options.time ?? 0 });
  }

  window.ContingencyScene = Object.freeze({ types: ['contingency'], labels, getStage, draw });
})();
