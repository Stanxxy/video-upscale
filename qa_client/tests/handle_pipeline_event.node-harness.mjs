#!/usr/bin/env node
// ============================================================================
// Deterministic headless-Node harness for vlm_studio.html's pipeline-event
// dispatch logic (handlePipelineEvent()) and the runState it mutates.
//
// Technique: per INS-077 (working_log/knowledge-base/insights/
// INS-077-youtube-iframe-api-overwrites-injected-mock-player-unless-network-
// blocked.md), a full browser (Playwright/chrome-devtools) fighting the real
// YouTube iframe API's window.YT race is the wrong tool for testing PURE
// event-dispatch/state logic that never touches the player. Instead this
// harness runs the page's REAL inline <script> body (extracted verbatim, not
// re-implemented/duplicated) inside a Node `vm` context with a minimal inert
// DOM stub — bypassing the DOM/YT-player boundary entirely. No browser, no
// network, no test framework/dependency (vanilla assert, per project
// convention — "no framework").
//
// This is a REGRESSION test for the bug found by the live end-to-end gate
// (evaluator loop, 2026-07-18): `case 'error':` in handlePipelineEvent only
// ever branched on `evt.chunk_index`, never `evt.highlight_index` — a
// highlight_analyze PASS-2 failure left the matching window's status stuck
// at 'running' forever (even past run_complete), so the Highlights rail row/
// timeline block/per-highlight card all silently showed "analyzing…" for a
// call that had already failed. Checks 1-3 cover the new highlight_map/
// highlight_start/highlight_result event paths that shipped with zero JS
// coverage (root cause of why the bug wasn't caught before the live gate);
// checks 4-6 are the regression itself.
//
// Run: node qa_client/tests/handle_pipeline_event.node-harness.mjs
// Exit code 0 = all checks passed; 1 = at least one failure (message printed).
// ============================================================================

import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const htmlPath = path.join(__dirname, '..', 'vlm_studio.html');
const html = fs.readFileSync(htmlPath, 'utf8');

const scriptMatch = html.match(/<script>([\s\S]*)<\/script>/);
assert.ok(scriptMatch, 'vlm_studio.html: could not locate the inline <script> block — has the page structure changed?');
let scriptBody = scriptMatch[1];

// Strip the trailing "Init" bootstrap block (setMode('event'); ... a live,
// unawaited loadPipelines() fetch; ...) — this harness has no reason to run
// page-bootstrap side effects; every function/variable this test needs is
// declared BEFORE that marker. Fails loudly (not silently) if the marker
// text ever moves, so a future refactor can't quietly make this harness stop
// testing what it claims to.
const initMarkerIndex = scriptBody.indexOf('// Init\n// ============================================================\nsetMode(\'event\');');
assert.ok(
  initMarkerIndex !== -1,
  'vlm_studio.html: "// Init" bootstrap marker not found in its expected exact form — ' +
  'update this harness\'s truncation point (it intentionally does not run the page-bootstrap ' +
  'block, which does a live loadPipelines() fetch).',
);
scriptBody = scriptBody.slice(0, initMarkerIndex);

// ---- Minimal inert DOM stub -------------------------------------------------
// Every element id the render functions this test exercises touch gets a
// single cached, mutable stub object (so writes made by one render call
// persist and are visible to the next, same as a real DOM element) exposing
// only the handful of members those functions actually call. Nothing here
// needs to look/behave like a real DOM node beyond that — the whole point of
// this technique is to bypass the DOM, not simulate it.
const elementsById = new Map();
function makeElementStub() {
  return {
    value: '', textContent: '', innerHTML: '', className: '',
    style: {}, scrollTop: 0, scrollHeight: 0, clientHeight: 0,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    querySelectorAll() { return []; },
    querySelector() { return null; },
    getAttribute() { return null; },
    setAttribute() {},
    addEventListener() {},
    appendChild() {},
  };
}
function getElementById(id) {
  if (!elementsById.has(id)) elementsById.set(id, makeElementStub());
  return elementsById.get(id);
}

const sandbox = {
  console,
  document: {
    getElementById,
    querySelectorAll() { return []; },
    querySelector() { return null; },
    addEventListener() {},
    createElement() { return makeElementStub(); }, // logMsg() builds a <div class="log-entry"> per line
    body: makeElementStub(),
  },
  location: { search: '' }, // isMock() reads location.search — force real (non-mock) code paths
  performance: { now: () => Date.now() },
  URLSearchParams,
  // Network is never expected to be reached (the Init block that calls
  // loadPipelines()/fetch() has been stripped above) — a throwing stub makes
  // an accidental network call fail loudly rather than hang/silently no-op.
  fetch: async () => { throw new Error('network disabled in this harness — should never be reached (Init block stripped)'); },
  setInterval: () => 0,
  clearInterval: () => {},
  setTimeout: () => 0,
  clearTimeout: () => {},
};
sandbox.window = sandbox;

const context = vm.createContext(sandbox);
vm.runInContext(scriptBody, context, { filename: 'vlm_studio.html (inline <script>, Init-block stripped)' });

function run(code) {
  return vm.runInContext(code, context);
}

// ---- Test helpers ------------------------------------------------------------
let failures = 0;
function check(name, fn) {
  try {
    fn();
    console.log('  ok   - ' + name);
  } catch (e) {
    failures += 1;
    console.error('  FAIL - ' + name + ': ' + e.message);
  }
}

console.log('handle_pipeline_event.node-harness.mjs — highlight-scan-analyze event coverage\n');

// Fresh run state, scoped to the highlight-scan-analyze pipeline (drives the
// currentPipelineId-gated render paths this pipeline uses — renderHighlightTimeline/
// renderHighlightsRail/renderDedupedClips's hide-for-this-pipeline branch —
// even though this harness never actually calls those render functions with
// DOM assertions; the point is the STATE `handlePipelineEvent` produces).
run("currentPipelineId = 'highlight-scan-analyze'; resetRunView(); runState.status = 'running';");

// 1. highlight_map populates runState.highlights (drives the timeline bar +
//    Highlights rail — previously zero JS coverage on this event at all).
run(`handlePipelineEvent({ type: 'highlight_map', highlights: [
  { index: 1, start_s: 10, end_s: 25, adjustment: null, start_is_synthetic: false, end_is_synthetic: false },
  { index: 2, start_s: 30, end_s: 40, adjustment: null, start_is_synthetic: false, end_is_synthetic: false }
], timing: { model_ms: 1200 } });`);
check('highlight_map populates runState.highlights (2 entries)', () => {
  assert.equal(run('runState.highlights.length'), 2);
});

// 2. highlight_start marks the matching window 'running' and captures BOTH
//    the analyzed (pre/post-roll-expanded) scope and the highlight's own
//    bounds (w.highlightBounds — previously captured but never rendered;
//    now surfaced via highlightCardRangeHtml()).
run(`handlePipelineEvent({ type: 'highlight_start', highlight_index: 1, scope: [5, 29], highlight_bounds: [10, 25] });`);
check('highlight_start marks window 1 running with scope + highlightBounds', () => {
  assert.equal(run("runState.windows.find(w => w.windowIndex === 1).status"), 'running');
  assert.equal(run("JSON.stringify(runState.windows.find(w => w.windowIndex === 1).scope)"), '[5,29]');
  assert.equal(run("JSON.stringify(runState.windows.find(w => w.windowIndex === 1).highlightBounds)"), '[10,25]');
});

// 3. highlight_result marks the matching window 'complete' with its clips.
run(`handlePipelineEvent({ type: 'highlight_result', highlight_index: 1, format: 'simplified-tags-time-v1',
  clips: [{ start_s: 11, end_s: 20, position: 'guard', actor: 'blue-gi', action_class: 'sweep', outcome: 'successful', confidence: 0.8 }],
  dropped_outside_bounds: 0, timing: { model_ms: 900 } });`);
check('highlight_result marks window 1 complete with 1 clip', () => {
  assert.equal(run("runState.windows.find(w => w.windowIndex === 1).status"), 'complete');
  assert.equal(run("runState.windows.find(w => w.windowIndex === 1).clips.length"), 1);
});

// 4-6. THE REGRESSION — an `error` event carrying `highlight_index` (no
//    `chunk_index`) must mark the matching window 'errored' with an
//    errorMessage, exactly like the pre-existing chunk_index branch does for
//    chunk-segment-tags. Before the fix, this event only updated
//    runState.errors[] (the Message Log) and left window 2's status stuck at
//    'running' — "analyzing…" — forever, even past run_complete.
run(`handlePipelineEvent({ type: 'highlight_start', highlight_index: 2, scope: [28, 44], highlight_bounds: [30, 40] });`);
run(`handlePipelineEvent({ type: 'error', stage_id: 'highlight_analyze', highlight_index: 2, message: 'Gemini 500' });`);
check('error with highlight_index marks the matching window errored (regression)', () => {
  assert.equal(run("runState.windows.find(w => w.windowIndex === 2).status"), 'errored');
  assert.equal(run("runState.windows.find(w => w.windowIndex === 2).errorMessage"), 'Gemini 500');
});
check('error with highlight_index still records the advisory message log entry (unchanged behavior)', () => {
  assert.equal(run('runState.errors[runState.errors.length - 1].message'), 'Gemini 500');
  assert.equal(run('runState.errors[runState.errors.length - 1].stageId'), 'highlight_analyze');
});

run(`handlePipelineEvent({ type: 'run_complete', clips: [
  { start_s: 11, end_s: 20, position: 'guard', actor: 'blue-gi', action_class: 'sweep', outcome: 'successful', confidence: 0.8 }
], format: 'simplified-tags-time-v1', dedup_applied: false, dedup_skip_reason: null, degraded_counts: null,
  per_stage_timing: {}, timing: { total_ms: 4200 } });`);
check('window 2 status stays errored across run_complete (never silently reset to a clean state)', () => {
  assert.equal(run("runState.windows.find(w => w.windowIndex === 2).status"), 'errored');
});
check('runState.sawRunComplete is true — errors are advisory/non-fatal, the run still completes', () => {
  assert.equal(run('runState.sawRunComplete'), true);
});

// ---- Rendered-HTML assertions (the actual user-visible regression) ---------
// Exercise the real render functions (not just runState) so this test would
// have caught the ACTUAL symptom (a permanently-yellow "analyzing…" row/an
// invisible-forever timeline block), not just an internal-state proxy for it.
check('highlightRailRowHtml(): errored highlight renders the errored badge + rail-row-errored class', () => {
  const html = run(`(function() {
    const h = runState.highlights.find(x => x.index === 2);
    return highlightRailRowHtml(h);
  })();`);
  assert.match(html, /rail-row-errored/, 'expected the .rail-row-errored class on an errored highlight\'s rail row');
  assert.match(html, />errored</, 'expected the "errored" status badge text');
  assert.match(html, /Gemini 500/, 'expected the captured errorMessage to be surfaced in the row');
});
check('highlightTimelineBlockHtml(): errored highlight gets timeline-block-errored (not the dead pending/running class)', () => {
  const html = run(`(function() {
    const h = runState.highlights.find(x => x.index === 2);
    return highlightTimelineBlockHtml(h, 0, 60);
  })();`);
  assert.match(html, /timeline-block-errored/, 'expected the errored timeline-block color class to be reachable for a highlight_index-keyed failure');
});
check('windowCardHtml(): errored highlight renders the (pipeline-neutral) PASS-2 failed message', () => {
  const html = run(`windowCardHtml(runState.windows.find(w => w.windowIndex === 2));`);
  assert.match(html, /PASS-2 failed: Gemini 500/, 'expected a generic "PASS-2 failed" label, not the old hardcoded "chunk failed" copy');
});

// ---- Defensive backwards-range guard (item 3) -------------------------------
check('eventPillHtml(): a backwards range (start > end) is visually flagged, not rendered as normal', () => {
  const normal = run(`eventPillHtml('1:25', '1:46')`);
  const backwards = run(`eventPillHtml('1:46', '1:25')`);
  assert.doesNotMatch(normal, /pill-inverted/, 'a normal forward range must NOT get the inverted-range class');
  assert.match(backwards, /pill-inverted/, 'a backwards range (start after end) must get the inverted-range class');
  assert.match(backwards, /backwards/, 'a backwards range must carry a visible warning label, not look like a normal pill');
});

// ---- highlightBounds surfaced on the per-highlight card (item 4) -----------
check('highlightCardRangeHtml(): renders BOTH the analyzed window and the highlight\'s own bounds when highlightBounds is present', () => {
  const w = { windowIndex: 1, scope: [5, 29], highlightBounds: [10, 25] };
  const html = run(`highlightCardRangeHtml(${JSON.stringify(w)})`);
  assert.match(html, />analyzed</, 'expected an "analyzed" caption next to the (expanded) scope pill');
  assert.match(html, />own bounds</, 'expected an "own bounds" caption next to the highlight\'s real [start_s,end_s] pill');
});
check('highlightCardRangeHtml(): falls back to the plain range pill when highlightBounds is absent (every OTHER pipeline\'s cards, unaffected)', () => {
  const w = { windowIndex: 1, scope: [5, 29] }; // no highlightBounds — e.g. a chunk-segment-tags card
  const html = run(`highlightCardRangeHtml(${JSON.stringify(w)})`);
  assert.doesNotMatch(html, />analyzed</, 'a card with no highlightBounds must render exactly like before (no new captions)');
  assert.doesNotMatch(html, />own bounds</);
});

// ---- Summary -----------------------------------------------------------------
console.log('');
if (failures > 0) {
  console.error(failures + ' check(s) FAILED');
  process.exit(1);
}
console.log('All checks passed.');
