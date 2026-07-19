#!/usr/bin/env node
// ============================================================================
// Deterministic headless-Node harness for vlm_studio.html's v2
// (`highlight-scan-critique-analyze`) pure-JS rendering/event logic.
//
// SAME technique as the sibling harness
// (handle_pipeline_event.node-harness.mjs, built for v1) — reused verbatim,
// not reinvented: extract the REAL inline <script> body from vlm_studio.html
// (not re-implemented/duplicated), run it in a Node `vm` sandbox with a
// minimal inert DOM stub, feed synthetic real-shaped SSE events (event
// shapes grepped from `service/pipelines/executors.py`'s `yield {...}`
// statements for `highlight_critique_result`/`highlight_start`/
// `highlight_result`) into `handlePipelineEvent()`, and assert on the
// resulting `runState`/rendered HTML. No browser, no network, no test
// framework/dependency (vanilla assert, per project convention).
//
// WHY a sibling file, not an extension of the v1 harness: the v1 harness is
// itself a REGRESSION test pinned to one specific v1 bug (its own file
// docstring says so) — bolting an unrelated v2 surface onto it would blur
// what regressed which. v2 gets its own file, same idiom.
//
// Motivation (evaluator MEDIUM gap, 2026-07-18 v2 convergence): the v2 UI
// (highlight_critique rendering, the "analyzed"/"ditched"/"errored"
// three-way distinction, the validator agree/disagree/ditch verdict
// summary, setV2GlobalDefault's cascade-without-clobber, renderCostBadge's
// N×(3+max_validator_iterations) formula) is evaluator-CONVERGED via live
// Gemini gates only — ZERO deterministic JS coverage existed for any of it.
// The ditch path in particular did NOT fire in any of the 3 live gate runs
// (rare-in-live), so it has NO coverage at all without this harness.
//
// Run: node qa_client/tests/handle_pipeline_event_v2.node-harness.mjs
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

// Strip the trailing "Init" bootstrap block — see the sibling v1 harness for
// the full rationale (this test has no reason to run page-bootstrap side
// effects; every function/variable it needs is declared BEFORE that marker).
const initMarkerIndex = scriptBody.indexOf('// Init\n// ============================================================\nsetMode(\'event\');');
assert.ok(
  initMarkerIndex !== -1,
  'vlm_studio.html: "// Init" bootstrap marker not found in its expected exact form — ' +
  'update this harness\'s truncation point (it intentionally does not run the page-bootstrap ' +
  'block, which does a live loadPipelines() fetch).',
);
scriptBody = scriptBody.slice(0, initMarkerIndex);

// ---- Minimal inert DOM stub (identical idiom to the sibling v1 harness) ---
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
    createElement() { return makeElementStub(); },
    body: makeElementStub(),
  },
  location: { search: '' },
  performance: { now: () => Date.now() },
  URLSearchParams,
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

console.log('handle_pipeline_event_v2.node-harness.mjs — highlight-scan-critique-analyze (v2) coverage\n');

// A real-shaped v2 PipelineDef fixture — field names/defaults transcribed
// from service/pipelines/models.py's HighlightScanConfig/
// HighlightCritiqueConfig/PositionAxisConfig/TechniqueAxisConfig/
// ValidatorAxisConfig/HighlightAnalyzeConfig (NOT re-derived/guessed).
// Every ThinkingQualityMixin field starts null EXCEPT
// ValidatorAxisConfig.media_resolution, which the backend itself defaults to
// "high" (LeCun's design-pass finding) — this is the one node the cascade
// test (item 4) must prove is never silently clobbered.
function freshV2PipelineDef(maxValidatorIterations) {
  return {
    id: 'highlight-scan-critique-analyze', label: 'Highlight scan + critique + analyze (native, v2)',
    stages: [
      { id: 'video_window', type: 'video_window', label: 'Video Window', enabled: true, config: {} },
      {
        id: 'highlight_scan', type: 'highlight_scan', label: 'Highlight Scan', enabled: true,
        config: {
          model: 'gemini-3.1-flash-lite', rough_fps: 1.0, min_highlight_s: 3.0, max_highlight_s: 30.0,
          system_prompt: null, initial_prompt: null, thinking: null, media_resolution: null,
        },
      },
      {
        id: 'highlight_critique', type: 'highlight_critique', label: 'Highlight Critique', enabled: true,
        config: { model: 'gemini-3.1-flash-lite', critique_backpad_s: 6.0, thinking: null, media_resolution: null },
      },
      {
        id: 'highlight_analyze', type: 'highlight_analyze', label: 'Highlight Analyze', enabled: true,
        config: {
          model: 'gemini-3.1-flash-lite', fps: 1, thinking: null, preroll_s: 5.0, postroll_s: 4.0,
          max_validator_iterations: maxValidatorIterations,
          position: { model: 'gemini-3.1-flash-lite', thinking: null, media_resolution: null },
          technique: { model: 'gemini-3.1-flash-lite', thinking: null, media_resolution: null },
          validator: { model: 'gemini-3.1-flash-lite', thinking: null, media_resolution: 'high' },
        },
      },
    ],
  };
}

// ============================================================================
// 1. highlight_critique_result rendering: corrected bounds APPLIED vs
//    movement_confirmed=false -> correction NOT applied (original kept) +
//    note shown. Real event shapes from executors.highlight_critique_node's
//    two `yield {"type": "highlight_critique_result", ...}` outcomes.
// ============================================================================
run("currentPipelineId = 'highlight-scan-critique-analyze'; currentPipelineDef = " + JSON.stringify(freshV2PipelineDef(1)) + "; resetRunView(); runState.status = 'running';");

run(`handlePipelineEvent({ type: 'highlight_map', highlights: [
  { index: 1, start_s: 20, end_s: 34, adjustment: null, start_is_synthetic: false, end_is_synthetic: false, description: 'guard pass attempt' },
  { index: 2, start_s: 50, end_s: 61, adjustment: null, start_is_synthetic: false, end_is_synthetic: false, description: 'scramble to back take' }
], timing: { model_ms: 900 } });`);

// 1a. Correction APPLIED (movement_confirmed=true, well-formed corrected_*).
run(`handlePipelineEvent({ type: 'highlight_critique_result', highlight_index: 1,
  movement_confirmed: true, corrected_start_s: 15, corrected_end_s: 34,
  critique_note: 'true setup starts earlier, at the grip fight', applied: true,
  timing: { model_ms: 700 } });`);
check('highlight_critique_result (applied): mutates the highlight with corrected_start_s/corrected_end_s', () => {
  assert.equal(run("runState.highlights.find(h => h.index === 1).corrected_start_s"), 15);
  assert.equal(run("runState.highlights.find(h => h.index === 1).corrected_end_s"), 34);
  assert.equal(run("runState.highlights.find(h => h.index === 1).movement_confirmed"), true);
});
check('highlightCardRangeHtmlV2 (applied): renders the struck scan pill + bold corrected pill, distinct states', () => {
  const html = run(`highlightCardRangeHtmlV2({ windowIndex: 1, scope: [10, 39] }, runState.highlights.find(h => h.index === 1))`);
  assert.match(html, /pill-struck/, 'expected the original scan bounds to render struck-through once a correction is applied');
  assert.match(html, />corrected</, 'expected the "corrected" badge');
  assert.doesNotMatch(html, /critique: not applied/, 'an APPLIED correction must not also show the not-applied note');
});
check('highlightRailRowHtmlV2 (applied): shows BOTH the struck scan time and the corrected time', () => {
  const html = run(`highlightRailRowHtmlV2(runState.highlights.find(h => h.index === 1))`);
  assert.match(html, /pill-struck/);
  assert.match(html, /corrected by highlight_critique/);
});

// 1b. movement_confirmed=false -> correction NOT applied, even though the
// executor docstring notes a real model CAN still emit well-formed
// corrected_start_s/corrected_end_s alongside movement_confirmed=false — the
// backend drops those fields in that case (highlight["corrected_start_s"] is
// never set), so the event this test sends carries corrected_start_s=null/
// corrected_end_s=null, exactly like the real backend would emit here.
run(`handlePipelineEvent({ type: 'highlight_critique_result', highlight_index: 2,
  movement_confirmed: false, corrected_start_s: null, corrected_end_s: null,
  critique_note: 'movement_confirmed=false — original bounds kept, correction not applied',
  applied: false, timing: { model_ms: 650 } });`);
check('highlight_critique_result (movement_confirmed=false): correction NOT applied — corrected_* stay null', () => {
  const h = 'runState.highlights.find(h => h.index === 2)';
  assert.equal(run(h + '.corrected_start_s'), null);
  assert.equal(run(h + '.corrected_end_s'), null);
  assert.equal(run(h + '.movement_confirmed'), false);
});
check('highlightCardRangeHtmlV2 (not applied): shows the original bounds + an honest "not applied" note, no fake correction', () => {
  const html = run(`highlightCardRangeHtmlV2({ windowIndex: 2, scope: [44, 66] }, runState.highlights.find(h => h.index === 2))`);
  assert.doesNotMatch(html, /pill-struck/, 'a not-applied critique must not render a struck-through pill (nothing was superseded)');
  assert.doesNotMatch(html, />corrected</, 'a not-applied critique must not render the "corrected" badge');
  assert.match(html, /critique: not applied/, 'expected the honest not-applied note');
});
check('highlightRailRowHtmlV2 (not applied): "critique: not applied" badge carries the critique_note as its title', () => {
  const html = run(`highlightRailRowHtmlV2(runState.highlights.find(h => h.index === 2))`);
  assert.doesNotMatch(html, /pill-struck/);
  assert.match(html, /critique: not applied/);
  assert.match(html, /movement_confirmed=false — original bounds kept, correction not applied/);
});

// ============================================================================
// 2. highlight_result v2 fields — status="analyzed" vs "ditched" vs a
//    transport `error` event. THE headline gap: ditched must render
//    rail-row-ditched (neutral), NEVER rail-row-errored (red); a transport
//    error must still render errored (red). Real event shapes from
//    executors.highlight_analyze_node's `yield {"type": "highlight_result",
//    ..., "status": status, "ditch_reason": ditch_reason, ...}` and the
//    pre-existing `yield {"type": "error", ..., "highlight_index": ...}`.
// ============================================================================
run("currentPipelineId = 'highlight-scan-critique-analyze'; currentPipelineDef = " + JSON.stringify(freshV2PipelineDef(1)) + "; resetRunView(); runState.status = 'running';");
run(`handlePipelineEvent({ type: 'highlight_map', highlights: [
  { index: 1, start_s: 5, end_s: 18, adjustment: null, start_is_synthetic: false, end_is_synthetic: false },
  { index: 2, start_s: 22, end_s: 40, adjustment: null, start_is_synthetic: false, end_is_synthetic: false },
  { index: 3, start_s: 45, end_s: 60, adjustment: null, start_is_synthetic: false, end_is_synthetic: false }
], timing: { model_ms: 800 } });`);
run(`handlePipelineEvent({ type: 'highlight_start', highlight_index: 1, scope: [0, 22], highlight_bounds: [5, 18], authoritative_bounds: [5, 18] });`);
run(`handlePipelineEvent({ type: 'highlight_start', highlight_index: 2, scope: [17, 44], highlight_bounds: [22, 40], authoritative_bounds: [22, 40] });`);
run(`handlePipelineEvent({ type: 'highlight_start', highlight_index: 3, scope: [40, 64], highlight_bounds: [45, 60], authoritative_bounds: [45, 60] });`);

// Highlight 1: status="analyzed" (the ordinary, successful case).
run(`handlePipelineEvent({ type: 'highlight_result', highlight_index: 1, format: 'simplified-tags-time-v1',
  clips: [{ start_s: 5, end_s: 18, position: 'guard', action_class: 'sweep', outcome: 'successful' }],
  status: 'analyzed', ditch_reason: null, validator_rounds: 1, verdict: 'agree',
  timing: { model_ms: 4200 }, axis_calls: {} });`);
check('highlight_result (status=analyzed): window marked complete, ditchStatus="analyzed", 1 clip', () => {
  assert.equal(run("runState.windows.find(w => w.windowIndex === 1).status"), 'complete');
  assert.equal(run("runState.windows.find(w => w.windowIndex === 1).ditchStatus"), 'analyzed');
  assert.equal(run("runState.windows.find(w => w.windowIndex === 1).clips.length"), 1);
});
check('windowCardHtmlV2 (analyzed): neither ditched nor errored styling; real event count badge', () => {
  const html = run(`windowCardHtmlV2(runState.windows.find(w => w.windowIndex === 1))`);
  assert.doesNotMatch(html, /rail-row-ditched/);
  assert.doesNotMatch(html, /window-card-errored/);
  assert.match(html, /1 event\(s\)/);
});

// Highlight 2: status="ditched" (a real, successful validator "ditch"
// verdict — build plan/backend contract: zero clips by design).
run(`handlePipelineEvent({ type: 'highlight_result', highlight_index: 2, format: 'simplified-tags-time-v1',
  clips: [], status: 'ditched', ditch_reason: 'validator invoked the strict ditch criterion',
  validator_rounds: 1, verdict: 'ditch', timing: { model_ms: 3900 }, axis_calls: {} });`);
check('highlight_result (status=ditched): window ditchStatus="ditched", ditch_reason captured, ZERO clips', () => {
  assert.equal(run("runState.windows.find(w => w.windowIndex === 2).ditchStatus"), 'ditched');
  assert.equal(run("runState.windows.find(w => w.windowIndex === 2).ditchReason"), 'validator invoked the strict ditch criterion');
  assert.equal(run("runState.windows.find(w => w.windowIndex === 2).clips.length"), 0);
  // Ditched is a COMPLETED run outcome, not a failure — status stays 'complete'.
  assert.equal(run("runState.windows.find(w => w.windowIndex === 2).status"), 'complete');
});
check('windowCardHtmlV2 (ditched): rail-row-ditched (neutral), NEVER window-card-errored (red); ditch_reason always-visible; no clip tags', () => {
  const html = run(`windowCardHtmlV2(runState.windows.find(w => w.windowIndex === 2))`);
  assert.match(html, /rail-row-ditched/, 'a ditched highlight must render the neutral ditched class');
  assert.doesNotMatch(html, /window-card-errored/, 'a ditched highlight must NOT render as an errored (red) card — ditch is not a transport failure');
  assert.doesNotMatch(html, /class="badge badge-b">ERRORED</, 'a ditched highlight must not show the ERRORED badge');
  assert.match(html, />ditched</, 'expected the neutral "ditched" status badge');
  assert.match(html, /Ditched by the validator: validator invoked the strict ditch criterion/);
});
check('highlightRailRowHtmlV2 (ditched): rail-row-ditched class, NOT rail-row-errored; "ditched" badge, not an event count', () => {
  const html = run(`highlightRailRowHtmlV2(runState.highlights.find(h => h.index === 2))`);
  assert.match(html, /class="rail-row rail-row-ditched"/, 'expected the exact neutral ditched row class');
  assert.doesNotMatch(html, /rail-row-errored/);
  assert.match(html, /class="badge badge-neutral">ditched</, 'expected the neutral ditched badge, not an "N tag(s)" count');
  assert.match(html, /Ditched: validator invoked the strict ditch criterion/);
});

// Highlight 3: a genuine PASS-2/validator TRANSPORT error (e.g. Gemini 500) —
// no highlight_result ever arrives for this highlight (the sequential loop
// `continue`s past it). This is the pre-existing v1 error-by-highlight_index
// path (executors.py's position/technique/validator call error branches all
// carry stage_id="highlight_analyze" + highlight_index), reused unchanged by
// v2 — must remain visually DISTINCT from the ditched case above.
run(`handlePipelineEvent({ type: 'error', stage_id: 'highlight_analyze', highlight_index: 3, message: 'validator call: Gemini 500 INTERNAL' });`);
check('error event with highlight_index (transport failure): window marked errored, NOT ditched', () => {
  assert.equal(run("runState.windows.find(w => w.windowIndex === 3).status"), 'errored');
  assert.equal(run("runState.windows.find(w => w.windowIndex === 3).ditchStatus"), undefined);
});
check('windowCardHtmlV2 (transport error): window-card-errored (red) + ERRORED badge, NEVER rail-row-ditched', () => {
  const html = run(`windowCardHtmlV2(runState.windows.find(w => w.windowIndex === 3))`);
  assert.match(html, /window-card-errored/, 'expected the RED errored card class for a genuine transport failure');
  assert.match(html, /class="badge badge-b">ERRORED</);
  assert.doesNotMatch(html, /rail-row-ditched/, 'a transport error must never render with the neutral ditched styling');
  assert.match(html, /PASS-2 failed: validator call: Gemini 500 INTERNAL/);
});
check('highlightRailRowHtmlV2 (transport error): rail-row-errored (RED), NEVER rail-row-ditched', () => {
  const html = run(`highlightRailRowHtmlV2(runState.highlights.find(h => h.index === 3))`);
  assert.match(html, /class="rail-row rail-row-errored"/);
  assert.doesNotMatch(html, /rail-row-ditched/);
  assert.match(html, /class="badge badge-b">errored</);
});

// ============================================================================
// 3. Validator verdict rendering — agree / disagree(wrong_label -> X) / ditch,
//    via the real v2ValidatorSummaryHtml() collapsible summary. Raw JSON
//    shapes transcribed from service/pipelines/highlight_axes.py's schemas
//    (adversarial_case/verdict/reason/evidence on the validator call;
//    wrong_label/final_position/final_action_class/final_outcome on disagree).
// ============================================================================
check('v2ValidatorSummaryHtml: verdict="agree" renders the single "agreed" outcome label', () => {
  const w = {
    verdict: 'agree', ditchStatus: 'analyzed', validatorRounds: 1,
    axisCalls: {
      validator: { raw_response_text: JSON.stringify({ verdict: 'agree', adversarial_case: 'is this really a sweep?', reason: null, evidence: 'hips clearly rotate under' }) },
      position: { raw_response_text: JSON.stringify({ position: 'guard', justification: 'both athletes on the mat, one in guard' }) },
      technique: { raw_response_text: JSON.stringify({ action_class: 'sweep', outcome: 'successful', justification: 'top position achieved' }) },
    },
  };
  const html = run(`v2ValidatorSummaryHtml(${JSON.stringify(w)})`);
  assert.match(html, /1 round\(s\) — agreed/);
  assert.doesNotMatch(html, /disagreed/);
  assert.doesNotMatch(html, />ditched</);
});
check('v2ValidatorSummaryHtml: verdict="disagree" renders exactly ONE corrected field (wrong_label -> final value)', () => {
  const w = {
    verdict: 'disagree', ditchStatus: 'analyzed', validatorRounds: 1,
    axisCalls: {
      validator: {
        raw_response_text: JSON.stringify({
          verdict: 'disagree', wrong_label: 'action_class', final_action_class: 'kimura',
          adversarial_case: 'could this be a kimura instead of an armbar?', reason: 'grip position matches kimura, not armbar',
          evidence: 'far-side wrist + same-side shoulder grip',
        }),
      },
      position: { raw_response_text: JSON.stringify({ position: 'side-control', justification: 'clear side control' }) },
      technique: { raw_response_text: JSON.stringify({ action_class: 'armbar', outcome: 'successful', justification: 'arm extended' }) },
    },
  };
  const html = run(`v2ValidatorSummaryHtml(${JSON.stringify(w)})`);
  assert.match(html, /1 round\(s\) — disagreed \(action_class → kimura\)/, 'expected the single-field disagree summary naming the corrected axis and its new value');
  assert.doesNotMatch(html, /\bagreed\b/, 'must not ALSO render the plain "agreed" outcome label (only "disagreed (...)")');
});
check('v2ValidatorSummaryHtml: ditchStatus="ditched" renders "ditched" regardless of the raw verdict text, expands to the transcript', () => {
  const w = {
    verdict: 'ditch', ditchStatus: 'ditched', validatorRounds: 1,
    axisCalls: {
      validator: { raw_response_text: JSON.stringify({ verdict: 'ditch', reason: 'no real BJJ technique visible in this window', adversarial_case: 'is anything happening here at all?', evidence: 'athletes standing, re-gripping' }) },
      position: { raw_response_text: JSON.stringify({ position: 'standing', justification: 'no ground engagement' }) },
      technique: { raw_response_text: JSON.stringify({ action_class: 'none', outcome: 'no_change', justification: 'no clear technique' }) },
    },
  };
  const html = run(`v2ValidatorSummaryHtml(${JSON.stringify(w)})`);
  assert.match(html, /1 round\(s\) — ditched/);
  assert.match(html, /no real BJJ technique visible in this window/, 'expected the ditch reason surfaced in the expanded transcript');
});

// ============================================================================
// 4. setV2GlobalDefault cascade-without-clobber — the load-bearing property:
//    changing the pipeline-level global default cascades into every
//    sub-config still following the OLD default, but must NEVER clobber a
//    node that has already diverged/is intentionally custom (ValidatorAxisConfig's
//    backend default media_resolution="high" is the canonical real example —
//    service/pipelines/models.py's own docstring calls this out by name).
// ============================================================================
run("currentPipelineId = 'highlight-scan-critique-analyze'; currentPipelineDef = " + JSON.stringify(freshV2PipelineDef(1)) + "; v2GlobalDefault = { thinking: 'off', media_resolution: 'low' };");
// Simulate the real on-load pass: fills every null field with the CURRENT
// global default (validator's media_resolution is already "high" — a
// non-null backend default — so it is left alone here, exactly like the real
// loadPipelineDefaults() flow).
run('applyV2GlobalDefaultToNulls();');
check('applyV2GlobalDefaultToNulls: nulled fields adopt the global default; validator\'s non-null "high" default is untouched', () => {
  assert.equal(run("v2QualityTarget('highlight_scan', null).media_resolution"), 'low');
  assert.equal(run("v2QualityTarget('highlight_analyze', 'position').media_resolution"), 'low');
  assert.equal(run("v2QualityTarget('highlight_analyze', 'validator').media_resolution"), 'high');
});

// Now the actual cascade-without-clobber under test: change the GLOBAL
// default. Every OTHER target (still following the old "low") must move to
// "medium"; validator (already diverged to "high", never following "low" to
// begin with) must NOT move.
run("setV2GlobalDefault('media_resolution', 'medium');");
check('setV2GlobalDefault cascades into sub-configs still following the OLD default', () => {
  assert.equal(run("v2QualityTarget('highlight_scan', null).media_resolution"), 'medium');
  assert.equal(run("v2QualityTarget('highlight_critique', null).media_resolution"), 'medium');
  assert.equal(run("v2QualityTarget('highlight_analyze', 'position').media_resolution"), 'medium');
  assert.equal(run("v2QualityTarget('highlight_analyze', 'technique').media_resolution"), 'medium');
});
check('setV2GlobalDefault NEVER clobbers a diverged/custom node (validator stays "high")', () => {
  assert.equal(run("v2QualityTarget('highlight_analyze', 'validator').media_resolution"), 'high');
});
check('v2IsCustom: validator (diverged) is flagged custom; position (followed the cascade) is not', () => {
  assert.equal(run("v2IsCustom(v2QualityTarget('highlight_analyze', 'validator'))"), true);
  assert.equal(run("v2IsCustom(v2QualityTarget('highlight_analyze', 'position'))"), false);
});
check('v2CustomBadgeHtml: renders the "custom" badge only for the diverged validator node', () => {
  assert.match(run("v2CustomBadgeHtml(v2QualityTarget('highlight_analyze', 'validator'))"), /badge-hint/);
  assert.equal(run("v2CustomBadgeHtml(v2QualityTarget('highlight_analyze', 'position'))"), '');
});

// ============================================================================
// 5. renderCostBadge formula — N × (3 + max_validator_iterations) + 1 scan
//    call (build plan UI item 6 / LeCun's cost model: 1 critique + 2 analyze
//    [position + technique] + L validator rounds, per highlight). Covers
//    BOTH the real-highlight-count path (post-scan) and the pre-scan
//    scope÷avg-highlight-length fallback.
// ============================================================================
run("currentPipelineId = 'highlight-scan-critique-analyze'; currentPipelineDef = " + JSON.stringify(freshV2PipelineDef(2)) + "; resetRunView();");
run(`handlePipelineEvent({ type: 'highlight_map', highlights: [
  { index: 1, start_s: 1, end_s: 5, adjustment: null, start_is_synthetic: false, end_is_synthetic: false },
  { index: 2, start_s: 6, end_s: 10, adjustment: null, start_is_synthetic: false, end_is_synthetic: false },
  { index: 3, start_s: 11, end_s: 15, adjustment: null, start_is_synthetic: false, end_is_synthetic: false },
  { index: 4, start_s: 16, end_s: 20, adjustment: null, start_is_synthetic: false, end_is_synthetic: false }
], timing: { model_ms: 500 } });`);
check('renderCostBadge (post-scan, real N): 4 highlights × (2 analyze + 2 validator iterations + 1 critique = 5) + 1 scan = 21', () => {
  run('renderCostBadge();');
  const badgeHtml = run("document.getElementById('cost-estimate-badge').innerHTML");
  assert.match(badgeHtml, /4 highlight\(s\) × ~5 calls ≈ 21 Gemini call\(s\)/);
});

// Pre-scan fallback: no highlights yet, only a run-scope duration. min/max
// highlight length come off the SAME fixture's highlight_scan stage
// (3.0s/30.0s -> avg 16.5s); scope 0:00→2:00 (120s) -> round(120/16.5) = 7.
run("currentPipelineId = 'highlight-scan-critique-analyze'; currentPipelineDef = " + JSON.stringify(freshV2PipelineDef(2)) + "; resetRunView();");
run("document.getElementById('run-start').value = '0:00'; document.getElementById('run-end').value = '2:00';");
check('renderCostBadge (pre-scan fallback): estimates N from scope ÷ avg highlight length, still applies the same per-highlight factor', () => {
  run('renderCostBadge();');
  const badgeHtml = run("document.getElementById('cost-estimate-badge').innerHTML");
  assert.match(badgeHtml, /~7 \(estimated from scope ÷ avg highlight length\)/);
  assert.match(badgeHtml, /~5 calls ≈ 36 Gemini call\(s\)/, 'expected 1 + 7*5 = 36');
});

// Hidden entirely for every OTHER pipeline (build plan: never shown outside
// highlight-scan-critique-analyze).
check('renderCostBadge: hidden (display:none) for a different pipeline id', () => {
  run("currentPipelineId = 'highlight-scan-analyze';");
  run('renderCostBadge();');
  assert.equal(run("document.getElementById('cost-estimate-badge').style.display"), 'none');
});

// ---- Summary -----------------------------------------------------------------
console.log('');
if (failures > 0) {
  console.error(failures + ' check(s) FAILED');
  process.exit(1);
}
console.log('All checks passed.');
