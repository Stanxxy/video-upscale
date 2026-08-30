#!/usr/bin/env node
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(testDir, '..', 'vlm_studio.html'), 'utf8');
const match = html.match(/<script>([\s\S]*)<\/script>/);
assert.ok(match, 'inline script is present');
const marker = match[1].indexOf("// Init\n// ============================================================\nsetMode('event');");
assert.notEqual(marker, -1, 'Init marker is present');

const elements = new Map();
let canvasDrawArgs = null;
function stub(tagName = '') {
  const element = {
    value: '', textContent: '', innerHTML: '', className: '', style: {},
    scrollTop: 0, scrollHeight: 0, clientHeight: 0,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    querySelectorAll() { return []; }, querySelector() { return null; },
    getAttribute() { return null; }, setAttribute() {}, removeAttribute(name) { delete this[name]; },
    addEventListener() {}, appendChild() {},
    getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 50 }; },
  };
  if (tagName === 'canvas') {
    element.getContext = () => ({ drawImage(...args) { canvasDrawArgs = args; } });
    element.toDataURL = () => 'data:image/jpeg;base64,Y3JvcA==';
  }
  return element;
}
function byId(id) {
  if (!elements.has(id)) elements.set(id, stub());
  return elements.get(id);
}
const sandbox = {
  console,
  document: {
    getElementById: byId, querySelectorAll() { return []; }, querySelector() { return null; },
    addEventListener() {}, createElement(tagName) { return stub(tagName); }, body: stub(),
  },
  location: { search: '' }, performance: { now: () => Date.now() }, URLSearchParams,
  fetch: async () => { throw new Error('network disabled'); },
  setInterval: () => 0, clearInterval() {}, setTimeout: () => 0, clearTimeout() {},
};
sandbox.window = sandbox;
const context = vm.createContext(sandbox);
vm.runInContext(match[1].slice(0, marker), context, { filename: 'vlm_studio.html' });
const run = code => vm.runInContext(code, context);

assert.match(
  run("POSITION_TIMELINE_DEFAULT.stages.find(stage => stage.id === 'position_scan').config.system_prompt"),
  /random gym/,
);
assert.match(
  run("POSITION_TIMELINE_DEFAULT.stages.find(stage => stage.id === 'position_scan').config.initial_prompt"),
  /\$start_sec.*\$end_sec/s,
);
assert.match(
  run("POSITION_TIMELINE_DEFAULT.stages.find(stage => stage.id === 'position_critique').config.system_prompt"),
  /verify one proposed boundary/i,
);
assert.match(
  run("POSITION_TIMELINE_DEFAULT.stages.find(stage => stage.id === 'position_critique').config.initial_prompt"),
  /\$proposed_at_s.*\$state_before_json.*\$state_after_json/s,
);
assert.match(
  run("POSITION_TIMELINE_DEFAULT.stages.find(stage => stage.id === 'highlight_critique').config.system_prompt"),
  /correcting timestamp drift/i,
);
assert.match(
  run("POSITION_TIMELINE_DEFAULT.stages.find(stage => stage.id === 'highlight_critique').config.initial_prompt"),
  /\$window_start_sec.*\$description.*\$position_context_json/s,
);
assert.equal(
  run("POSITION_TIMELINE_DEFAULT.stages.find(stage => stage.id === 'highlight_critique').config.fps"),
  1,
);
assert.deepEqual(
  JSON.parse(run("JSON.stringify(POSITION_TIMELINE_DEFAULT.stages.find(stage => stage.id === 'position_technique_candidates').config)")),
  { transition_pre_s: 4, transition_post_s: 2, merge_tolerance_s: 2 },
);
assert.ok(
  run("POSITION_TIMELINE_DEFAULT.stages.findIndex(stage => stage.id === 'position_technique_candidates')") <
  run("POSITION_TIMELINE_DEFAULT.stages.findIndex(stage => stage.id === 'highlight_critique')"),
);

run("currentPipelineId = 'position-timeline-techniques'; currentPipelineDef = JSON.parse(JSON.stringify(POSITION_TIMELINE_DEFAULT));");
run('renderDag()');
assert.equal(byId('dag-nodes').className, 'dag-row dag-row-wrapped');
assert.match(
  html,
  /@media \(min-width: 1100px\)[\s\S]*\.dag-row-wrapped[\s\S]*grid-template-columns:\s*repeat\(4,\s*minmax\(0,\s*1fr\)\)/,
);
run("currentPipelineDef = { id: 'compact', label: 'Compact', stages: POSITION_TIMELINE_DEFAULT.stages.slice(0, 6) }; renderDag();");
assert.equal(byId('dag-nodes').className, 'dag-row');
run("currentPipelineDef = JSON.parse(JSON.stringify(POSITION_TIMELINE_DEFAULT)); renderDag();");
const techniqueScanConfigHtml = run(
  "nodeConfigFieldsHtml(currentPipelineDef.stages.find(stage => stage.id === 'highlight_scan'))",
);
assert.match(techniqueScanConfigHtml, /v2-thinking-quality-row/);
assert.match(techniqueScanConfigHtml, />Thinking</);
assert.match(techniqueScanConfigHtml, />Quality</);
assert.match(techniqueScanConfigHtml, /setNodeConfigValue\('thinking', 'high'\)/);
assert.match(techniqueScanConfigHtml, /setNodeConfigValue\('media_resolution', 'high'\)/);
assert.doesNotMatch(techniqueScanConfigHtml, /<label>thinking<\/label>/);
assert.doesNotMatch(techniqueScanConfigHtml, /<label>media_resolution<\/label>/);

const positionCritiqueConfigHtml = run(
  "nodeConfigFieldsHtml(POSITION_TIMELINE_DEFAULT.stages.find(stage => stage.id === 'position_critique'))",
);
assert.match(positionCritiqueConfigHtml, /System prompt \(Position Critique\)/);
assert.match(positionCritiqueConfigHtml, /Initial prompt template \(Position Critique\)/);
assert.match(positionCritiqueConfigHtml, /\$state_before_json/);
assert.match(positionCritiqueConfigHtml, /aria-label="System prompt \(Position Critique\)"/);
assert.match(positionCritiqueConfigHtml, /aria-label="Initial prompt template \(Position Critique\)"/);

const techniqueCritiqueConfigHtml = run(
  "nodeConfigFieldsHtml(POSITION_TIMELINE_DEFAULT.stages.find(stage => stage.id === 'highlight_critique'))",
);
assert.match(techniqueCritiqueConfigHtml, /System prompt \(Technique Critique\)/);
assert.match(techniqueCritiqueConfigHtml, /Initial prompt template \(Technique Critique\)/);
assert.match(techniqueCritiqueConfigHtml, /\$critique_forwardpad_s/);
assert.match(techniqueCritiqueConfigHtml, /\$position_context_json/);
assert.match(techniqueCritiqueConfigHtml, /FPS/);
assert.match(techniqueCritiqueConfigHtml, /aria-label="System prompt \(Technique Critique\)"/);

const candidateConfigHtml = run(
  "nodeConfigFieldsHtml(POSITION_TIMELINE_DEFAULT.stages.find(stage => stage.id === 'position_technique_candidates'))",
);
assert.match(candidateConfigHtml, /Transition look-back/);
assert.match(candidateConfigHtml, /Transition look-forward/);
assert.match(candidateConfigHtml, /Scan merge tolerance/);

run(`currentPipelineDef = JSON.parse(JSON.stringify(POSITION_TIMELINE_DEFAULT));
selectedNodeIndex = currentPipelineDef.stages.findIndex(stage => stage.id === 'position_critique');
onNodePromptInput('system_prompt', 'EDITED POSITION SYSTEM');`);
assert.equal(
  run("currentPipelineDef.stages.find(stage => stage.id === 'position_critique').config.system_prompt"),
  'EDITED POSITION SYSTEM',
);
run(`selectedNodeIndex = currentPipelineDef.stages.findIndex(stage => stage.id === 'highlight_critique');
onNodePromptInput('initial_prompt', 'EDITED $description TEMPLATE');
setNodeConfigValue('fps', 15);`);
assert.equal(
  run("currentPipelineDef.stages.find(stage => stage.id === 'highlight_critique').config.initial_prompt"),
  'EDITED $description TEMPLATE',
);
assert.equal(
  run("currentPipelineDef.stages.find(stage => stage.id === 'highlight_critique').config.fps"),
  15,
);
run('selectedNodeIndex = null;');

byId('ref-a-id').value = 'player_a';
byId('ref-a-name').value = 'Avery';
byId('ref-b-id').value = 'player_b';
byId('ref-b-name').value = 'Blake';

context.fetch = async (url, options) => {
  assert.match(String(url), /\/qa\/reference-frame$/);
  assert.deepEqual(JSON.parse(options.body), { youtube_id: 'abcdefghijk', timestamp_s: 12.5 });
  return {
    ok: true,
    async json() {
      return {
        frame_data_url: 'data:image/jpeg;base64,ZnJhbWU=',
        captured_at_ms: 12500,
        width: 200,
        height: 100,
      };
    },
  };
};
run(`currentVideoId = 'abcdefghijk'; currentTime = 12.5; ytReady = true;
ytPlayer = { getCurrentTime() { return 12.5; }, pauseVideo() {} };`);
await run("beginReferenceCapture('a')");
assert.equal(byId('reference-crop-modal').hidden, false);
assert.equal(run('activeReferenceSlot'), 'a');
byId('reference-crop-frame').naturalWidth = 200;
byId('reference-crop-frame').naturalHeight = 100;
run("adjustReferenceCropWithKeyboard({ key: 'Enter', shiftKey: false, preventDefault() {} });");
assert.deepEqual(
  JSON.parse(run('JSON.stringify(referenceCropBox)')),
  { x: 25, y: 10, w: 50, h: 30 },
);
run('referenceCropBox = { x: 10, y: 10, w: 50, h: 30 }; confirmReferenceCrop();');
assert.equal(byId('reference-crop-modal').hidden, true);
assert.deepEqual(
  JSON.parse(run('JSON.stringify(playerReferenceCaptures.a.boundingBox)')),
  { x: 20, y: 20, w: 100, h: 60 },
);
assert.equal(run('playerReferenceCaptures.a.imageBase64'), 'data:image/jpeg;base64,Y3JvcA==');
assert.ok(canvasDrawArgs, 'the frozen source image is cropped through canvas');

run(`playerReferenceCaptures.a = {
  imageBase64: 'data:image/jpeg;base64,YQ==',
  boundingBox: { x: 10, y: 20, w: 30, h: 40 },
  capturedAtMs: 12500,
};
playerReferenceCaptures.b = {
  imageBase64: 'data:image/jpeg;base64,Yg==',
  boundingBox: { x: 50, y: 60, w: 70, h: 80 },
  capturedAtMs: 12600,
};`);

run("currentPipelineId = 'position-timeline-techniques'; resetRunView();");
run(`handlePipelineEvent(${JSON.stringify({
  type: 'position_map',
  initial_state: { position: 'standing' },
  transitions: [{ index: 0, at_s: 10 }],
})})`);
run(`handlePipelineEvent(${JSON.stringify({
  type: 'position_critique_result', transition_index: 0, confirmed: true,
  corrected_at_s: 9.5, state_after: { position: 'half_guard' }, critique_note: 'leg trapped',
})})`);
run(`handlePipelineEvent(${JSON.stringify({
  type: 'technique_candidate_map', scan_candidate_count: 1, transition_candidate_count: 1,
  merged_candidate_count: 1, final_candidate_count: 1,
  candidates: [{ index: 0, start_s: 12, end_s: 15,
    candidate_sources: ['technique_scan', 'position_transition'],
    position_context: { position_before: 'half_guard', position_after: 'other',
      position_transitions: [{ at_s: 14, position_before: 'half_guard', position_after: 'other' }] } }],
})})`);
run(`handlePipelineEvent(${JSON.stringify({
  type: 'technique_result', highlight_index: 0, status: 'analyzed', ditch_reason: null,
  event: { start_s: 12, end_s: 15, action_class: 'guard_pass', specific_technique_guess: 'knee cut', actor_player_id: 'player_a', outcome: 'unsuccessful', confidence: 0.8, evidence: 'knee drives inside', candidate_sources: ['technique_scan', 'position_transition'], position_context: { position_before: 'half_guard', position_after: 'other' } },
})})`);
run(`handlePipelineEvent(${JSON.stringify({
  type: 'timeline_result', player_ids: ['player_a', 'player_b'],
  position_intervals: [
    { start_s: 0, end_s: 9.5, duration_s: 9.5, position: 'standing', player_roles: { player_a: 'neutral', player_b: 'neutral' }, position_note: '', inactive_reason: null },
    { start_s: 9.5, end_s: 20, duration_s: 10.5, position: 'other', player_roles: { player_a: 'top', player_b: 'bottom' }, position_note: 'headquarters: one leg pinned between the knees', inactive_reason: null },
  ],
  technique_events: [{ start_s: 12, end_s: 15, action_class: 'guard_pass', specific_technique_guess: 'knee cut', actor_player_id: 'player_a', outcome: 'unsuccessful', confidence: 0.8, evidence: 'knee drives inside', position: 'other', position_note: 'headquarters: one leg pinned between the knees', candidate_sources: ['technique_scan', 'position_transition'], position_before: 'half_guard', position_after: 'other' }],
  statistics: {
    active_duration_s: 20, inactive_duration_s: 0,
    by_position: { standing: { duration_s: 9.5 }, other: { duration_s: 10.5 } },
    by_player: {
      player_a: { role_totals: { top_s: 10.5, bottom_s: 0, neutral_s: 9.5 }, by_position: {} },
      player_b: { role_totals: { top_s: 0, bottom_s: 10.5, neutral_s: 9.5 }, by_position: {} },
    },
    techniques_by_player: { player_a: 1 },
  },
})})`);
run(`handlePipelineEvent(${JSON.stringify({
  type: 'run_complete', clips: [], format: 'position-technique-timeline-v1',
  node_costs: [
    { stage_id: 'video_window', label: 'Video Window', stage_type: 'video_window', model: null, call_count: 0, prompt_token_count: 0, candidates_token_count: 0, thoughts_token_count: 0, total_token_count: 0, estimated_usd: 0, pricing_status: 'not_applicable' },
    { stage_id: 'position_scan', label: 'Position Scan', stage_type: 'position_scan', model: 'gemini-3.7-flash', call_count: 1, prompt_token_count: 1200, cached_content_token_count: 300, candidates_token_count: 80, thoughts_token_count: 220, total_token_count: 1500, estimated_usd: 0.002025, pricing_status: 'priced' },
  ],
  cost_summary: { call_count: 1, estimated_usd: 0.002025, pricing_tier: 'standard_paid', pricing_source_url: 'https://ai.google.dev/gemini-api/docs/pricing', rate_valid_through: '2026-12-31' },
  per_stage_timing: {}, timing: { total_ms: 1000 }, dedup_applied: false, dedup_skip_reason: null, degraded_counts: {},
})})`);

assert.equal(run('runState.positionTransitions[0].corrected_at_s'), 9.5);
assert.equal(run('runState.highlights.length'), 1);
assert.equal(run('runState.candidateSummary.merged_candidate_count'), 1);
assert.equal(run('runState.techniqueEvents.length'), 1);
assert.equal(byId('timeline-results-card').style.display, 'block');
assert.match(byId('position-timeline-body').innerHTML, /position-segment/);
assert.match(byId('position-timeline-body').innerHTML, /technique-marker/);
assert.match(byId('position-statistics-body').innerHTML, /Objective player totals/);
assert.match(byId('position-statistics-body').innerHTML, /Player time by position and role/);
assert.match(byId('position-statistics-body').innerHTML, /Position interval details/);
assert.match(byId('position-statistics-body').innerHTML, /headquarters: one leg pinned/);
assert.match(byId('position-statistics-body').innerHTML, /1 scan candidate/);
assert.match(byId('position-statistics-body').innerHTML, /1 active transition/);
assert.match(byId('position-statistics-body').innerHTML, /1 merged/);
assert.match(byId('position-statistics-body').innerHTML, /1 final/);
assert.match(byId('position-statistics-body').innerHTML, /Technique Scan \+ Position transition/);
assert.match(byId('position-statistics-body').innerHTML, /half guard.*other/);
assert.doesNotMatch(byId('position-statistics-body').innerHTML.toLowerCase(), /dominan/);
assert.equal(run('runState.nodeCosts.length'), 2);
assert.match(byId('position-statistics-body').innerHTML, /DAG node cost/);
assert.match(byId('position-statistics-body').innerHTML, /Position Scan/);
assert.match(byId('position-statistics-body').innerHTML, /Cached input/);
assert.match(byId('position-statistics-body').innerHTML, />300</);
assert.match(byId('position-statistics-body').innerHTML, /\$0\.002025/);
assert.match(byId('position-statistics-body').innerHTML, /Standard paid-tier estimate/);
assert.match(byId('dedup-rail-list').innerHTML, /knee cut/);
assert.deepEqual(
  JSON.parse(run('JSON.stringify(collectPlayerReferences()[0])')),
  {
    player_id: 'player_a', player_name: 'Avery',
    image_base64: 'data:image/jpeg;base64,YQ==',
    bounding_box: { x: 10, y: 20, w: 30, h: 40 },
    captured_at_ms: 12500,
  },
);
assert.doesNotMatch(html, /id="ref-[ab]-url"/);
assert.match(html, /Capture current frame/);
assert.match(html, /id="reference-crop-modal"/);
assert.deepEqual(
  JSON.parse(run('JSON.stringify(sourceCropFromDisplayBox({x:25,y:10,w:50,h:40},100,50,1920,1080))')),
  { x: 480, y: 216, w: 960, h: 864 },
);

run('sleepAbortable = async function() { return; }; resetRunView();');
await run(`mockPipelineRunStream({
  youtube_id: 'abcdefghijk', youtube_url: 'https://www.youtube.com/watch?v=abcdefghijk',
  scope: { type: 'range', start_sec: 0, end_sec: 30 },
  pipeline: JSON.parse(JSON.stringify(POSITION_TIMELINE_DEFAULT)),
  player_references: collectPlayerReferences()
}, null, handlePipelineEvent)`);
assert.equal(run('runState.sawRunComplete'), true);
assert.equal(run('runState.positionIntervals.length'), 4);
assert.equal(run('runState.statistics.techniques_by_player.player_b'), 0);
assert.match(byId('position-statistics-body').innerHTML, /north-south with player B circling/);

byId('ref-b-id').value = 'player_a';
assert.equal(run('validatePlayerReferences().valid'), false);

console.log('position timeline UI harness: all checks passed');
