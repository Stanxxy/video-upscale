#!/usr/bin/env node
// ============================================================================
// Deterministic headless-Node harness for highlight_eval.html's video-source
// fallback state machine (setSourceStatus/showFallbackVideoElement/
// loadMatchVideoWithFallback/triggerServerFallback) -- Item 2, AC2-3/AC2-8.
//
// Same technique as qa_client/tests/handle_pipeline_event.node-harness.mjs
// (INS-077): the real YouTube IFrame API's window.YT race makes a full
// browser the wrong tool for testing PURE state-transition logic that never
// actually needs a real player -- this harness runs the page's REAL inline
// <script> body (extracted verbatim) inside a Node `vm` context with a
// minimal inert DOM + fake YT.Player stub, bypassing the DOM/YT boundary.
//
// Run: node eval/adcc/predict/dashboard_client/tests/load_match_video_fallback.node-harness.mjs
// Exit code 0 = all checks passed; 1 = at least one failure (message printed).
// ============================================================================

import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const htmlPath = path.join(__dirname, '..', 'highlight_eval.html');
const html = fs.readFileSync(htmlPath, 'utf8');

const scriptMatch = html.match(/<script>([\s\S]*)<\/script>/);
assert.ok(scriptMatch, 'highlight_eval.html: could not locate the inline <script> block.');
const scriptBody = scriptMatch[1];

// ---- Minimal inert DOM stub -------------------------------------------------
const elementsById = new Map();
function makeElementStub(id) {
  return {
    id, value: '', textContent: '', innerHTML: '', className: '',
    style: {}, disabled: false, src: '',
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    appendChild() {}, querySelector() { return null; }, querySelectorAll() { return []; },
    addEventListener(evt, cb) {
      this._listeners = this._listeners || {};
      (this._listeners[evt] = this._listeners[evt] || []).push(cb);
    },
    fire(evt, arg) { ((this._listeners || {})[evt] || []).forEach((cb) => cb(arg)); },
    load() {}, play() { return Promise.resolve(); },
  };
}
const REQUIRED_IDS = [
  'match-select', 'match-load-note', 'offcorpus-url', 'btn-load-offcorpus', 'offcorpus-note',
  'yt-player', 'fallback-video', 'video-source-badge', 'btn-load-match-video', 'btn-preview',
  'preview-result', 'confirm-checkbox', 'btn-run', 'status-line', 'system-prompt', 'followup-prompt',
  'model-select', 'fps-select', 'frame-quality-select', 'refine-pad-before-input',
  'refine-pad-after-input', 'refine-fps-select', 'cap-input', 'thinking-toggle', 'refine-toggle',
  'refine-knobs', 'tau-t-presets', 'live-run-card', 'results-card', 'lr-settings-echo', 'lr-calls',
  'lr-dollars', 'lr-call-split', 'lr-phase', 'lr-phase-stat', 'lr-terminal-banner', 'call-list',
  'inspector-card', 'btn-stop', 'prf1-body', 'score-settings-caption', 'hl-rail-gold', 'hl-rail-pred',
  'hl-container', 'hl-connectors-svg', 'btn-rescore', 'tau-t-input', 'pv-calls',
  'pv-dollars-realistic', 'pv-window', 'pv-video-kind', 'pv-message', 'pv-typical-aside',
  'pv-long-window-note',
];
for (const id of REQUIRED_IDS) elementsById.set(id, makeElementStub(id));

const fakeDocument = {
  getElementById(id) {
    if (!elementsById.has(id)) elementsById.set(id, makeElementStub(id));
    return elementsById.get(id);
  },
  createElement() { return makeElementStub('__created__'); },
  querySelectorAll() { return []; },
  addEventListener() {},
};

// Fake YT.Player -- exposes addEventListener/loadVideoById/getPlayerState so
// createOrCuePlayer()'s existing code path constructs it without a real
// browser or network fetch of the IFrame API script.
let lastFakeYTPlayer = null;
class FakeYTPlayer {
  constructor(elId, opts) {
    this._listeners = {};
    this._opts = opts;
    lastFakeYTPlayer = this;
    setTimeout(() => opts.events.onReady && opts.events.onReady(), 0);
  }
  addEventListener(evt, cb) { (this._listeners[evt] = this._listeners[evt] || []).push(cb); }
  fire(evt, arg) { (this._listeners[evt] || []).forEach((cb) => cb(arg)); }
  loadVideoById() {}
  getCurrentTime() { return 0; }
  getPlayerState() { return -1; }
  seekTo() {}
  pauseVideo() {}
  playVideo() {}
}

const sandbox = {
  document: fakeDocument,
  window: {},
  console,
  setTimeout, clearTimeout, setInterval,
  URL,
  YT: { Player: FakeYTPlayer, PlayerState: { PLAYING: 1, BUFFERING: 3 } },
  fetch: async () => ({ ok: true, json: async () => ({ matches: [] }) }),
};
sandbox.window.onYouTubeIframeAPIReady = undefined;
vm.createContext(sandbox);
vm.runInContext(scriptBody, sandbox, { filename: 'highlight_eval.html:<script>' });

// The IIFE's `ytApiReady` starts false in the real page (waits for the real
// <script src=".../iframe_api"> to call window.onYouTubeIframeAPIReady) --
// simulate that call now so loadPlayer()/createOrCuePlayer() proceed
// synchronously-ish in this harness, matching a real "API already loaded"
// state.
sandbox.window.onYouTubeIframeAPIReady();

let failures = 0;
function check(name, fn) {
  try { fn(); console.log('  ✓', name); }
  catch (e) { failures += 1; console.error('  ✗', name, '\n   ', e.message); }
}

console.log('load_match_video_fallback.node-harness.mjs');

// 1. setSourceStatus / showFallbackVideoElement basics.
check('setSourceStatus(null) hides the badge', () => {
  sandbox.window.resetVideoSourceState();
  const badge = fakeDocument.getElementById('video-source-badge');
  assert.equal(badge.style.display, 'none');
});

// 2. triggerServerFallback drives preparing -> server on a synthetic canplay.
check('triggerServerFallback shows "Preparing server copy…" then "Server (fallback)" on canplay', () => {
  sandbox.window.resetVideoSourceState();
  const token = sandbox.window.getVideoLoadToken();
  sandbox.window.triggerServerFallback('m1', token);
  const badge = fakeDocument.getElementById('video-source-badge');
  assert.equal(badge.textContent, 'Preparing server copy…');
  const videoEl = fakeDocument.getElementById('fallback-video');
  assert.equal(videoEl.src, '/highlight/media/m1');
  videoEl.oncanplay();
  assert.equal(badge.textContent, 'Server (fallback)');
});

// 3. triggerServerFallback -> onerror drives "Failed".
check('triggerServerFallback shows "Failed" on a video element error', () => {
  sandbox.window.resetVideoSourceState();
  const token = sandbox.window.getVideoLoadToken();
  sandbox.window.triggerServerFallback('m1', token);
  const videoEl = fakeDocument.getElementById('fallback-video');
  videoEl.onerror();
  const badge = fakeDocument.getElementById('video-source-badge');
  assert.equal(badge.textContent, 'Failed');
});

// 4. Stale-token guard: an OLD triggerServerFallback callback must never
// clobber a NEWER load's UI state.
check('a stale token never overwrites a newer load\'s status', () => {
  sandbox.window.resetVideoSourceState(); // bump the token to a fresh value
  const staleToken = 0; // deliberately NOT the current token
  sandbox.window.triggerServerFallback('m1', staleToken);
  const badge = fakeDocument.getElementById('video-source-badge');
  assert.equal(badge.style.display, 'none'); // untouched -- the stale call was ignored
});

// 5. Product race (Task 17): after fallback, a late YT BUFFERING for the
// pre-fallback token must NOT flip the badge back to "YouTube".
check('late YT BUFFERING after triggerServerFallback does not set YouTube badge', () => {
  sandbox.window.resetVideoSourceState();
  lastFakeYTPlayer = null;
  sandbox.window.loadMatchVideoWithFallback('m1', 'dQw4w9WgXcQ');
  assert.ok(lastFakeYTPlayer, 'FakeYTPlayer was constructed');
  const ytToken = sandbox.window.getVideoLoadToken();

  // YT error → settle + triggerServerFallback (bumps videoLoadToken).
  lastFakeYTPlayer.fire('onError', { data: 2 });
  const badge = fakeDocument.getElementById('video-source-badge');
  assert.equal(badge.textContent, 'Preparing server copy…');
  fakeDocument.getElementById('fallback-video').onerror();
  assert.equal(badge.textContent, 'Failed');

  // Late YT event for the OLD load token (closed over by onStateChange).
  lastFakeYTPlayer.fire('onStateChange', { data: 3 }); // BUFFERING
  assert.equal(badge.textContent, 'Failed', 'late BUFFERING must not restore YouTube badge');
  assert.notEqual(sandbox.window.getVideoLoadToken(), ytToken, 'fallback must bump videoLoadToken');
});

console.log(failures > 0 ? `${failures} check(s) FAILED` : 'all checks passed');
// Force exit: loadMatchVideoWithFallback → YT onReady starts a setInterval poller.
process.exit(failures > 0 ? 1 : 0);
