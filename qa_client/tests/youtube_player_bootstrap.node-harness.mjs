#!/usr/bin/env node
// Deterministic contract tests for VLM Studio's YouTube IFrame API bootstrap.
// Runs the real inline page script with an inert DOM and a stub YT.Player.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(__dirname, '..', 'vlm_studio.html'), 'utf8');
const scriptMatch = html.match(/<script>([\s\S]*)<\/script>/);
assert.ok(scriptMatch, 'could not locate VLM Studio inline script');
const initMarker = '// Init\n// ============================================================\nsetMode(\'event\');';
const initIndex = scriptMatch[1].indexOf(initMarker);
assert.notEqual(initIndex, -1, 'could not locate VLM Studio Init marker');
const scriptBody = scriptMatch[1].slice(0, initIndex);

const elements = new Map();
function elementStub() {
  return {
    value: '', textContent: '', innerHTML: '', className: '', disabled: false,
    style: {}, children: [], scrollTop: 0, scrollHeight: 0, clientHeight: 0,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    appendChild(child) { this.children.push(child); },
    addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; },
    getAttribute() { return null; }, setAttribute() {}, removeAttribute() {},
  };
}
function getElementById(id) {
  if (!elements.has(id)) elements.set(id, elementStub());
  return elements.get(id);
}

const injectedScripts = [];
const documentStub = {
  body: elementStub(),
  head: {
    appendChild(script) { injectedScripts.push(script); },
  },
  createElement(tagName) {
    return { ...elementStub(), tagName: tagName.toUpperCase(), dataset: {} };
  },
  getElementById,
  querySelector(selector) {
    if (selector === 'script[data-vlm-youtube-api]') {
      return injectedScripts.find(script => script.dataset?.vlmYoutubeApi) || null;
    }
    return null;
  },
  querySelectorAll() { return []; },
  addEventListener() {},
};

const sandbox = {
  console,
  document: documentStub,
  location: {
    search: '',
    origin: 'http://localhost:8765',
    href: 'http://localhost:8765/vlm_studio.html',
  },
  URL,
  URLSearchParams,
  performance: { now: () => Date.now() },
  fetch: async () => { throw new Error('network disabled in bootstrap harness'); },
  setInterval: () => 0,
  clearInterval() {},
  setTimeout: () => 1,
  clearTimeout() {},
};
sandbox.window = sandbox;
const context = vm.createContext(sandbox);
vm.runInContext(scriptBody, context, { filename: 'vlm_studio.html' });
const run = code => vm.runInContext(code, context);

let failures = 0;
function check(name, fn) {
  try {
    fn();
    console.log('  ok   - ' + name);
  } catch (error) {
    failures += 1;
    console.error('  FAIL - ' + name + ': ' + error.message);
  }
}

console.log('youtube_player_bootstrap.node-harness.mjs — YouTube load recovery\n');

check('the YouTube API is injected only after the inline callbacks are registered', () => {
  assert.doesNotMatch(html, /<script[^>]+src=["']https:\/\/www\.youtube\.com\/iframe_api/);
});

run("document.getElementById('yt-input').value = 'https://youtu.be/y2hXwEib0qY'; loadVideo();");

check('youtu.be input queues the exact video id while the API is loading', () => {
  assert.equal(run('pendingLoadId'), 'y2hXwEib0qY');
  assert.equal(run('currentVideoId'), 'y2hXwEib0qY');
});

check('Load injects one identifiable YouTube API script', () => {
  assert.equal(injectedScripts.length, 1);
  assert.equal(injectedScripts[0].src, 'https://www.youtube.com/iframe_api');
  assert.equal(injectedScripts[0].dataset.vlmYoutubeApi, 'true');
});

let createdPlayer = null;
sandbox.YT = {
  Player: function Player(target, options) {
    createdPlayer = { target, options };
    return {
      loadVideoById() {},
      getDuration() { return 438; },
      getPlayerState() { return 5; },
    };
  },
};
run('onYouTubeIframeAPIReady()');

check('API readiness consumes the queued id and creates the player', () => {
  assert.equal(run('pendingLoadId'), null);
  assert.equal(createdPlayer.target, 'yt-player');
  assert.equal(createdPlayer.options.videoId, 'y2hXwEib0qY');
});

check('player configuration identifies the page origin and reports errors', () => {
  assert.equal(createdPlayer.options.playerVars.origin, 'http://localhost:8765');
  assert.equal(createdPlayer.options.events.onError, run('onPlayerError'));
});

run("ytApiReady = false; ytPlayer = null; pendingLoadId = null; document.getElementById('yt-input').value = 'y2hXwEib0qY'; loadVideo();");
check('Load recovers when YT.Player exists but its one-shot callback was missed', () => {
  assert.equal(run('ytApiReady'), true);
  assert.ok(createdPlayer);
});

run('onPlayerError({ data: 153 })');
check('YouTube error 153 becomes visible and actionable', () => {
  const message = getElementById('yt-error').textContent;
  assert.match(message, /153/);
  assert.match(message, /Referer|origin/i);
  assert.equal(getElementById('yt-error').style.display, 'block');
});

if (failures) {
  console.error(`\n${failures} check(s) failed.`);
  process.exitCode = 1;
} else {
  console.log('\nAll checks passed.');
}
