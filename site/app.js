'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHTML = value => String(value).replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
const number = value => value === null ? 'N/A' : Number(value).toFixed(4);
const signed = value => `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(value !== 0 && Math.abs(value) < .0001 ? 6 : 4)}`;
const icon = name => `<svg viewBox="0 0 24 24" aria-hidden="true">${{
  play: '<path d="m8 5 11 7-11 7z"/>',
  pause: '<path d="M8 5v14M16 5v14"/>',
  replay: '<path d="M4 10a8 8 0 1 1 1 7M4 4v6h6"/>',
  intervention: '<path d="m3 12 5-5m-5 5 5 5m-5-5h18M17 7l4 5-4 5"/>',
}[name]}</svg>`;

const examples = {
  scene: {
    metric: 'Scene', original: 'coast-base', counterfactual: 'coast-base',
    query: ['a street', 'a road'], labels: ['Original query', 'Synonymous query'],
    intervention: 'Same pixels · equivalent wording',
    expectation: 'Expected: a stable score under equivalent scene wording.',
  },
  subject: {
    metric: 'Subject Consistency', original: 'coast-base', counterfactual: 'coast-background-blur',
    query: ['Target: the car', 'Target: the same car'], labels: ['Original scene', 'Background blurred · first 1.25 s'],
    intervention: 'Upper background only · car preserved',
    expectation: 'Expected: subject consistency stays stable when only the background changes.',
  },
  spatial: {
    metric: 'Spatial Relationship', original: 'coast-base', counterfactual: 'coast-mirror',
    query: ['a car on the left of a bicycle', 'a car on the left of a bicycle'], labels: ['Original scene', 'Horizontally mirrored'],
    intervention: 'Same query · reversed spatial relation',
    expectation: 'Expected: a lower score after the requested left–right relation is reversed.',
  },
};

function scoreCard(label, kind, values) {
  const valid = values.every(value => value !== null);
  const delta = valid ? `Δ ${signed(values[1] - values[0])}` : 'Incomplete pair';
  const chart = valid ? `<div class="score-track" aria-hidden="true"><span class="score-path" style="left:${Math.min(...values) * 100}%;width:${Math.abs(values[1] - values[0]) * 100}%"></span><span class="score-point before" style="left:${values[0] * 100}%"></span><span class="score-point after" style="left:${values[1] * 100}%"></span></div><div class="score-labels" aria-hidden="true"><span>0</span><span>Score</span><span>1</span></div>` : '';
  return `<div class="score-card ${kind}"><div class="score-heading"><span><i class="dot ${kind}"></i>${label}</span><span class="score-delta">${delta}</span></div><div class="score-values"><div><p class="score-value">${number(values[0])}</p><p class="score-state">Original</p></div><span class="score-arrow" aria-hidden="true">→</span><div class="score-after"><p class="score-value">${number(values[1])}</p><p class="score-state">Counterfactual</p></div></div>${chart}</div>`;
}

const players = [];

function mountPlayer(root, key) {
  const videos = $$('video', root);
  const playButton = $('.play-button', root);
  const timeline = $('.timeline', root);
  const time = $('.time-display', root);
  let loaded = false;
  let playing = false;
  let wasUserPaused = false;
  let animationFrame = 0;
  let visible = false;
  let pendingTime = 0;
  let operation = 0;
  const duration = () => Number.isFinite(videos[0].duration) ? videos[0].duration : 121 / 24;

  function setButton() {
    playButton.innerHTML = `${icon(playing ? 'pause' : 'play')}${playing ? 'Pause pair' : 'Play pair'}`;
    playButton.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} both ${examples[key].metric} videos`);
    playButton.setAttribute('aria-pressed', String(playing));
  }

  function load() {
    if (loaded) return;
    loaded = true;
    videos.forEach(video => {video.src = video.dataset.src; video.load();});
  }

  function seek(seconds) {
    pendingTime = Math.max(0, Math.min(seconds, duration()));
    videos.forEach(video => {if (video.readyState >= 1) video.currentTime = pendingTime;});
    updateProgress();
  }

  function updateProgress() {
    const current = videos[0].readyState >= 1 ? videos[0].currentTime : pendingTime;
    timeline.value = String(Math.min(1000, Math.round(current / duration() * 1000)));
    timeline.setAttribute('aria-valuetext', `${current.toFixed(1)} of ${duration().toFixed(2)} seconds`);
    time.textContent = `${current.toFixed(1)} / ${duration().toFixed(2)} s`;
  }

  function tick() {
    if (!playing) return;
    // One clock for each pair; drift correction also handles buffering.
    const master = videos[0];
    if (Math.abs(videos[1].currentTime - master.currentTime) > .1 && videos[1].readyState >= 2) videos[1].currentTime = master.currentTime;
    updateProgress();
    animationFrame = requestAnimationFrame(tick);
  }

  function pause(user = false) {
    operation++;
    playing = false;
    if (user) wasUserPaused = true;
    cancelAnimationFrame(animationFrame);
    videos.forEach(video => video.pause());
    setButton();
  }

  async function play(user = false) {
    if (playing) return;
    load();
    if (user) wasUserPaused = false;
    players.forEach(player => {if (player.root !== root) player.pause();});
    const token = ++operation;
    playing = true;
    setButton();
    if (videos[0].ended) seek(0);
    if (videos[1].readyState >= 1) videos[1].currentTime = videos[0].currentTime;
    const results = await Promise.allSettled(videos.map(video => video.play()));
    if (token !== operation) return;
    if (results.some(result => result.status === 'rejected')) {pause(); return;}
    setButton();
    cancelAnimationFrame(animationFrame);
    tick();
  }

  playButton.addEventListener('click', () => playing ? pause(true) : play(true));
  $('.replay-button', root).addEventListener('click', () => {load(); seek(0); play(true);});
  timeline.addEventListener('input', () => {load(); seek(Number(timeline.value) / 1000 * duration());});
  videos.forEach(video => {
    video.addEventListener('loadedmetadata', () => {video.currentTime = pendingTime; updateProgress();});
    video.addEventListener('error', () => {
      pause();
      if ($('.video-error', video.parentElement)) return;
      const error = document.createElement('div');
      error.className = 'video-error';
      error.innerHTML = `This video could not load.<a href="${escapeHTML(video.dataset.src)}">Open the MP4 directly ↗</a>`;
      video.parentElement.append(error);
    });
  });
  videos[0].addEventListener('ended', () => {if (playing && visible) {seek(0); play();} else pause();});
  const player = {root, pause, play, load};
  players.push(player);
  const preloadObserver = new IntersectionObserver(entries => {if (entries.some(entry => entry.isIntersecting)) {load(); preloadObserver.disconnect();}}, {rootMargin: '450px'});
  preloadObserver.observe(root);
  const motionReduced = matchMedia('(prefers-reduced-motion: reduce)');
  const observer = new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    if (!visible) pause();
    else if (!motionReduced.matches && !navigator.connection?.saveData && !wasUserPaused && !document.hidden) play();
  }, {threshold: .5});
  observer.observe($('.video-pair', root));
  motionReduced.addEventListener('change', () => {if (motionReduced.matches) pause();});
  setButton(); updateProgress();
}

function mountExample(key, score) {
  const example = examples[key];
  const root = $(`[data-case="${key}"]`);
  const panel = (name, index) => `<figure class="video-panel"><div class="video-frame"><video muted playsinline preload="none" poster="assets/${name}.webp" data-src="assets/${name}.mp4" aria-label="${escapeHTML(`${example.metric}: ${example.labels[index]}`)}"></video><span class="video-badge ${index ? 'cf' : ''}">${index ? 'Counterfactual' : 'Original'}</span></div><figcaption class="video-caption"><q>${escapeHTML(example.query[index])}</q><span>${escapeHTML(example.labels[index])}</span></figcaption></figure>`;
  root.innerHTML = `<div class="demo-toolbar"><span class="demo-metric">${example.metric}</span><span class="demo-intervention">${icon('intervention')}${example.intervention}</span></div><div class="video-pair">${panel(example.original, 0)}${panel(example.counterfactual, 1)}</div><div class="transport"><button class="play-button" type="button" aria-pressed="false">${icon('play')}Play pair</button><button class="replay-button" type="button" aria-label="Replay both ${example.metric} videos from the start">${icon('replay')}</button><input class="timeline" type="range" min="0" max="1000" value="0" step="1" aria-label="${example.metric} synchronized video position"><output class="time-display">0.0 / 5.04 s</output><span class="sync-label">Synchronized playback</span></div><div class="score-region" aria-label="${example.metric} per-video score comparison">${scoreCard('VBench 1.0', 'origin', score.origin)}${scoreCard('Ours', 'repair', score.repair)}</div><p class="case-verdict"><strong>${escapeHTML(example.expectation)}</strong> ${escapeHTML(score.observation)}</p><p class="score-note">Per-video measurements for this LTX pair. ${escapeHTML(score.note || '')}</p>`;
  mountPlayer(root, key);
}

function mountResults(data) {
  const tabs = $$('[data-results]');
  function select(kind, focus = false) {
    tabs.forEach(tab => {
      const active = tab.dataset.results === kind;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      if (active && focus) tab.focus();
    });
    $('#results-panel').setAttribute('aria-labelledby', `tab-${kind}`);
    $('#result-explainer').textContent = data[kind].description;
    $('#table-note').textContent = data[kind].note;
    $('#result-rows').innerHTML = data[kind].rows.map(row => `<tr><td>${escapeHTML(row.dimension)}</td><td>${number(row.origin[0])}<span class="table-arrow">→</span>${number(row.origin[1])}</td><td>${number(row.repair[0])}<span class="table-arrow">→</span>${number(row.repair[1])}</td><td>${escapeHTML(row.pairs)}</td></tr>`).join('');
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => select(tab.dataset.results));
    tab.addEventListener('keydown', event => {
      if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      select(tabs[next].dataset.results, true);
    });
  });
  select('invariance');
}

async function readData(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Cannot load ${path}: ${response.status}`);
  return response.json();
}

async function init() {
  const tasks = await Promise.allSettled([readData('data/demo-scores.json'), readData('data/paper-results.json')]);
  if (tasks[0].status === 'fulfilled') {
    const data = tasks[0].value;
    Object.keys(examples).forEach(key => mountExample(key, data.cases[key]));
    $('#inference-provenance').textContent = data.provenance.summary;
  } else {
    $$('.demo').forEach(root => {root.innerHTML = '<p class="load-error">The demonstration data could not load. Please reload this page, or <a href="data/demo-scores.json">open the score data</a>.</p>';});
  }
  if (tasks[1].status === 'fulfilled') mountResults(tasks[1].value);
  else $('#result-explainer').innerHTML = 'The table could not load. <a href="data/paper-results.json">Download its source data</a> or read the paper.';
}

document.addEventListener('visibilitychange', () => {if (document.hidden) players.forEach(player => player.pause());});
const dialog = $('#figure-dialog');
$('[data-open-figure]').addEventListener('click', () => dialog.showModal());
$('.dialog-close', dialog).addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {if (event.target === dialog) dialog.close();});
$('#copy-citation').addEventListener('click', async () => {
  const text = $('#bibtex').textContent;
  try {
    await navigator.clipboard.writeText(text);
    $('#copy-citation').innerHTML = 'Copied <span aria-hidden="true">✓</span>';
    $('#copy-status').textContent = 'BibTeX copied to clipboard.';
    setTimeout(() => {$('#copy-citation').innerHTML = 'Copy BibTeX <span aria-hidden="true">⧉</span>';}, 2500);
  } catch {
    const range = document.createRange(); range.selectNodeContents($('#bibtex'));
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    $('#copy-status').textContent = 'Citation selected. Press Control+C or Command+C to copy.';
  }
});
const chapters = $$('.chapter[id]');
const chapterLinks = $$('.chapter-nav a');
const chapterObserver = new IntersectionObserver(entries => {
  const entry = entries.filter(item => item.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
  if (!entry) return;
  chapterLinks.forEach(link => {
    const active = link.hash === `#${entry.target.id}`;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
  });
}, {rootMargin: '-10% 0px -45% 0px', threshold: [0, .1, .3]});
chapters.forEach(chapter => chapterObserver.observe(chapter));
init();
