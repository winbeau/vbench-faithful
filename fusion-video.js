// Load near the viewport once. Scrolling never changes the user's playback intent.
(() => {
  'use strict';
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const pauseIcon = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M4 2v10M10 2v10" fill="none" stroke="currentColor" stroke-width="2"/></svg>';
  const playIcon = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M4 2l8 5-8 5z" fill="currentColor"/></svg>';
  const groups = [...document.querySelectorAll('.semantic-pair')].map(element => ({
    element, videos: [...element.querySelectorAll('video[data-src]')],
    buttons: [...element.querySelectorAll('.study-playback')],
    activated: false, wantsPlay: !reducedMotion, starting: false,
    attempt: 0, startedAt: 0, blocked: false,
  })).filter(group => group.videos.length);

  function updateButtons(group) {
    const failed = group.videos.some(video => video.error);
    const paused = !group.wantsPlay || group.blocked || failed;
    const label = failed ? 'Retry both videos' : paused ? 'Play both videos' : 'Pause both videos';
    group.element.classList.toggle('is-paused', paused);
    group.element.dataset.playback = failed ? 'error' : paused ? 'paused' :
      group.videos.some(video => video.paused || video.readyState < 3) ? 'loading' : 'playing';
    group.buttons.forEach(button => {
      button.setAttribute('aria-label', label);
      button.title = label;
      button.innerHTML = paused ? playIcon : pauseIcon;
    });
  }
  function stop(group) {
    // Invalidate an unfinished play promise; it must not undo a later user action.
    group.attempt += 1;
    group.starting = false;
    group.videos.forEach(video => video.pause());
    updateButtons(group);
  }
  function load(group) {
    if (group.activated) return;
    group.activated = true;
    group.videos.forEach(video => {
      video.muted = true;
      video.preload = 'auto';
      video.src = video.dataset.src;
      video.load();
    });
  }
  function align(group) {
    const [leader, ...others] = group.videos;
    if (leader.seeking || leader.readyState < 3) return;
    others.forEach(video => {
      if (video.seeking || video.readyState < 3) return;
      const delta = Math.abs(video.currentTime - leader.currentTime);
      const circularDelta = Number.isFinite(leader.duration) ? Math.min(delta, Math.abs(leader.duration-delta)) : delta;
      if (circularDelta > 0.10) video.currentTime = leader.currentTime;
    });
  }
  function reconcile(group) {
    if (!group.activated || !group.wantsPlay || group.blocked || document.hidden ||
        group.videos.some(video => video.error || video.readyState < 2)) return;
    if (group.starting) return;
    if (group.videos.every(video => !video.paused)) return;
    group.starting = true;
    group.startedAt = performance.now();
    const attempt = ++group.attempt;
    align(group);
    Promise.allSettled(group.videos.map(video => video.play())).then(results => {
      if (attempt !== group.attempt) return;
      group.starting = false;
      if (results.some(result => result.status === 'rejected' && result.reason?.name === 'NotAllowedError')) {
        group.blocked = true;
        stop(group);
      } else if (!group.wantsPlay || document.hidden) stop(group);
      updateButtons(group);
      // AbortError/temporary buffering keeps playback intent and is retried by media events/watchdog.
    });
  }
  const byElement = new Map(groups.map(group => [group.element, group]));
  const loader = new IntersectionObserver(entries => entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const group = byElement.get(entry.target);
    load(group);
    reconcile(group);
    loader.unobserve(entry.target);
  }), {rootMargin:'240px'});
  groups.forEach(group => {
    group.videos.forEach(video => {
      ['loadeddata', 'canplay', 'playing', 'seeked', 'pause', 'stalled'].forEach(event => {
        video.addEventListener(event, () => { updateButtons(group); reconcile(group); });
      });
      // Never pause a pair on "waiting": doing so can consume the only canplay event
      // while a previous play promise is still pending and strand both videos.
      video.addEventListener('waiting', () => updateButtons(group));
      video.addEventListener('error', () => updateButtons(group));
    });
    group.buttons.forEach(button => button.addEventListener('click', () => {
      const retry = group.blocked || group.videos.some(video => video.error);
      group.wantsPlay = retry || !group.wantsPlay;
      group.blocked = false;
      if (!group.wantsPlay) { stop(group); return; }
      load(group);
      group.videos.forEach(video => { if (video.error) video.load(); });
      reconcile(group);
      updateButtons(group);
    }));
    updateButtons(group);
    loader.observe(group.element);
  });
  setInterval(() => groups.forEach(group => {
    if (!group.activated || !group.wantsPlay || document.hidden) return;
    if (group.starting && performance.now()-group.startedAt > 3000) {
      group.attempt += 1;
      group.starting = false;
    }
    reconcile(group);
    if (!group.videos.some(video => video.paused)) align(group);
  }), 300);

  const heroes = [...document.querySelectorAll('video[data-hero-src]')];
  const heroLoader = new IntersectionObserver(entries => entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const video = entry.target;
    video.muted = true;
    video.src = video.dataset.heroSrc;
    if (!reducedMotion && !document.hidden) video.play().catch(() => {});
    heroLoader.unobserve(video);
  }), {rootMargin:'120px'});
  heroes.forEach(video => heroLoader.observe(video));
  document.addEventListener('visibilitychange', () => {
    groups.forEach(group => document.hidden ? stop(group) : reconcile(group));
    heroes.forEach(video => {
      if (document.hidden) video.pause();
      else if (!reducedMotion && video.getAttribute('src')) video.play().catch(() => {});
    });
  });
})();
