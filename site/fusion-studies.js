(() => {
  'use strict';
  const $ = selector => document.querySelector(selector);
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const semantics = window.SEMANTIC_CASE_PLAN.cases;
  const semanticDimensions = [...new Set(semantics.map(c => c.dimension))];
  const nuisances = window.NUISANCE_CASE_PLAN;
  const spatials = window.SPATIAL_CASE_PLAN;

  function chapterHead(panel, name) {
    return `<header class="study-chapter-head"><h2>${name}</h2></header>`;
  }
  function visual(image, side, mirrored, label, overlay='', single=false, position='42%', media=null) {
    if (media?.src) {
      return `<div class="study-frame is-video"${media.poster?` style="background-image:url('${escape(media.poster)}');background-size:cover;background-position:center"`:""}><video class="study-frame-video" data-src="${escape(media.src)}" ${media.poster?`poster="${escape(media.poster)}"`:''} muted playsinline loop preload="none" aria-label="${escape(label)}"></video>${overlay}<button class="study-playback" type="button" aria-label="Pause both videos" title="Pause both videos"><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M4 2v10M10 2v10" stroke="currentColor" stroke-width="2"/></svg></button></div>`;
    }
    return `<div class="study-frame"><div class="study-frame-art${mirrored?' is-mirrored':''}" style="background-image:url('${image}');background-position:${single?'center':side==='right'?'right':'left'} ${position};${single?'background-size:cover;':''}" role="img" aria-label="${escape(label)}"></div>${overlay}</div>`;
  }
  function semanticScores(study, side) {
    return `<div class="semantic-scores" aria-label="Scores for this ${side?'counterfactual':'original'} case">${[['VBench','vbench'],['Ours','ours']].map(([name,kind])=>{
      const value = study.scores?.[kind]?.[side];
      const status = study.score_status?.[kind]?.[side];
      const missing = status && status !== 'succeeded';
      const reason = study.score_reasons?.[kind]?.[side] || 'The evaluator returned no score for this input.';
      const scope = study.score_scope ? ` ${study.score_scope}` : '';
      const title = Number.isFinite(value) ? 'Measured score for this case.' + scope : missing ? reason : 'This new case has not been scored yet.';
      return `<div class="semantic-score is-${kind}" title="${escape(title)}"><span>${name}</span><span class="semantic-score-value">${Number.isFinite(value)?value.toFixed(4):missing?'N/A':'—'}</span></div>`;
    }).join('')}</div>`;
  }
  function semanticPair(study) {
    const excerpt = study.excerpt;
    return `<div class="semantic-pair">${study.views.map((view,i)=>{
      const label = i?'Counterfactual':'Original';
      const value = (study.video || study.videos)?.[i?'counterfactual':'original'];
      const media = typeof value === 'string' ? {src:value} : value;
      const overlay = `${!media?.src && view.preview==='jitter'?`<canvas class="nuisance-jitter" data-jitter-source="${escape(view.image)}" data-ready="false" aria-label="Still-image illustration of 8 pixel local jitter: ${escape(study.scene)}" hidden></canvas>`:''}<span class="semantic-condition">${label}</span>`;
      return `<figure class="semantic-view" aria-label="${label}: ${escape(study.scene)}">${visual(view.image,view.side,Boolean(view.mirrored),`${label}: ${study.scene}`,overlay + (study.guide ? spatialGuide(study,view) : ''),view.layout==='single',view.position,media)}${semanticScores(study,i)}<figcaption class="semantic-prompt">… ${escape(excerpt.before)}<mark>${escape(excerpt.focus[i])}</mark>${escape(excerpt.after)} …</figcaption></figure>`;
    }).join('')}</div>`;
  }
  $('#chapters').innerHTML = `
    <section class="study-chapter study-semantic" id="layer-1"><div class="wrap">${chapterHead('a','Target Substitution')}<div class="semantic-sheet" id="semantic-study"></div></div></section>
    <section class="study-chapter study-nuisance" id="layer-2"><div class="wrap">${chapterHead('b','Nuisance Entanglement')}<div id="nuisance-study"></div></div></section>
    <section class="study-chapter study-spatial" id="layer-3"><div class="wrap">${chapterHead('c','Evidence Collapse')}<div id="spatial-study"></div></div></section>`;

  function renderSemantic() {
    $('#semantic-study').innerHTML = semanticDimensions.map((dimension,i)=>`<section class="semantic-column" aria-labelledby="semantic-dimension-${i}"><header class="semantic-dimension"><h3 id="semantic-dimension-${i}">${escape(dimension)}</h3><span aria-hidden="true">${String(i+1).padStart(2,'0')}</span></header>${semantics.filter(study=>study.dimension===dimension).map(study=>`<article class="semantic-case" id="case-${study.id}" aria-label="${escape(study.scene)}">${semanticPair(study)}</article>`).join('')}</section>`).join('');
  }
  function renderNuisance() {
    const card = study => `<article class="semantic-case nuisance-case" id="case-${study.id}" aria-label="${escape(study.scene)}">${semanticPair(study)}</article>`;
    $('#nuisance-study').innerHTML = nuisances.dimensions.map((dimension,i)=>`<section class="nuisance-row${i===0?' nuisance-native-square':''}" id="${i===0?'dynamic-jitter':i===1?'subject-consistency':'background-consistency'}" aria-labelledby="nuisance-dimension-${i}"><header class="nuisance-dimension"><h3 id="nuisance-dimension-${i}">${escape(dimension.name)}</h3><span class="nuisance-heading-rule" aria-hidden="true"></span><p>${escape(dimension.note)}</p></header><div class="nuisance-grid">${nuisances.cases.filter(study=>study.dimension===dimension.name).map(card).join('')}</div>${i===0 && nuisances.static_controls?.length?`<div class="static-controls"><div class="static-control-heading"><h4>Static controls</h4><p>One frozen frame. The same 8 px jitter.</p></div><div class="nuisance-grid">${nuisances.static_controls.map(card).join('')}<aside class="static-control-note"><p>The original stays completely still.</p><p>These repeated-frame controls isolate the effect of jitter. All candidates remain visible, including those that leave the score unchanged.</p><small>Both scores use the same first 2 seconds of each 5-second clip.</small></aside></div></div>`:''}</section>`).join('');
  }
  function spatialGuide(study, view) {
    const guide = study.guide;
    let subjectX, subjectY, objectX, objectY, direction, path;
    if (guide.axis === 'y') {
      const swapped = view.side === 'right';
      subjectX = objectX = guide.x;
      subjectY = swapped ? guide.object_y : guide.subject_y;
      objectY = swapped ? guide.subject_y : guide.object_y;
      const sign = subjectY < objectY ? -1 : 1;
      const start = objectY + sign*9, end = subjectY - sign*9;
      direction = sign < 0 ? 'above' : 'below';
      path = `M ${guide.x} ${start} V ${end} M ${guide.x-2} ${end-sign*3} L ${guide.x} ${end} L ${guide.x+2} ${end-sign*3}`;
    } else {
      subjectX = view.mirrored ? 100-guide.subject_x : guide.subject_x;
      objectX = view.mirrored ? 100-guide.object_x : guide.object_x;
      subjectY = objectY = guide.y;
      const sign = subjectX < objectX ? -1 : 1;
      const start = objectX + sign*11, end = subjectX - sign*11;
      direction = sign < 0 ? 'left' : 'right';
      path = `M ${start} ${guide.y} H ${end} M ${end-sign*3} ${guide.y-3} L ${end} ${guide.y} L ${end-sign*3} ${guide.y+3}`;
    }
    return `<div class="spatial-direction-guide" aria-hidden="true" data-direction="${direction}"><svg viewBox="0 0 100 100" preserveAspectRatio="none" focusable="false"><path d="${path}"/></svg><span class="spatial-anchor is-subject" style="left:${subjectX}%;top:${subjectY}%">${escape(study.target.subject)}</span><span class="spatial-anchor is-object" style="left:${objectX}%;top:${objectY}%">${escape(study.target.object)}</span></div>`;
  }
  function renderSpatial() {
    $('#spatial-study').innerHTML = `<section class="spatial-row is-guided" aria-labelledby="spatial-dimension">
      <header class="nuisance-dimension spatial-dimension"><h3 id="spatial-dimension">Spatial Relationship</h3><span class="nuisance-heading-rule" aria-hidden="true"></span><p>${escape(spatials.note)}</p><button class="nuisance-preview-toggle spatial-guide-toggle" id="spatial-guide-toggle" type="button" aria-pressed="true" aria-controls="spatial-gallery" title="Toggle schematic direction guides; these are not model detections."><span aria-hidden="true">↔</span> Hide direction</button></header>
      <div class="nuisance-grid spatial-grid" id="spatial-gallery">${spatials.cases.map(study=>`<article class="semantic-case nuisance-case spatial-case" id="case-${study.id}" aria-label="${escape(study.scene)}">${semanticPair(study)}</article>`).join('')}</div>
    </section>`;
    $('#spatial-guide-toggle').addEventListener('click', event => {
      const button = event.currentTarget;
      const active = button.getAttribute('aria-pressed') !== 'true';
      button.setAttribute('aria-pressed', String(active));
      button.innerHTML = `<span aria-hidden="true">↔</span> ${active?'Hide':'Show'} direction`;
      $('.spatial-row').classList.toggle('is-guided', active);
    });
  }
  renderSemantic();
  renderNuisance();
  renderSpatial();
})();
