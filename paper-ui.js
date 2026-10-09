(() => {
  'use strict';
  const figure = document.querySelector('#figure-dialog');
  const trigger = document.querySelector('[data-open-figure]');
  const toast = document.querySelector('#toast');
  let timer;
  trigger.addEventListener('click', () => figure.showModal());
  figure.querySelector('[data-close]').addEventListener('click', () => figure.close());
  figure.addEventListener('click', event => {
    if (event.target !== figure) return;
    const bounds = figure.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right ||
        event.clientY < bounds.top || event.clientY > bounds.bottom) figure.close();
  });
  figure.addEventListener('close', () => trigger.focus({preventScroll: true}));
  document.querySelector('[data-copy-bib]').addEventListener('click', async () => {
    const citation = document.querySelector('#bibtex-code').textContent;
    let copied = false;
    try {
      await navigator.clipboard.writeText(citation);
      copied = true;
    } catch {
      const field = document.createElement('textarea');
      field.value = citation;
      field.style.cssText = 'position:fixed;opacity:0';
      document.body.append(field);
      field.select();
      copied = document.execCommand('copy');
      field.remove();
    }
    toast.textContent = copied ? 'BibTeX copied' : 'Select the citation to copy it.';
    toast.classList.add('visible');
    clearTimeout(timer);
    timer = setTimeout(() => toast.classList.remove('visible'), 2400);
  });
})();
