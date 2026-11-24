export function setupPageContents(article, sidebar, onNavigate) {
  if (!sidebar) return;
  const nav = sidebar.querySelector('.page-toc-links');
  const links = [...nav.querySelectorAll('[data-toc-link]')];
  const entries = links.map(link => ({ link, heading: document.getElementById(link.dataset.tocLink) }))
    .filter(entry => entry.heading && article.contains(entry.heading));
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let current = null;
  let queued = false;

  function update() {
    queued = false;
    const top = article.getBoundingClientRect().top + 28;
    const visible = entries.filter(entry => entry.heading.getClientRects().length);
    let selected = visible[0];
    for (const entry of visible) {
      if (entry.heading.getBoundingClientRect().top <= top) selected = entry;
      else break;
    }
    if (article.scrollTop > 0 && article.scrollTop + article.clientHeight >= article.scrollHeight - 2) {
      selected = visible.at(-1);
    }
    if (!selected) return;
    if (current !== selected.link) {
      for (const { link } of entries) link.removeAttribute('aria-current');
      selected.link.setAttribute('aria-current', 'location');
      current = selected.link;
    }
    const item = current.getBoundingClientRect(), box = nav.getBoundingClientRect();
    if (item.top < box.top) nav.scrollTop += item.top - box.top;
    else if (item.bottom > box.bottom) nav.scrollTop += item.bottom - box.bottom;
  }

  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(update);
  }

  function moveTo(target, behavior = 'instant') {
    const ancestors = [];
    for (let parent = target.parentElement; parent && parent !== article; parent = parent.parentElement) {
      if (parent.tagName === 'DETAILS' && !parent.open) ancestors.push(parent);
    }
    for (const details of ancestors.reverse()) details.open = true;
    const top = target.getBoundingClientRect().top - article.getBoundingClientRect().top + article.scrollTop - 20;
    article.scrollTo({ top: Math.max(0, top), behavior });
    schedule();
  }

  function followHash() {
    if (!location.hash) article.scrollTo({ top: 0, behavior: 'instant' });
    else {
      let id;
      try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
      const target = document.getElementById(id);
      if (target && article.contains(target)) moveTo(target);
    }
    schedule();
  }

  nav.addEventListener('click', event => {
    const link = event.target.closest('[data-toc-link]');
    if (!link || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    const target = document.getElementById(link.dataset.tocLink);
    if (!target || !article.contains(target)) return;
    event.preventDefault();
    const url = new URL(location.href);
    url.hash = target.id;
    if (url.href !== location.href) history.pushState(null, '', url);
    onNavigate();
    article.focus({ preventScroll: true });
    moveTo(target, reduceMotion.matches ? 'instant' : 'smooth');
  });
  article.addEventListener('scroll', schedule, { passive: true });
  article.addEventListener('toggle', schedule, true);
  const resize = new ResizeObserver(schedule);
  resize.observe(article);
  resize.observe(nav);
  for (const child of article.children) resize.observe(child);
  window.addEventListener('hashchange', followHash);
  window.addEventListener('popstate', followHash);
  document.fonts.ready.then(followHash);
  followHash();
}
