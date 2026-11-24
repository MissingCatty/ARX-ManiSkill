import { setupPageContents } from './page-toc.mjs';

const configurationIndex=document.querySelector('[data-configuration-routes]');
if(configurationIndex&&location.hash){
  let id;
  try{id=decodeURIComponent(location.hash.slice(1));}catch{}
  const target=JSON.parse(configurationIndex.dataset.configurationRoutes)[id];
  if(target)location.replace(target);
}

const siteHeader = document.querySelector('.site-header');
if (siteHeader) {
  const updateHeaderHeight = () => document.documentElement.style.setProperty(
    '--site-header-height', `${siteHeader.getBoundingClientRect().height}px`);
  updateHeaderHeight();
  new ResizeObserver(updateHeaderHeight).observe(siteHeader);
}

const input = document.querySelector('#case-search');
const outcome = document.querySelector('#case-outcome');
if (input && outcome) {
  const rows = [...document.querySelectorAll('[data-case-row]')];
  const count = document.querySelector('#case-count');
  const empty = document.querySelector('#empty-cases');
  const refresh = () => {
    const query = input.value.trim().toLowerCase();
    let visible = 0;
    for (const row of rows) {
      row.hidden = !row.textContent.toLowerCase().includes(query)
        || (outcome.value !== '' && row.dataset.group !== outcome.value);
      visible += row.hidden ? 0 : 1;
    }
    count.textContent = `${visible} / ${rows.length} 个场景`;
    empty.hidden = visible !== 0;
  };
  input.addEventListener('input', refresh);
  outcome.addEventListener('change', refresh);
}

const paperSearch = document.querySelector('#paper-search');
if (paperSearch) {
  const moduleSelect = document.querySelector('#paper-module');
  const statusSelect = document.querySelector('#paper-status');
  const cards = [...document.querySelectorAll('[data-paper-card]')];
  const count = document.querySelector('#paper-count');
  const empty = document.querySelector('#paper-empty');
  const refreshPapers = () => {
    const query = paperSearch.value.trim().toLowerCase();
    let visible = 0;
    for (const card of cards) {
      card.hidden = !card.dataset.search.includes(query)
        || (moduleSelect.value !== '' && !card.dataset.modules.split(' ').includes(moduleSelect.value))
        || (statusSelect.value !== '' && card.dataset.status !== statusSelect.value);
      visible += !card.hidden;
    }
    count.textContent = `${visible} / ${cards.length} 篇论文`;
    empty.hidden = visible !== 0;
  };
  paperSearch.addEventListener('input', refreshPapers);
  moduleSelect.addEventListener('change', refreshPapers);
  statusSelect.addEventListener('change', refreshPapers);
}

if (document.body.classList.contains('reading-page')) {
  const main = document.querySelector('#main');
  const article = main.querySelector('.article, .paper-article');
  article.id = 'reading-content';
  article.tabIndex = 0;
  article.setAttribute('aria-label', '正文内容');
  document.querySelector('.skip').href = '#reading-content';
  const narrow = window.matchMedia('(max-width: 1000px)');
  const toolbar = document.createElement('nav');
  toolbar.className = 'reading-toolbar';
  toolbar.setAttribute('aria-label', '阅读工具');
  const panels = [
    [main.querySelector('.version-sidebar, .paper-sidebar'), '本页目录'],
    [main.querySelector('.version-todos'), '待办事项'],
  ].filter(([panel]) => panel);
  let active = null;
  const close = (restoreFocus = false) => {
    if (!active) return;
    const previous = active;
    previous.panel.classList.remove('reading-panel-open');
    previous.button.setAttribute('aria-expanded', 'false');
    active = null;
    if (restoreFocus) previous.button.focus({ preventScroll: true });
  };
  for (const [panel, title] of panels) {
    if (!panel.id) panel.id = 'reading-navigation';
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = title;
    button.setAttribute('aria-label', title);
    button.setAttribute('aria-controls', panel.id);
    button.setAttribute('aria-expanded', 'false');
    const heading = document.createElement('div');
    heading.className = 'reading-panel-heading';
    const label = document.createElement('strong');
    label.textContent = title;
    const dismiss = document.createElement('button');
    dismiss.type = 'button';
    dismiss.textContent = '关闭 ×';
    dismiss.setAttribute('aria-label', '关闭' + title);
    dismiss.addEventListener('click', () => close(true));
    heading.append(label, dismiss);
    panel.prepend(heading);
    button.addEventListener('click', () => {
      const wasOpen = active?.panel === panel;
      close();
      if (wasOpen) return;
      panel.classList.add('reading-panel-open');
      button.setAttribute('aria-expanded', 'true');
      active = { panel, button };
      dismiss.focus({ preventScroll: true });
    });
    toolbar.append(button);
  }
  main.prepend(toolbar);
  document.addEventListener('pointerdown', event => {
    if (active && !active.panel.contains(event.target) && !toolbar.contains(event.target)) close();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && active) { event.preventDefault(); close(true); }
  });
  setupPageContents(article, main.querySelector('[data-page-toc]'), () => close());
  const followToolHash = () => {
    if (narrow.matches && location.hash === '#todos' && active?.panel.id !== 'todos') {
      toolbar.querySelector('button[aria-controls="todos"]')?.click();
    }
  };
  window.addEventListener('hashchange', followToolHash);
  narrow.addEventListener('change', () => { close(); followToolHash(); });
  followToolHash();
}
