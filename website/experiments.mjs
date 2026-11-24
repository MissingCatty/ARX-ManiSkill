import { readFileSync } from 'node:fs';
import { join } from 'node:path';

export const versionUrl = (id, code) => `/iterations/${id}/versions/${code}/`;
export const batchUrl = (id, code, batch) => `${versionUrl(id, code)}batches/${batch}/`;

export function readExperiments(root, registered) {
  const catalog = JSON.parse(readFileSync(join(root, '内容/experiments.json'), 'utf8'));
  if (catalog.policy?.newVersions !== 'explicit-user-request-only'
    || catalog.policy.minorChanges !== 'edit-existing-version'
    || catalog.policy.repeatRuns !== 'append-batch') throw new Error('实验版本维护约定缺失');
  const sources = new Map(registered.map(entry => [entry.id, entry]));
  const experiments = new Map(), legacy = new Map();
  for (const config of catalog.experiments) {
    const entry = sources.get(config.id);
    if (!entry || entry.experimentOwner || experiments.has(config.id) || !config.objective?.trim()) {
      throw new Error('实验主页登记无效：' + config.id);
    }
    if (entry.archived && !entry.archiveReason?.trim()) throw new Error('已封存实验缺少归档原因：' + entry.id);
    const seen = new Set();
    let previousNumber = 0;
    const iterations = config.iterations.map((node, index) => {
      const number = /^V[1-9]\d*$/.test(node.code) ? Number(node.code.slice(1)) : NaN;
      if (!Number.isSafeInteger(number) || number <= previousNumber || index === 0 && number !== 1 || seen.has(node.code)) {
        throw new Error('版本代号须从 V1 起递增，已删除版本的编号不复用');
      }
      previousNumber = number;
      if (node.parent !== null && !seen.has(node.parent)) throw new Error('内部父版本未登记：' + node.code);
      if (!node.authorization?.request?.trim()
        || index > 0 && node.authorization.kind !== 'explicit-user-request') {
        throw new Error('新增版本需要用户明确要求，不能由小修或重复运行自动创建：' + node.code);
      }
      if (node.archived && !node.archiveReason?.trim()) throw new Error('已封存版本缺少归档原因：' + node.code);
      const source = sources.get(node.sourceId);
      if (node.contentFile && !/^[a-z0-9][a-z0-9.-]*\.md$/.test(node.contentFile)) throw new Error('版本正文文件名无效');
      if (!source || source.experimentOwner && (source.experimentOwner !== entry.id || source.experimentVersion !== node.code)
        || ![node.title,node.change,node.pipeline,node.scenes,node.resultSummary].every(value => typeof value === 'string' && value.trim())
        || !Array.isArray(node.settings) || !node.settings.length) throw new Error('实验版本设置不完整：' + entry.id + '/' + node.code);
      const batchIds = new Set();
      for (const batch of node.batches) {
        if (!/^[a-z0-9][a-z0-9_-]*$/.test(batch.id) || batchIds.has(batch.id)
          || ![batch.title,batch.summary,batch.date].every(value => typeof value === 'string' && value.trim())
          || !batch.outputs?.length || !batch.artifacts?.every(item => item.label && /^\/(?!\/)/.test(item.url))) {
          throw new Error('运行批次登记不完整：' + batch.id);
        }
        batchIds.add(batch.id);
      }
      seen.add(node.code);
      return { ...node, source, url: versionUrl(entry.id, node.code),
        raw: readFileSync(join(root, '内容/迭代', node.contentFile ?? source.file), 'utf8') };
    });
    const current = iterations.find(node => node.code === config.current);
    if (!current) throw new Error('最新版本未登记：' + entry.id);
    const experiment = { ...config, entry, iterations, current };
    experiments.set(entry.id, experiment);
    for (const node of iterations) if (node.source.experimentOwner) {
      if (legacy.has(node.source.id)) throw new Error('既有网址重复归档');
      legacy.set(node.source.id, { experiment, node });
    }
  }
  for (const entry of registered) {
    if (entry.experimentOwner ? !legacy.has(entry.id) : !experiments.has(entry.id)) {
      throw new Error('已有实验尚未登记主页或内部版本：' + entry.id);
    }
  }
  return { experiments, legacy, policy: catalog.policy };
}

export function archivedNotice(entry, esc) {
  return entry.archived ? `<section class="experiment-archive-notice" data-experiment-archived aria-labelledby="archive-reason">
    <span class="experiment-archive-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8h16v12H4zM3 4h18v4H3zM9 12h6"/></svg></span>
    <div><div class="experiment-archive-heading"><span class="experiment-archive-badge">已封存</span><h2 id="archive-reason">归档原因</h2></div><p>${esc(entry.archiveReason)}</p></div>
  </section>` : '';
}

export function renderArchivedContent(entry, content, esc) {
  if (!entry.archived) return content;
  return `${archivedNotice(entry,esc)}
    <details class="experiment-archived-content" data-archived-content>
      <summary><span><strong>原有实验内容</strong><small>实验设置、历史结果与产物</small></span><span class="experiment-archive-toggle"><span class="archive-expand">展开查看</span><span class="archive-collapse">收起内容</span><svg aria-hidden="true" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="m7 4 6 6-6 6"/></svg></span></summary>
      <div class="experiment-archived-body">${content}</div>
    </details>`;
}

function nodeCard(experiment, node, esc) {
  const children = experiment.iterations.filter(item => item.parent === node.code);
  const archived=Boolean(experiment.entry.archived||node.archived);
  return `<li><a class="experiment-node${archived?' experiment-node-archived':''}" href="${node.url}" data-internal-version="${node.code}" data-internal-parent="${node.parent ?? ''}" data-internal-archived="${archived}"${node.archiveReason?` title="${esc(node.archiveReason)}"`:''}>
    <div class="experiment-node-top"><span class="experiment-code">${node.code}</span><span${archived?' class="experiment-node-archive-label"':''}>${esc(archived ? '已封存' : node.status)}</span></div>
    <strong>${esc(node.title)}</strong><p>${esc(node.resultSummary)}</p></a>
    ${children.length ? `<ul>${children.map(item => nodeCard(experiment,item,esc)).join('')}</ul>` : ''}</li>`;
}

export function renderBatchList(experiment, nodes, esc) {
  const batches = nodes.flatMap(node => node.batches.map(batch => ({ node, batch })));
  if (!batches.length) return '<p class="experiment-empty">尚无运行批次记录。</p>';
  return `<div class="experiment-batches">${batches.map(({node,batch}) => `<a class="experiment-batch" href="${batchUrl(experiment.entry.id,node.code,batch.id)}">
    <div><span class="experiment-code">${node.code}</span><strong>${esc(batch.title)}</strong><span aria-hidden="true">↗</span></div>
    <p>${esc(batch.summary)}</p><small>${esc(batch.date)} · ${esc(batch.id)}</small></a>`).join('')}</div>`;
}

export function renderExperimentOverview(experiment, esc) {
  const {entry, iterations, current} = experiment;
  const batchCount = iterations.reduce((total, node) => total + node.batches.length, 0);
  return `<section class="experiment-introduction"><h2>实验目标</h2><p>${esc(experiment.objective)}</p>
    <div class="experiment-overview-stats"><span><b>${iterations.length}</b> 个已登记版本</span><span><b>${batchCount}</b> 个运行批次</span>${experiment.relation==='parallel'?'<span>并列方案</span>':`<a href="${current.url}">最新记录 <b>${current.code}</b> ↗</a>`}</div>
  </section>
  <section class="experiment-section" id="internal-iterations"><h2>内部迭代</h2>
    ${experiment.relation === 'parallel' ? '<p class="experiment-context">以下方案并列比较，编号用于区分记录。</p>' : ''}
    <details class="experiment-tree"><summary><span>查看迭代版本树</span><small>${iterations.map(node=>node.code).join(' · ')}</small></summary>
      <nav aria-label="${esc(entry.title)}内部迭代树"><div class="experiment-tree-root">${esc(entry.title)}</div><ul class="experiment-branches">${iterations.filter(node=>node.parent===null).map(node=>nodeCard(experiment,node,esc)).join('')}</ul></nav>
    </details>
  </section>
  <section class="experiment-section"><h2>${iterations.length>1 ? '各版本方案与结果' : '版本结果'}</h2>
    <div class="table-scroll experiment-comparison" tabindex="0"><table><thead><tr><th>版本与方案</th><th>评测场景</th><th>主要结果</th></tr></thead><tbody>
    ${iterations.map(node=>`<tr><th><a href="${node.url}"><span class="experiment-code">${node.code}</span> ${esc(node.title)}</a></th><td>${esc(node.scenes)}</td><td>${esc(node.resultSummary)}</td></tr>`).join('')}
    </tbody></table></div>
  </section>
  <section class="experiment-section"><h2>运行批次</h2>${renderBatchList(experiment,iterations,esc)}</section>`;
}

export function renderVersionSummary(experiment, node, esc) {
  return `<section class="experiment-section"><h2>本版方案</h2><p>${esc(node.change)}</p>
      ${node.parent ? `<p class="experiment-context">父版本：<a href="${versionUrl(experiment.entry.id,node.parent)}">${node.parent}</a></p>` : ''}</section>
    <section class="experiment-section"><h2>链路与固定设置</h2><p>${esc(node.pipeline)}</p>${settingsTable(node,esc)}</section>
    <section class="experiment-section"><h2>评测场景</h2><p>${esc(node.scenes)}</p></section>
    <section class="experiment-section"><h2>结果概览</h2><p class="experiment-result">${esc(node.resultSummary)}</p></section>
    <section class="experiment-section"><h2>运行批次</h2>${renderBatchList(experiment,[node],esc)}</section>`;
}

export function renderVersionNavigation(experiment,node,esc) {
  return `<nav class="experiment-version-switch" aria-label="内部版本">${experiment.iterations.map(item=>`<a href="${item.url}"${item.code===node.code?' aria-current="page"':''}>${item.code}<span>${esc(item.title)}</span></a>`).join('')}</nav>`;
}

export function settingsTable(node, esc) {
  return `<div class="table-scroll" tabindex="0"><table><thead><tr><th>设置</th><th>说明</th></tr></thead><tbody>${node.settings.map(([label,value])=>`<tr><th>${esc(label)}</th><td>${esc(value)}</td></tr>`).join('')}</tbody></table></div>`;
}

export function renderBatchDetails(experiment, node, batch, esc) {
  return `<section class="experiment-section"><h2>本次运行</h2><p>${esc(batch.title)}</p><p class="experiment-context">批次标识：<code>${esc(batch.id)}</code></p></section>
    <section class="experiment-section"><h2>冻结设置与场景</h2><p>${esc(node.scenes)}</p>${settingsTable(node,esc)}</section>
    <section class="experiment-section"><h2>结果统计</h2><p class="experiment-result">${esc(batch.summary)}</p></section>
    <section class="experiment-section"><h2>原始产物与核验</h2><ul class="experiment-output-paths">${batch.outputs.map(path=>`<li><code>${esc(path)}</code></li>`).join('')}</ul>
    <nav class="experiment-evidence" aria-label="批次产物">${batch.artifacts.map(item=>`<a href="${esc(item.url)}">${esc(item.label)} ↗</a>`).join('')}</nav></section>`;
}
