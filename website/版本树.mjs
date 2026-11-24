const typeLabels = { main: '主版本', experiment: '实验版本' };

export function versionType(version) {
  return typeLabels[version.type];
}

export function readVersions(versions, records) {
  const entries = new Map();
  for (const entry of [...versions, ...records]) {
    const { id, file, title, status, date, summary } = entry;
    if (typeof id !== 'string' || !/^[a-z0-9]+(?:[.-][a-z0-9]+)*$/.test(id) || entries.has(id)) {
      throw new Error(`版本标识无效或重复：${id}`);
    }
    if (typeof file !== 'string' || !/^[^/\\]+\.md$/.test(file)
      || ![title, status, date, summary].every(field => typeof field === 'string' && field.trim())) {
      throw new Error(`版本登记缺少有效字段：${id}`);
    }
    entries.set(id, entry);
  }

  const nodes = new Map(versions.map(version => [version.id, { ...version, children: [] }]));
  const roots = [...nodes.values()].filter(version => version.parent === null);
  if (!roots.length || roots.some(root => root.type !== 'main')) {
    throw new Error('版本树需要至少一个主版本根节点；并列根节点均须为主版本');
  }
  for (const node of nodes.values()) {
    if (!Object.hasOwn(typeLabels, node.type)) throw new Error(`未知版本类型：${node.id}`);
    if (node.parent === null) continue;
    const parent = nodes.get(node.parent);
    if (!parent) throw new Error(`未登记父版本：${node.id} → ${node.parent}`);
    parent.children.push(node);
  }
  for (const record of records) {
    if (!nodes.has(record.parent)) throw new Error(`未登记阶段记录所属版本：${record.id}`);
  }

  let leafCount = 0;
  const rows = [];
  const placed = new Set();
  function place(node, depth) {
    if (placed.has(node.id)) throw new Error('版本树存在循环继承：' + node.id);
    placed.add(node.id);
    node.start = leafCount;
    (rows[depth] ??= []).push(node);
    if (node.children.length) {
      for (const child of node.children) place(child, depth + 1);
    } else {
      leafCount++;
    }
    node.span = leafCount - node.start;
    node.center = node.start + node.span / 2;
  }
  for (const root of roots) place(root, 0);
  if (rows.flat().length !== nodes.size) throw new Error('版本树存在循环继承');

  const css = [`.tree-graph { --lanes: ${leafCount}; --tree-min-width: ${leafCount * 140}px; }`];
  const percent = lane => `${100 * lane / leafCount}%`;
  for (const node of nodes.values()) {
    css.push(`.tree-node[data-version="${node.id}"] { grid-column: ${node.start + 1} / span ${node.span}; }`);
    if (node.parent !== null) {
      const parent = nodes.get(node.parent);
      css.push(`.tree-branch[data-child="${node.id}"] { --child-x: ${percent(node.center)}; --parent-x: ${percent(parent.center)}; }`);
    }
  }
  return { nodes, rows: rows.reverse(), css: css.join('\n') };
}

export function renderTree(tree, esc) {
  const rows = tree.rows.map(nodes => {
    const cards = nodes.map(node => `<a class="tree-node tree-node-${node.type}" href="/iterations/${node.id}/" data-version="${node.id}" data-parent="${node.parent ?? ''}">
      <span class="tree-node-meta"><span>${versionType(node)}</span><span>${esc(node.status)}</span></span>
      <strong>${esc(node.shortTitle ?? node.title)}</strong>
      ${node.metric ? `<span class="tree-metric">${esc(node.metric)}</span>` : ''}
    </a>`).join('');
    const branches = nodes.filter(node => node.parent !== null).map(node =>
      `<span class="tree-branch" data-child="${node.id}"><i class="branch-from"></i><i class="branch-across"></i><i class="branch-to"></i></span>`).join('');
    return `<div class="tree-row">${cards}</div>${branches ? `<div class="tree-links" aria-hidden="true">${branches}</div>` : ''}`;
  }).join('');
  return `<div class="tree-scroll" tabindex="0" role="region" aria-label="项目版本树"><div class="tree-graph">${rows}</div></div>`;
}
