import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { gzipSync } from 'node:zlib';
import { build as bundle } from 'esbuild';
import MarkdownIt from 'markdown-it';
import { readVersions, renderTree, versionType } from './版本树.mjs';
import { readWorkflow, renderWorkflow, workflowMarkdown, workflowGraphData } from './流程卡片.mjs';
import { readUpgradePlan } from './未来升级.mjs';
import { readExperimentGallery, renderExperimentGallery } from './实验图集.mjs';
import { readExperiments, versionUrl, batchUrl, renderExperimentOverview, renderVersionSummary, renderVersionNavigation, renderBatchList, renderBatchDetails, renderArchivedContent } from './experiments.mjs';
import { resources, snapshotFile, snapshotJson, saveManifest } from './published-resources.mjs';
import { prepareMedia, optimizeHtml, copyOptimizedMedia, compressText } from './media-optimization.mjs';

const root = dirname(fileURLToPath(import.meta.url));
const documents = resolve(root, '../modules/claro/docs');
const benchmark = resolve(root, '../modules/maniskill');
const content = join(root, '内容');
const readJson = path => JSON.parse(readFileSync(path, 'utf8'));
const site = readJson(join(content, '网站.json'));
const literature = readJson(join(content, '论文调研/论文目录.json'));
const paperModules = new Map(literature.modules.map(module => [module.id, module]));
const paperIds = new Set();
for (const paper of literature.papers) {
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(paper.id) || paperIds.has(paper.id)
    || !/^[^/\\]+\.md$/.test(paper.file) || !paper.title || !paper.source || !paper.fulltext
    || !Number.isInteger(paper.year) || !paper.modules.length
    || paper.modules.some(id => !paperModules.has(id))
    || paper.versions.some(id => !site.versions.some(version => version.id === id))) {
    throw new Error('论文登记内容无效：' + paper.id);
  }
  paperIds.add(paper.id);
}
const sceneGallery = site.sceneGallery ? readJson(join(content, site.sceneGallery)) : null;
const report = snapshotJson('claro-paired',join(documents, '06_做配对验证/resources/S5_固定挡物接续01/配对合并验收.json'));
const fixed = snapshotJson('baseline-paired',join(benchmark, 'docs/resource/两条管线200例成功场景分布/统计核对.json'));
const fixedMetrics = snapshotJson('baseline-metrics',join(benchmark, 'outputs', fixed.batch, 'independent_metrics.json'));
const mpc = snapshotJson('mpc-paired',join(benchmark, 'docs/resource/arx_graspcl_mpc_200_20261119_03/统计核对.json'));
const { records = [], configurationRecords = [] } = site;
const configurationRoutes = Object.fromEntries(configurationRecords.flatMap(record=>[
  [record.id,`/configuration/${record.id}/`],
  ...(record.sections??[]).map(section=>[section.id,`/configuration/${record.id}/#${section.id}`])
]));
const registeredVersions = site.versions;
const registeredById = new Map(registeredVersions.map(entry=>[entry.id,entry]));
const {experiments,legacy:internalVersions} = readExperiments(root,registeredVersions);
const versions = registeredVersions.filter(entry=>!entry.experimentOwner&&!entry.archived);
const archivedVersions = registeredVersions.filter(entry=>!entry.experimentOwner&&entry.archived);
const ownerId = id => registeredById.get(id)?.experimentOwner ?? id;
const sourceUrl = entry => entry.experimentOwner ? versionUrl(entry.experimentOwner,entry.experimentVersion) : `/iterations/${entry.id}/`;
const extraRecords = [...records,...registeredVersions.filter(entry=>entry.experimentOwner||entry.archived)]
  .map(entry=>({...entry,parent:entry.experimentOwner??ownerId(entry.parent)}));
const tree = readVersions(versions, extraRecords);
const workflows = new Map(registeredVersions.filter(v => v.workflow).map(v => [v.id, readWorkflow(content, v)]));
const experimentGalleries = new Map(registeredVersions.filter(v=>v.gallery).map(v=>[v.id,readExperimentGallery(content,v)]));
const workflowAssets = new Map();
const media = new Map();
for (const [folder, mapping] of [
  [documents, site.assets], [benchmark, site.benchmarkAssets],
  [join(content, '资源'), site.localAssets], [join(content, '资源'), sceneGallery?.assets],
  ...[...experimentGalleries.values()].map(gallery=>[join(content,'资源'),gallery.assets])
]) {
  for (const [name, path] of Object.entries(mapping ?? {})) {
    if (!/^[a-z0-9][a-z0-9.-]*$/.test(name) || media.has(name) || resolve(folder,path)!==join(folder,path)) {
      throw new Error(`非法或重复的素材路径：${name}`);
    }
    const source = snapshotFile('media/'+name,resolve(folder,path));
    media.set(name, source);
  }
}
const markdown = new MarkdownIt({ html: false, linkify: false });
const esc = markdown.utils.escapeHtml;

const renderImage = markdown.renderer.rules.image;
const renderLink = markdown.renderer.rules.link_open;
markdown.renderer.rules.link_open = (tokens, index, options, env, renderer) => {
  if (env?.resolveUpgradeLink) {
    tokens[index].attrSet('href',env.resolveUpgradeLink(tokens[index].attrGet('href')));
  }
  if (/\.(?:png|jpe?g|gif|webp|svg|avif|bmp)(?:[?#]|$)/i.test(tokens[index].attrGet('href') ?? '')) {
    tokens[index].attrSet('target', '_blank');
    tokens[index].attrSet('rel', 'noopener');
    if (!tokens[index].attrGet('title')) tokens[index].attrSet('title', '在新标签页查看原图');
  }
  return renderLink ? renderLink(tokens, index, options, env, renderer)
    : renderer.renderToken(tokens, index, options);
};
markdown.renderer.rules.image = (tokens, index, options, env, renderer) => {
  if (env?.resolveUpgradeLink) tokens[index].attrSet('src', env.resolveUpgradeLink(tokens[index].attrGet('src')));
  tokens[index].attrSet('loading', 'lazy');
  tokens[index].attrSet('decoding', 'async');
  const picture = renderImage(tokens, index, options, env, renderer);
  let linkDepth = 0;
  for (const token of tokens.slice(0, index)) {
    if (token.type === 'link_open') linkDepth++;
    else if (token.type === 'link_close') linkDepth--;
  }
  if (linkDepth > 0) return picture;
  return `<a class="image-original-link" href="${esc(tokens[index].attrGet('src'))}" target="_blank" rel="noopener" title="在新标签页查看原图">${picture}</a>`;
};
markdown.renderer.rules.table_open = () => '<div class="table-scroll" tabindex="0"><table>';
markdown.renderer.rules.table_close = () => '</table></div>';
markdown.renderer.rules.heading_open = (tokens,index,options,env,renderer) => {
  const title=tokens[index+1]?.content;
  if (env?.upgradeAnchors?.[title]) tokens[index].attrSet('id',env.upgradeAnchors[title]);
  return renderer.renderToken(tokens,index,options);
};
for (const rule of ['th_open','td_open']) {
  markdown.renderer.rules[rule] = (tokens,index,options,env,renderer) => {
    const token=tokens[index];
    const alignment=/^text-align:(left|center|right)$/.exec(token.attrGet('style') ?? '');
    if (alignment) {
      token.attrs=token.attrs.filter(([name])=>name!=='style');
      token.attrJoin('class','align-'+alignment[1]);
    }
    return renderer.renderToken(tokens,index,options);
  };
}
markdown.renderer.rules.evidence_video = (tokens, i) => {
  const { href, label } = tokens[i].meta;
  return video(href, label);
};
markdown.core.ruler.after('inline', 'video_links', state => {
  for (let i = 1; i < state.tokens.length - 1; i++) {
    const tokens = state.tokens[i].children;
    if (state.tokens[i - 1].type !== 'paragraph_open' || state.tokens[i + 1].type !== 'paragraph_close'
      || tokens?.length !== 3 || tokens[0].type !== 'link_open' || tokens[1].type !== 'text'
      || tokens[2].type !== 'link_close') continue;
    const href = tokens[0].attrGet('href');
    if (!/^\/media\/[a-z0-9.-]+\.mp4$/.test(href ?? '') || !media.has(basename(href))) continue;
    const t = new state.Token('evidence_video', '', 0);
    t.meta = { href, label: tokens[1].content };
    state.tokens.splice(i - 1, 3, t);
  }
});

function write(path, body) {
  const file = join(output, path);
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, path.endsWith('.html') ? optimizeHtml(body, optimizedMedia) : body);
}

const upgradePlan=site.roadmap ? readUpgradePlan(root,site.roadmap,markdown) : null;
for (const name of ['cover.jpg','comparison-cover.jpg']) snapshotFile('media/'+name,join(root,'页面',name));
const optimizedMedia = prepareMedia(root);

function withPageContents(body) {
  const marker = '[[page-contents]]';
  if (body.split(marker).length !== 2) throw new Error('详情页缺少目录位置');
  const usedIds = new Set([...body.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]));
  const headings = [];
  const updated = body.replace(/(<article\b[^>]*>)([\s\S]*?)(<\/article>)/, (_, open, article, close) => {
    const content = article.replace(/<h([1-6])\b([^>]*)>([\s\S]*?)<\/h\1>/g,
      (heading, level, attrs, inner) => {
        const title = markdown.utils.unescapeAll(inner.replace(/<[^>]*>/g, '')).replace(/\s+/g, ' ').trim();
        if (!title) return heading;
        let id = attrs.match(/\bid="([^"]+)"/)?.[1];
        if (!id) {
          const base = level === '1' && headings.length === 0 ? 'page-overview'
            : 'section-' + createHash('sha256').update(title).digest('hex').slice(0, 10);
          id = base;
          for (let n = 2; usedIds.has(id); n++) id = `${base}-${n}`;
          usedIds.add(id);
          attrs += ` id="${id}"`;
        }
        if (headings.some(item => item.id === id)) throw new Error('详情页标题锚点重复：' + id);
        headings.push({ id, title, level: Number(level) });
        return `<h${level}${attrs}>${inner}</h${level}>`;
      });
    return open + content + close;
  });
  if (!headings.length) throw new Error('详情页没有可用于目录的标题');
  const sidebar = body.includes('class="container paper-detail"') ? 'paper-sidebar' : 'version-sidebar';
  const contents = `<aside class="${sidebar} contents-sidebar" data-page-toc aria-labelledby="page-contents-title">
    <div class="contents-heading"><h2 id="page-contents-title">本页目录</h2><span>${headings.length} 个标题</span></div>
    <nav class="page-toc-links" aria-label="本页目录">${headings.map((item, index) =>
      `<a href="#${encodeURIComponent(item.id)}" data-toc-link="${esc(item.id)}" data-level="${item.level}"${index === 0 ? ' aria-current="location"' : ''}>${esc(index === 0 && item.level === 1 ? '页面概览' : item.title)}</a>`).join('')}</nav>
  </aside>`;
  return updated.replace(marker, contents);
}

function page(title, active, body, reading = false) {
  if (reading) body = withPageContents(body);
  const menu = [
    ['项目概览', '/', 'home'], ['迭代版本', '/iterations/', 'iterations'],
    ['实验成果', '/results/', 'results'], ['论文调研', '/papers/', 'papers'],
    ['配置与核查', '/configuration/', 'configuration'], ['升级计划','/roadmap/','roadmap']
  ]
    .map(([text, url, id]) => `<a href="${url}"${active === id ? ' aria-current="page"' : ''}>${text}</a>`).join('');
  const html = `<!doctype html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="${esc(site.description)}">
<title>${esc(title)} · ${esc(site.title)}</title>
<link rel="stylesheet" href="/site.css"><link rel="stylesheet" href="/version-tree.css"><link rel="stylesheet" href="/research.css"><link rel="stylesheet" href="/reading.css"><link rel="stylesheet" href="/experiments.css"><script type="module" src="/site.mjs"></script>
</head><body class="page-${active}${reading ? ' reading-page' : ''}">
<a class="skip" href="#main">跳到正文</a>
<header class="site-header"><div class="container header-inner">
<a class="wordmark" href="/"><span class="brand-symbol" aria-hidden="true">A<span>R</span>X</span><span class="brand-name">抓取研究</span></a>
<nav aria-label="主导航">${menu}</nav>
</div></header>
<main id="main">${body}</main>
<footer class="site-footer"><div class="container"><span>${esc(site.title)} · 仿真实验记录</span><a href="/iterations/">浏览版本记录</a></div></footer>
</body></html>`;
  return html.replace(/href="\/configuration\/#([^"]+)"/g,(match,id)=>configurationRoutes[id]?`href="${esc(configurationRoutes[id])}"`:match);
}

function video(src, title) {
  const posters = {
    '/media/i0069-comparison.mp4': '/media/comparison-cover.jpg',
    '/media/i0069-regrasp.mp4': '/media/cover.jpg',
    '/media/first-regrasp.mp4': '/media/cover.jpg'
  };
  const poster = posters[src] ? ` poster="${posters[src]}"` : '';
  return `<figure class="video-figure"><video controls playsinline preload="none"${poster} aria-label="${esc(title)}">
<source src="${esc(src)}" type="video/mp4">浏览器可以通过下方链接打开视频。
</video><figcaption><span>${esc(title)}</span><a href="${esc(src)}" download>下载视频</a></figcaption></figure>`;
}

function versionList(entries) {
  return `<div class="version-list">${entries.map(v => `<a class="version-row" href="/iterations/${v.id}/">
    <span class="version-tag">${v.archived ? '已封存' : versionType(v) ?? '阶段记录'}</span>
    <div><h3>${esc(v.title)}</h3><p>${esc(v.summary)}</p></div>
    <span class="version-metric">${esc(v.metric ?? v.status)}</span></a>`).join('')}</div>`;
}

function archiveList() {
  return archivedVersions.length ? `<details class="experiment-archives" id="archived-experiments"><summary>已封存实验（${archivedVersions.length}）</summary>${versionList(archivedVersions)}</details>` : '';
}

function todoScope(entry) {
  const nodes=experiments.get(entry.id)?.iterations??[];
  const ids=[...new Set([entry.id,...nodes.map(node=>node.sourceId)])];
  const codes=Object.fromEntries(nodes.map(node=>[node.sourceId,node.code]));
  const links=Object.fromEntries(nodes.map(node=>[node.sourceId,node.url+'#todos']));
  return `data-todo-versions="${esc(ids.join(','))}" data-todo-codes="${esc(JSON.stringify(codes))}" data-todo-links="${esc(JSON.stringify(links))}"`;
}

const layoutRows = Object.entries(fixed.by_layout).sort(([a], [b]) => a.localeCompare(b));
const benchmarkTotal = layoutRows.reduce((total, [, layout]) => total + layout.registered, 0);
if (!fixed.checks.final_verification_passed || !mpc.registered_final_verification.passed
  || mpc.checked_same_case_id_layout_model_seed_and_reference_outcome !== benchmarkTotal
  || mpc.results.completed !== benchmarkTotal) {
  throw new Error('200 例对照缺少完整的共同场景验收');
}

const benchmarkColumns = [['baseline', '主版本'], ['graspcl', 'GraspCL'], ['mpc', 'GraspCL＋MPC']];

function benchmarkTable() {
  const totals = [fixed.successes.fixed_zerograsp, fixed.successes.fixed_graspcl, mpc.results.successes];
  const resultCell = (count, total) =>
    `<td><span>${count}/${total}</span> <small>(${(100 * count / total).toFixed(1)}%)</small></td>`;
  const rows = layoutRows.map(([key, row]) => `<tr><th scope="row">${esc(key.slice(0, 2))} · ${esc(row.name)}</th>
    ${resultCell(row.zerograsp_successes, row.registered)}${resultCell(row.graspcl_successes, row.registered)}
    ${resultCell(mpc.results.by_layout[key].success, row.registered)}</tr>`).join('');
  return `<div class="table-scroll results-table benchmark-table" tabindex="0"><table>
    <thead><tr><th scope="col">场景布局</th>${benchmarkColumns.map(([id, title]) => `<th scope="col"><a href="/iterations/${id}/">${title}</a></th>`).join('')}</tr></thead>
    <tbody>${rows}<tr><th scope="row">合计</th>${totals.map(count => resultCell(count, benchmarkTotal)).join('')}</tr></tbody>
  </table></div>`;
}

function failureTable() {
  const reasons = [
    '初始化／观察失败', '无可用候选', '抓取 IK／场景碰撞未通过',
    '预抓取或抬升 IK／碰撞未通过', '路径搜索失败', '测量状态碰撞',
    '动作未收敛', 'MPC 未找到安全推进步', '未形成双指接触',
    '未达到当组抬升／保持标准', '出现禁止接触'
  ];
  const raw = mpc.results.failure_reasons;
  const planning = mpc.initial_planning_failures_by_deepest_reached_stage;
  const motionReasons = [
    'mpc_approach_not_converged', 'close_not_converged', 'route_mpc_approach_not_converged',
    'lift_not_converged', 'mpc_handoff_not_settled'
  ];
  const distributions = [
    fixedMetrics.arms.fixed_zerograsp.failure_distribution,
    fixedMetrics.arms.fixed_graspcl.failure_distribution,
    {
      '初始化／观察失败': raw.independent_scene_admission_failed,
      '无可用候选': 0,
      '抓取 IK／场景碰撞未通过': planning.grasp_ik,
      '预抓取或抬升 IK／碰撞未通过': planning.pre_or_lift_ik,
      '路径搜索失败': planning.path,
      '测量状态碰撞': raw.measured_start_state_in_collision,
      '动作未收敛': motionReasons.reduce((sum, reason) => sum + (raw[reason] ?? 0), 0),
      'MPC 未找到安全推进步': raw.smooth_mpc_no_safe_interval,
      '未形成双指接触': raw.no_bilateral_target_contact,
      '未达到当组抬升／保持标准': raw.insufficient_hold_height,
      '出现禁止接触': raw.forbidden_robot_contact
    }
  ];
  const totals = [
    benchmarkTotal - fixed.successes.fixed_zerograsp,
    benchmarkTotal - fixed.successes.fixed_graspcl,
    benchmarkTotal - mpc.results.successes
  ];
  if (Object.values(planning).reduce((sum, count) => sum + count, 0)
    !== raw.no_feasible_native_post_or_recalled_candidate) {
    throw new Error('MPC 初始规划失败的阶段归类与原始例数不一致');
  }
  distributions.forEach((distribution, index) => {
    const counts = reasons.map(reason => distribution[reason] ?? 0);
    if (!counts.every(count => Number.isInteger(count) && count >= 0)
      || counts.reduce((sum, count) => sum + count, 0) !== totals[index]) {
      throw new Error(`${benchmarkColumns[index][1]} 失败分布与已验证结果不一致`);
    }
  });
  const rows = reasons.map(reason => `<tr><th scope="row">${esc(reason)}</th>${distributions.map(
    distribution => `<td>${distribution[reason] ?? 0}</td>`).join('')}</tr>`).join('');
  return `<div class="table-scroll results-table benchmark-table" tabindex="0"><table aria-labelledby="failure-title">
    <thead><tr><th scope="col">失败原因</th>${benchmarkColumns.map(([id, title]) =>
      `<th scope="col"><a href="/iterations/${id}/">${title}</a></th>`).join('')}</tr></thead>
    <tbody>${rows}<tr><th scope="row">失败合计</th>${totals.map(count => `<td><span>${count}</span></td>`).join('')}</tr></tbody>
  </table></div>`;
}

function pipelineDemos() {
  const demos = [
    ['baseline', 'ZeroGrasp＋cuRobo', '原生抓取候选与全局轨迹规划'],
    ['graspcl', '加入 GraspCL', '冻结示教召回与抓取候选重评分'],
    ['mpc', '加入 MPC（已封存）', '历史对照：朝向搜索、闭环跟踪与腕部精调']
  ];
  return `<div class="demo-grid">${demos.map(([id, title, description]) => `<article class="demo-item">
    <video controls playsinline preload="none" poster="/media/${id}-poster.jpg" aria-label="${esc(title)} 抓取录像">
      <source src="/media/${id}-demo.mp4" type="video/mp4">
    </video><h3><a href="/iterations/${id}/">${esc(title)} <span aria-hidden="true">↗</span></a></h3><p>${esc(description)}</p>
  </article>`).join('')}</div>`;
}

function overview() {
  return page('项目概览', 'home', `<div class="research-home">
  <header class="research-hero container">
    <p class="hero-context">ARX ROBOTICS <span>×</span> MANISKILL</p>
    <h1><span class="hero-arx">ARX</span> 桌面抓取研究</h1>
    <p class="hero-subtitle">抓取生成、运动规划与闭环执行</p>
    <p class="hero-description"><span>以 RGB-D 感知与 ManiSkill 物理仿真为基础，</span><span>追踪 ARX 两条并列主线及其抓取研究。</span></p>
    <div class="hero-actions">
      <a class="research-button primary" href="#iterations">研究版本与进度 <span aria-hidden="true">↗</span></a>
      <a class="research-button" href="/results/">实验成果 <span aria-hidden="true">↗</span></a>
      <a class="research-button" href="/configuration/#scene-gallery">评测场景 <span aria-hidden="true">↗</span></a>
      ${upgradePlan ? '<a class="research-button" href="#roadmap">升级计划 <span aria-hidden="true">↗</span></a>' : ''}
    </div>
    <div class="hero-methods">ZeroGrasp <span>·</span> SAM3 <span>·</span> GraspGenX <span>·</span> cuRobo <span>·</span> GraspCL <span>·</span> CLARO</div>
  </header>

  <section class="research-section method-section" aria-labelledby="method-title"><div class="container">
    <div class="research-heading"><h2 id="method-title">从观测到抓取</h2>
      <p>ZeroGrasp 与 SAM3＋RGB-D 点云＋GraspGenX 两条主线并列，分别记录进度与结果。</p></div>
    <div class="mainline-label"><h3><a href="/iterations/baseline/">主线一 · ZeroGrasp＋cuRobo</a></h3><span>原主线与实测结果保留</span></div>
    <div class="method-flow">
      <div class="method-step"><span class="method-label">感知</span><h3>腕部 RGB-D</h3><p>目标图像、深度与实例掩码</p></div>
      <div class="method-step"><span class="method-label">生成</span><h3>ZeroGrasp</h3><p>六自由度候选与原生后处理</p><a href="/iterations/graspcl/">＋ GraspCL 召回与重评分</a></div>
      <div class="method-step"><span class="method-label">规划</span><h3>cuRobo</h3><p>IK、场景碰撞与轨迹搜索</p></div>
      <div class="method-step"><span class="method-label">执行</span><h3>ARX 抓取与抬升</h3><p>接近、闭爪、抬升与保持</p><a href="/iterations/reachability-improvement/">查看可达性改进方案</a></div>
    </div>
    <div class="mainline-label"><h3><a href="/iterations/sam3-graspgenx/">主线二 · SAM3＋RGB-D 点云＋GraspGenX＋cuRobo</a></h3><span>${esc(registeredById.get('sam3-graspgenx').status)}</span></div>
    <div class="method-flow">
      <div class="method-step"><span class="method-label">语义感知</span><h3>头部 RGB-D＋SAM3</h3><p>用任务描述得到目标实例掩码</p><a href="/iterations/sam3-graspgenx/steps/segmentation/">查看语义分割方案</a></div>
      <div class="method-step"><span class="method-label">三维几何</span><h3>RGB-D＋RaySt3R</h3><p>用当前观测和目标掩码补全三维点云</p><a href="/iterations/sam3-graspgenx/steps/point-cloud/">查看点云接口</a></div>
      <div class="method-step"><span class="method-label">抓取生成</span><h3>GraspGenX</h3><p>六自由度抓法、评分与原生候选处理</p><a href="/iterations/sam3-graspgenx/steps/grasp-generation/">查看候选与坐标适配</a></div>
      <div class="method-step"><span class="method-label">规划与执行</span><h3>cuRobo＋ARX</h3><p>运动学、碰撞、路径与物理抓取核验</p><a href="/iterations/sam3-graspgenx/">进入新主线</a></div>
    </div>
  </div></section>

  <section class="research-section" aria-labelledby="versions-title"><div class="container">
    <div class="research-heading"><h2 id="versions-title">研究版本与进度</h2><p>两条主线在底部并列；从下往上查看各自的后续版本，点击节点查看方法和记录。</p></div>
    <div class="overview-layout">
      <section class="version-tree" id="iterations" aria-labelledby="tree-title">
        <div class="section-heading"><h3 id="tree-title">版本演进</h3><a href="/iterations/">全部版本 ↗</a></div>
        ${renderTree(tree, esc)}
      </section>
      <section class="todo-panel" id="project-todos" data-todo-summary aria-labelledby="project-todos-title">
        <div class="section-heading"><h3 id="project-todos-title">待办汇总</h3><span class="todo-count" data-todo-count aria-live="polite"></span></div>
        <p class="todo-feedback" role="status" data-todo-feedback>正在读取待办…</p>
        <div class="todo-groups">${versions.map(v => `<section class="todo-group" data-todo-group="${v.id}" ${todoScope(v)} hidden>
          <h4><a href="/iterations/${v.id}/#todos">${esc(v.title)}</a></h4><ul data-todo-items></ul>
        </section>`).join('')}</div>
      </section>
    </div>
    ${archiveList()}
    <script type="module" src="/todos.mjs"></script>
  </div></section>



  ${upgradePlan ? `<section class="research-section roadmap-section" id="roadmap" aria-labelledby="roadmap-title"><div class="container">
    <div class="research-heading"><h2 id="roadmap-title">未来升级计划</h2><p>查看已完成的基础验证、正在进行的工作与后续改进方向。</p></div>
    <div class="roadmap-summary">${upgradePlan.summary}</div>
    <div class="roadmap-actions"><a class="research-button primary" href="/roadmap/">查看完整计划与实施细节 <span aria-hidden="true">↗</span></a></div>
  </div></section>` : ''}

  <section class="research-section configuration-section" aria-labelledby="configuration-title"><div class="container">
    <div class="research-heading"><h2 id="configuration-title">实验配置与核查</h2><p>查看场景、工具坐标和运动学设置。</p></div>
    <div class="configuration-links">${configurationRecords.map(record => `<a href="/configuration/#${esc(record.id)}">
      <h3>${esc(record.title)} <span aria-hidden="true">↗</span></h3><p>${esc(record.summary)}</p>
    </a>`).join('')}</div>
  </div></section>
  </div>`);
}

function papersPage() {
  const cards = literature.papers.map(paper => {
    const modules = paper.modules.map(id => paperModules.get(id).title).join(' · ');
    const method = readFileSync(join(content, '论文调研', paper.file), 'utf8');
    const search = [paper.id, paper.name, paper.title, paper.authors, paper.summary, modules, method].join(' ').toLowerCase();
    return `<article class="paper-card" data-paper-card data-modules="${esc(paper.modules.join(' '))}" data-status="${esc(paper.status)}" data-search="${esc(search)}">
      <div class="paper-card-meta"><span>${esc(modules)}</span><span>${paper.year}</span></div>
      <h2><a href="/papers/${paper.id}/">${esc(paper.name)} <span aria-hidden="true">↗</span></a></h2>
      <p class="paper-card-title">${esc(paper.title)}</p><p class="paper-summary">${esc(paper.summary)}</p>
      <div class="paper-card-bottom"><span class="paper-status">${esc(paper.status)}</span><a href="/papers/${paper.id}/">阅读具体方法 →</a></div>
    </article>`;
  }).join('');
  return page('论文调研与方法', 'papers', `<div class="container papers-page">
    <header class="page-heading papers-heading"><p class="eyebrow">方法与研究依据</p><h1>论文调研与方法</h1>
      <p>按模块查阅参考论文，追踪具体方法和当前链路中的接入情况。</p></header>
    <div class="paper-filters" role="search" aria-label="查找参考论文">
      <label class="paper-search">搜索论文或方法<input type="search" id="paper-search" placeholder="标题、作者、方法关键词"></label>
      <label>所属模块<select id="paper-module"><option value="">全部模块</option>${literature.modules.map(module =>
        `<option value="${module.id}">${esc(module.title)}</option>`).join('')}</select></label>
      <label>接入状态<select id="paper-status"><option value="">全部状态</option>${[...new Set(literature.papers.map(paper => paper.status))].map(status =>
        `<option value="${esc(status)}">${esc(status)}</option>`).join('')}</select></label>
    </div>
    <div class="paper-list-heading"><span id="paper-count" aria-live="polite">${literature.papers.length} 篇论文</span><span>每篇论文保存为独立方法记录</span></div>
    <div class="paper-grid">${cards}</div><p class="paper-empty" id="paper-empty" hidden>没有匹配的论文，请调整关键词或筛选条件。</p>
  </div>`);
}

function paperPage(paper) {
  const raw = readFileSync(join(content, '论文调研', paper.file), 'utf8');
  write(`papers/${paper.id}/method.md`, `# ${paper.title}\n\n${paper.authors} · ${paper.year}\n\n${raw}`);
  return page(`${paper.name} 方法`, 'papers', `<div class="container paper-detail">
    [[page-contents]]
    <article class="paper-article"><header class="page-heading paper-heading">
      <p class="eyebrow">${esc(paper.modules.map(id => paperModules.get(id).title).join(' · '))}</p>
      <h1>${esc(paper.name)} 的方法</h1><p class="paper-fulltitle">${esc(paper.title)}</p>
      <p class="paper-authors">${esc(paper.authors)}</p>
      <div class="paper-metadata"><span>${esc(paper.venue)}</span><span class="paper-status">${esc(paper.status)}</span><span>原文核对 ${esc(paper.verifiedAt)}</span></div>
      <div class="paper-actions"><a href="${esc(paper.source)}" target="_blank" rel="noopener">论文原文 ↗</a><a href="${esc(paper.fulltext)}" target="_blank" rel="noopener">阅读全文 ↗</a><a href="./method.md" download>下载方法记录</a></div>
      <p class="paper-used-by">关联版本 ${paper.versions.map(id => {
        const v = registeredById.get(id);
        return `<a href="${sourceUrl(v)}">${esc(v.experimentOwner ? `${registeredById.get(v.experimentOwner).title} ${v.experimentVersion} · ${v.shortTitle??v.title}` : v.shortTitle ?? v.title)}</a>`;
      }).join(' · ')}</p>
    </header><div class="markdown">${markdown.render(raw)}</div></article>
  </div>`, true);
}

function sceneGalleryHtml() {
  if (!sceneGallery || sceneGallery.groups.length !== 5 || sceneGallery.groups.some(group => group.cases.length !== 40)) {
    throw new Error('场景图集需包含五类布局，每类 40 个场景');
  }
  return `<div class="scene-gallery">
    <nav class="scene-tabs" aria-label="场景复杂度">${sceneGallery.groups.map(group =>
      `<a href="#gallery-${esc(group.id)}">${esc(group.id.slice(0, 2))} · ${esc(group.title)}</a>`).join('')}</nav>
    ${sceneGallery.groups.map(group => `<section class="scene-group" id="gallery-${esc(group.id)}">
      <h3>${esc(group.id.slice(0, 2))} · ${esc(group.title)} <span>40 个场景</span></h3>
      <div class="scene-scroll" tabindex="0" role="region" aria-label="${esc(group.title)}场景拼图">
        <div class="scene-montage">
          <img src="${esc(group.sheet.url)}" alt="${esc(group.title)}的 40 个场景，按编号排列" loading="lazy" decoding="async" width="${group.sheet.width}" height="${group.sheet.height}">
          <div class="scene-hotspots">${group.cases.map(item => {
            const title = `${item.label} · ${item.target}`;
            const attrs = `data-scene-title="${esc(item.id)}" data-scene-caption="目标物体 ${esc(item.target)} · ${esc(group.title)}"`;
            return item.url
              ? `<a href="${esc(item.url)}" target="_blank" rel="noopener" ${attrs} aria-label="在新标签页放大查看 ${esc(title)}" title="${esc(title)} · 在新标签页查看原图"></a>`
              : `<span class="scene-missing" role="img" ${attrs} aria-label="${esc(title)} · 无历史截图：${esc(item.reason)}" title="${esc(title)} · ${esc(item.reason)}"></span>`;
          }).join('')}</div>
        </div>
      </div>
    </section>`).join('')}
  </div>`;
}

function configurationPage() {
  return page('实验配置与核查', 'configuration', `<div class="container configuration-page" data-configuration-routes="${esc(JSON.stringify(configurationRoutes))}">
    <header class="page-heading"><h1>实验配置与核查</h1><p>选择一项查看设置、对比图和核查结果。</p></header>
    <nav class="configuration-links" aria-label="配置记录目录">${configurationRecords.map(record =>
      `<a class="configuration-index-card" href="/configuration/${record.id}/"><h2>${esc(record.title)} <span aria-hidden="true">↗</span></h2><p>${esc(record.summary)}</p><small>${esc(record.status)}</small></a>`).join('')}</nav>
  </div>`);
}

function configurationDetail(record) {
  let download='';
  const items=record.sections??[record];
  const body=items.map(item=>{
    let raw=readFileSync(join(content,'实验配置',item.file),'utf8').replace(/^# [^\n]+\n+/, '');
    if(record.sections)raw=raw.replace(/^(#{2,6})\s+/gm,(match,marks)=>'#'.repeat(Math.max(3,marks.length))+' ');
    download+=`\n## ${item.title}\n\n${raw}\n`;
    return `<section class="configuration-record" id="${item.id}">${record.sections?`<h2>${esc(item.title)}</h2>`:''}<div class="markdown">${markdown.render(raw)}</div>${item.gallery?`<h2>按场景布局浏览</h2>${sceneGalleryHtml()}`:''}</section>`;
  }).join('');
  write(`configuration/${record.id}/record.md`,`# ${record.title}\n${download}`);
  return page(record.title,'configuration',`<div class="container article-layout">[[page-contents]]<article class="article">
    <header class="page-heading"><p class="eyebrow"><a href="/configuration/">实验配置与核查</a><span class="status">${esc(record.status)}</span></p><h1>${esc(record.title)}</h1><p>${esc(record.summary)}</p></header>
    ${body}<p class="article-foot"><a href="./record.md" download>下载本页记录</a></p></article></div>`,true);
}

function iterationsPage() {
  return page('迭代版本', 'iterations', `<div class="container list-page"><header class="page-heading">
    <h1>迭代版本</h1><p>选择版本查看技术改动、评测结果和参考文献。</p>
    </header>${versionList(versions)}${archiveList()}</div>`);
}

function versionTodos(version) {
  return `<section class="todo-panel version-todos" id="todos" data-todo-version="${version.id}" aria-labelledby="todos-title">
    <div class="section-heading"><h2 id="todos-title">待办事项</h2><span class="todo-count" data-todo-count aria-live="polite"></span></div>
    <form data-todo-form>
      <label for="todo-text">新增待办</label>
      <fieldset class="todo-fields" disabled>
        <input id="todo-text" name="text" type="text" maxlength="250" placeholder="记录下一步要做的事" required>
        <button type="submit">＋ 添加待办</button>
      </fieldset>
    </form>
    <p class="todo-feedback" role="status" data-todo-feedback>正在读取待办…</p>
    <ul class="todo-items" data-todo-items></ul>
  </section>
  <script type="module" src="/todos.mjs"></script>`;
}

function renderSource(v, options={}) {
  const raw = options.raw ?? readFileSync(join(content, '迭代', v.file), 'utf8');
  const workflow = workflows.get(v.id);
  const marker = '[[流程卡片]]';
  if (workflow && raw.split(marker).length !== 2) throw new Error('流程卡片标记缺失或重复：' + v.id);
  const gallery=options.omitGallery ? null : experimentGalleries.get(v.id);
  let downloadBody=workflow ? raw.replace(marker,workflowMarkdown(workflow)) : raw;
  if (gallery) downloadBody=downloadBody.replace('[[实验图集]]',`![十个代表场景](${gallery.sheet.url})\n\n${gallery.cases.map(item=>`- [${item.id}](${item.url})：${item.caption}`).join('\n')}`);
  const rendered = markdown.render(raw);
  let articleBody = workflow ? rendered.replace(`<p>${marker}</p>`, renderWorkflow(workflow, esc, workflowAssets.get(v.id))) : rendered;
  if (gallery) {
    if (raw.split('[[实验图集]]').length!==2) throw new Error('实验图集标记缺失或重复：'+v.id);
    articleBody=articleBody.replace('<p>[[实验图集]]</p>',renderExperimentGallery(gallery,esc));
  }
  return {articleBody,downloadBody};
}

function iteration(v) {
  const {articleBody,downloadBody}=renderSource(v);
  write(`iterations/${v.id}/record.md`,downloadBody);
  const experiment=experiments.get(v.id);
  const node = tree.nodes.get(v.id);
  const parent = registeredById.get(v.parent);
  const related = records.filter(record => ownerId(record.parent) === v.id);
  const subrecords = related.length ? `<section class="evidence-section"><h2>历史阶段记录</h2>${versionList(related)}</section>` : '';
  const children = node?.children.length
    ? `<section class="evidence-section"><h2>后续版本</h2>${versionList(node.children)}</section>` : '';
  const originalContent = `${experiment ? renderExperimentOverview(experiment,esc) : ''}${literature.papers.some(paper => paper.versions.includes(v.id)) ? `<nav class="version-papers" aria-label="参考论文方法"><span>参考论文方法</span>${literature.papers.filter(paper => paper.versions.includes(v.id)).map(paper => `<a href="/papers/${paper.id}/">${esc(paper.name)} ↗</a>`).join('')}</nav>` : ''}<div class="markdown">${articleBody}</div>
    <div class="article-foot"><a href="./record.md" download>下载 Markdown 原文</a></div>${children}${subrecords}`;
  return page(v.title, 'iterations', `<div class="container article-layout${experiment ? ' has-todos' : ''}">
   [[page-contents]]
   <article class="article"><header class="page-heading">
   <p class="eyebrow"><span>${experiment ? '实验主页' : '历史阶段记录'}</span><time datetime="${esc(v.date)}">${esc(v.date)}</time><span class="status">${esc(v.status)}</span></p>
   <h1>${esc(v.title)}</h1><p>${esc(v.summary)}</p>
   ${parent ? `<p class="parent-version">所属研究 <a href="${sourceUrl(parent)}">${esc(parent.title)}</a></p>` : ''}
   </header>${renderArchivedContent(v,originalContent,esc)}</article>
   ${experiment ? versionTodos(v) : ''}
   </div>`, true);
}

function experimentIteration(experiment,node) {
  const {articleBody,downloadBody}=renderSource(node.source,{raw:node.raw,omitGallery:node.reportLayout});
  const download=`# ${node.code} · ${node.title}\n\n${node.pipeline}\n\n${node.scenes}\n\n${downloadBody}`;
  write(node.url.slice(1)+'record.md',download);
  if (node.source.experimentOwner) write(`iterations/${node.source.id}/record.md`,download);
  const stages=records.filter(record=>record.parent===node.source.id);
  const archive = node.archived ? node : experiment.entry;
  const originalContent = `${node.reportLayout?`<div class="markdown">${articleBody}</div><section class="experiment-section"><h2>运行批次</h2>${renderBatchList(experiment,[node],esc)}</section>`:`${renderVersionSummary(experiment,node,esc)}<div class="experiment-summary-separator"><h2>方法与相关研究记录</h2></div><div class="markdown">${articleBody}</div>`}
    ${stages.length?`<section class="evidence-section"><h2>历史阶段记录</h2>${versionList(stages)}</section>`:''}
    <p class="article-foot"><a href="${node.url}record.md" download>下载本版实验记录</a></p>`;
  return page(`${node.code} · ${node.title}`,'iterations',`<div class="container article-layout has-todos">[[page-contents]]
    <article class="article"><header class="page-heading"><p class="eyebrow"><a href="/iterations/${experiment.entry.id}/">${esc(experiment.entry.title)}</a><span>迭代详情</span><span class="status">${esc(archive.archived?'已封存':node.status)}</span></p>
    <h1>${node.code} · ${esc(node.title)}</h1><p>${esc(node.resultSummary)}</p></header>
    ${renderVersionNavigation(experiment,node,esc)}${renderArchivedContent(archive,originalContent,esc)}</article>${versionTodos(node.source)}</div>`,true);
}

function experimentBatch(experiment,node,batch) {
  const url=batchUrl(experiment.entry.id,node.code,batch.id);
  const text=`# ${batch.title}\n\n版本：${node.code}\n\n${node.scenes}\n\n${batch.summary}\n\n${batch.outputs.map(path=>'- '+path).join('\n')}`;
  write(url.slice(1)+'record.md',text);
  const archive=node.archived ? node : experiment.entry;
  const originalContent=`${renderBatchDetails(experiment,node,batch,esc)}<p class="article-foot"><a href="${url}record.md" download>下载批次记录</a></p>`;
  return page(batch.title,'iterations',`<div class="container article-layout has-todos">[[page-contents]]<article class="article">
    <header class="page-heading"><p class="eyebrow"><a href="/iterations/${experiment.entry.id}/">${esc(experiment.entry.title)}</a><a href="${node.url}">${node.code}</a><span>运行批次</span><time>${esc(batch.date)}</time></p><h1>${esc(batch.title)}</h1></header>
    ${renderArchivedContent(archive,originalContent,esc)}</article>${versionTodos(node.source)}</div>`,true);
}

function workflowStepPage(version, workflow, step) {
  const raw=step.markdown;
  write(`iterations/${version.id}/steps/${step.id}/record.md`,raw);
  const main=workflow.steps.filter(s=>/^\d+$/.test(s.number));
  const index=main.findIndex(s=>s.id===step.id);
  const previous=index>0?main[index-1]:null;
  const next=index>=0?main[index+1]:null;
  return page(`${version.title} · ${step.title}`,'iterations',`<div class="container article-layout workflow-detail">
    [[page-contents]]
    <article class="article"><header class="page-heading"><p class="eyebrow"><a href="/iterations/${version.id}/">${esc(version.title)}</a><span>流程步骤 ${esc(step.number)}</span><span class="status">${esc(step.status)}</span></p>
    <h1>${esc(step.title)}</h1><p>${esc(step.summary)}</p></header><div class="markdown">${markdown.render(raw)}</div>
    <nav class="workflow-pagination" aria-label="步骤导航">${previous?`<a href="${previous.href}">← 上一步：${esc(previous.title)}</a>`:'<span></span>'}
    ${next?`<a href="${next.href}">下一步：${esc(next.title)} →</a>`:`<a href="/iterations/${version.id}/#workflow">返回流程卡片</a>`}</nav>
    <div class="article-foot"><a href="./record.md" download>下载本步骤 Markdown</a></div></article></div>`, true);
}

const labelByModel = {
  '065-g_cups': '杯子', '065-b_cups': '杯子', '065-i_cups': '杯子', '065-j_cups': '杯子', '065-e_cups': '杯子',
  '036_wood_block': '木块', '002_master_chef_can': '罐头', '054_softball': '垒球', '044_flat_screwdriver': '一字螺丝刀',
  '073-e_lego_duplo': '蓝色积木', '053_mini_soccer_ball': '小足球', '051_large_clamp': '大夹具', '063-a_marbles': '弹珠'
};
function caseOutcome(row) {
  if (row.success) return ['rescued', '完成移障与重抓'];
  if (row.reason.includes('state_collision')) return ['retreat', '撤离起点碰撞'];
  if (row.reason.includes('insufficient_hold_height')) return ['hold', '挡物保持高度不足'];
  if (row.reason.includes('full_scene_path_collision')) return ['entry', '进场碰撞复核未通过'];
  if (row.reason.includes('抓取规划')) return ['plan', '挡物抓取规划未通过'];
  return ['other', `${row.phase ?? ''} · ${row.reason ?? ''}`];
}

function results() {
  const rescued = report.rows.find(row => row.success);
  const lift = (rescued.target_grasp.minimum_hold_lift_m * 1000).toFixed(2);
  const held = rescued.target_grasp.hold_seconds.toFixed(3);
  const rows = report.rows.map(r => {
    const [group, reason] = caseOutcome(r);
    const target = labelByModel[r.target_model] ?? r.target_model;
    const blocker = r.blocker_model === '065-i_cups' ? '绿杯' : labelByModel[r.blocker_model] ?? r.blocker_model;
    return `<tr data-case-row data-group="${group}"><th scope="row">${esc(r.case)}</th>
      <td>${esc(target)}</td><td>${esc(blocker)}</td>
      <td>${r.direct.success?'成功':'失败'}</td><td class="${r.success?'success-cell':''}">${esc(reason)}</td></tr>`;
  }).join('');
  return page('实验成果','results', `<div class="container results-page">
    <header class="page-heading"><h1>实验成果</h1><p>抓取管线的同场景对照与 CLARO 移障配对结果。</p></header>
    <section id="benchmark" class="result-section">
    <div class="section-heading"><div><h2>200 例同场景对照</h2><p>各版本使用相同的场景、目标和种子。实际抬高至少 1 厘米并连续保持 3 秒计为成功。</p></div></div>
    ${benchmarkTable()}
    <p class="research-table-note">表中为实际成功数及成功率，初始化失败计入分母。MPC 组同时采用非严格朝向搜索与闭环控制。</p></section>
    <section id="failures" class="result-section" aria-labelledby="failure-title">
    <div class="section-heading"><div><h2 id="failure-title">抓取在哪一步失败</h2><p>三个版本的失败分布，每例计入一个最终停止阶段。</p></div></div>
    ${failureTable()}
    <p class="research-table-note">MPC 的 14 例初始规划失败按候选到达的最深阶段归类。21 例动作未收敛包括腕部精调 9 例、闭爪 8 例、接近跟踪 2 例、交接 1 例、抬升 1 例。</p></section>
    <section id="demo" class="result-section" aria-labelledby="demo-title">
    <div class="section-heading"><div><h2 id="demo-title">三条管线的实际执行</h2><p>同一中度杂乱场景 I0001，目标为玩具飞机。</p></div></div>
    ${pipelineDemos()}</section>
    <section class="result-section" aria-labelledby="claro-result"><div class="section-heading"><div><h2 id="claro-result">CLARO 固定挡物配对</h2>
    <p>该组包含 ${report.validated_total} 个原先直接抓取失败的 S5 场景。对每个场景从相同初态比较直接抓取与移障重抓。</p></div><a href="/iterations/claro/">版本详情</a></div>
    <div class="summary-strip"><div><span>直接抓取</span><strong>${report.successful_direct}/${report.validated_total}</strong><small>成功场景</small></div>
    <div><span>移障后重抓</span><strong>${report.rescued}/${report.validated_total}</strong><small>救回 ${esc(rescued.case)}</small></div>
    <div><span>${esc(rescued.case)} 保持期最低抬升</span><strong>${lift} <em>毫米</em></strong><small>持续 ${held} 秒</small></div></div></section>
    <section id="comparison" class="result-section"><h2>I0069 对比录像</h2>
    <p>左侧展示直接抓取的端点拒绝，右侧展示移走绿杯后重新观测和实际重抓。</p>
    ${video('/media/i0069-comparison.mp4','直接抓取与移障重抓')}
    <p class="scope-note">片中 Agent 选挡来自独立验证。连续执行录像来自固定挡物配对。</p></section>
    <section id="before-after" class="result-section"><h2>同一腕部视角的变化</h2>
    <figure class="image-figure"><a class="image-original-link" href="/media/view-comparison.png" target="_blank" rel="noopener" title="在新标签页查看原图"><img src="/media/view-comparison.png" alt="腕部观测前后对照，黄色区域为积木的可见掩码" loading="lazy"></a>
    <figcaption>左为移障前，右为移障后。黄色区域是原目标积木的可见掩码。</figcaption></figure>
    <div class="split-metrics"><span>目标遮挡率 <b>22.8% → 0%</b></span><span>有效深度点 <b>2490 → 3226</b></span></div></section>
    <section id="cases" class="result-section"><div class="section-heading"><div><h2>逐场景结果</h2><p>按场景编号、目标或挡物查找，也可按停止阶段筛选。</p></div></div>
    <p class="scope-note"><a href="/iterations/claro/">12 个失败场景的具体原因与后续定位</a>已补充到 CLARO 版本主页；下表仍保留原实测结果，离线规划恢复不改记为实际成功。</p>
    <div class="filters"><label>搜索场景<input type="search" id="case-search" placeholder="例如 I0069 或绿杯"></label>
    <label>结果阶段<select id="case-outcome"><option value="">全部阶段</option><option value="rescued">完成重抓</option>
    <option value="plan">抓取规划失败</option><option value="entry">进场碰撞</option>
    <option value="hold">保持高度不足</option><option value="retreat">撤离起点碰撞</option></select></label>
    <span id="case-count" aria-live="polite">${report.validated_total} 个场景</span></div>
    <div class="table-scroll results-table" tabindex="0"><table><thead><tr>
    <th scope="col">场景</th><th scope="col">原目标</th><th scope="col">挡物</th><th scope="col">直接抓取</th><th scope="col">移障分支结果</th>
    </tr></thead><tbody>${rows}</tbody></table></div>
    <p id="empty-cases" class="scope-note" hidden>没有符合这组筛选条件的场景。</p></section>
    <section class="result-section" aria-labelledby="version-results-title">
    <div class="section-heading"><h2 id="version-results-title">各版本成果</h2></div>
    ${versionList(versions)}</section>
  </div>`);
}

const output = mkdtempSync(join(resources, '.build-'));
try {
  mkdirSync(join(output, 'media'));
  if (workflows.size) {
    const bundled = await bundle({
      entryPoints: { diagram: join(root,'页面/交互流程图.jsx'), loader: join(root,'页面/流程图入口.mjs') },
      outdir: join(output,'assets'), entryNames: '[name]-[hash]', outExtension: {'.js':'.mjs'},
      bundle: true, format: 'esm', target: ['es2022'], minify: true, write: false,
      metafile: true, define: {'process.env.NODE_ENV': '"production"'}, legalComments: 'eof',
    });
    for (const file of bundled.outputFiles) {
      mkdirSync(dirname(file.path),{recursive:true});
      writeFileSync(file.path,file.contents);
      writeFileSync(file.path+'.gz',gzipSync(file.contents,{level:9}));
    }
    const names=bundled.outputFiles.map(file=>basename(file.path));
    const asset=(prefix,extension)=>{
      const name=names.find(name=>name.startsWith(prefix)&&name.endsWith(extension));
      if (!name) throw new Error('流程图构建产物缺失：'+prefix+extension);
      return '/assets/'+name;
    };
    for (const workflow of workflows.values()) {
      const json=JSON.stringify(workflowGraphData(workflow));
      const hash=createHash('sha256').update(json).digest('hex').slice(0,12);
      const data=`assets/workflow-${workflow.versionId}-${hash}.json`;
      write(data,json);
      const source=media.get(basename(workflow.diagram));
      if (!source) throw new Error('流程图未登记：'+workflow.diagram);
      const fallback=readFileSync(source,'utf8').trim().replace('role="img"','role="group"');
      if (!fallback.startsWith('<svg ') || /<script|\son\w+\s*=/i.test(fallback)) throw new Error('流程图后备内容无效');
      workflowAssets.set(workflow.versionId,{
        client:asset('diagram-','.mjs'), loader:asset('loader-','.mjs'), stylesheet:asset('diagram-','.css'),
        data:'/'+data, fallback,
      });
    }
  }
  for (const [name, source] of media) cpSync(source, join(output, 'media', name));
  if (upgradePlan) {
    for (const document of upgradePlan.pages) {
      write(document.url.slice(1)+'index.html',page(document.title,'roadmap',`<div class="container article-layout">
        [[page-contents]]<article class="article upgrade-page">
        <header class="page-heading"><p class="eyebrow"><a href="/#roadmap">项目概览</a> / <a href="/roadmap/">未来升级计划</a></p>
          <h1>${esc(document.title)}</h1></header><div class="markdown">${document.html}</div>
        <p class="article-foot"><a href="${document.download}" download>下载 Markdown 原文</a></p></article></div>`,true));
    }
    for (const [name,source] of upgradePlan.files) {
      mkdirSync(join(output,'roadmap/files'),{recursive:true});
      cpSync(snapshotFile('attachments/'+name,source),join(output,'roadmap/files',name));
    }
  }
  for (const name of ['cover.jpg','comparison-cover.jpg']) cpSync(snapshotFile('media/'+name,join(root,'页面',name)),join(output,'media',name));
  for (const name of ['site.css','research.css','reading.css','experiments.css','site.mjs','page-toc.mjs','todos.mjs','notosc.woff','notosc-OFL.txt']) cpSync(join(root,'页面',name),join(output,name));
  write('version-tree.css', tree.css);
  write('index.html', overview());
  write('iterations/index.html', iterationsPage());
  for (const v of [...versions,...archivedVersions,...records]) write(`iterations/${v.id}/index.html`, iteration(v));
  for (const experiment of experiments.values()) for (const node of experiment.iterations) {
    const html=experimentIteration(experiment,node);
    write(node.url.slice(1)+'index.html',html);
    if (node.source.experimentOwner) write(`iterations/${node.source.id}/index.html`,html);
    for (const batch of node.batches) write(batchUrl(experiment.entry.id,node.code,batch.id).slice(1)+'index.html',experimentBatch(experiment,node,batch));
  }
  for (const version of registeredVersions) {
    const workflow=workflows.get(version.id);
    if (workflow) for (const step of workflow.steps) write(`iterations/${version.id}/steps/${step.id}/index.html`, workflowStepPage(version,workflow,step));
  }
  write('results/index.html',results());
  write('configuration/index.html', configurationPage());
  for(const record of configurationRecords)write(`configuration/${record.id}/index.html`,configurationDetail(record));
  write('papers/index.html', papersPage());
  for (const paper of literature.papers) write(`papers/${paper.id}/index.html`, paperPage(paper));
  copyOptimizedMedia(optimizedMedia,resources,output);
  const textCompression = compressText(output,readJson(join(root,'media-policy.json')).text);
  writeFileSync(join(resources,'optimized/build-report.json'),JSON.stringify({text:textCompression},null,2)+'\n');
  saveManifest();
  const dest = join(resources,'site');
  const backup = mkdtempSync(join(resources, '.build-previous-'));
  const saved = join(backup, 'dist');
  if (existsSync(dest)) renameSync(dest, saved);
  try { renameSync(output, dest); }
  catch (error) {
    if (existsSync(saved)) renameSync(saved, dest);
    throw error;
  }
  rmSync(backup, { recursive: true, force: true });
  const stepCount=[...workflows.values()].reduce((n,w)=>n+w.steps.length,0);
  console.log(`已生成 ${experiments.size} 个实验主页、${[...experiments.values()].reduce((n,e)=>n+e.iterations.length,0)} 个内部版本、${records.length} 个阶段记录页、${stepCount} 个流程步骤页`);
} catch (error) {
  if (existsSync(output)) rmSync(output,{recursive:true,force:true});
  throw error;
}
