import { readFileSync, realpathSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';

export function readWorkflow(content, version) {
  if (!version.workflow) return null;
  if (!/^[^/\\]+\.json$/.test(version.workflow)) throw new Error('流程登记文件名无效');
  const root=realpathSync(join(content,'流程'));
  const config=JSON.parse(readFileSync(join(root,version.workflow),'utf8'));
  const directory=realpathSync(resolve(root,config.directory));
  if (!directory.startsWith(root+sep) || !config.steps?.length || !config.groups?.length) throw new Error('流程目录或内容无效');
  const ids=new Set();
  for (const step of config.steps) {
    if (!/^[a-z][a-z0-9-]*$/.test(step.id) || ids.has(step.id) || !/^[^/\\]+\.md$/.test(step.file)
      || !['input','vision','geometry','execute','fixed','pending'].includes(step.kind)
      || ![step.number,step.title,step.summary,step.status].every(x=>typeof x==='string'&&x.trim())) throw new Error('流程步骤无效：'+step.id);
    ids.add(step.id);
    const source=realpathSync(join(directory,step.file));
    if (!source.startsWith(directory+sep)) throw new Error('步骤文档超出内容目录');
    step.markdown=readFileSync(source,'utf8');
    step.href=`/iterations/${version.id}/steps/${step.id}/`;
  }
  const used=[];
  for (const group of config.groups) {
    if (!['sequence','branches','pending'].includes(group.layout)) throw new Error('流程分组无效');
    for (const lane of group.layout==='branches'?group.lanes:[group]) {
      if (!lane.steps?.length || lane.steps.some(id=>!ids.has(id))) throw new Error('流程引用了未知步骤');
      used.push(...lane.steps);
    }
  }
  if (used.length!==ids.size || new Set(used).size!==ids.size) throw new Error('流程步骤遗漏或重复');
  const { positions, edges } = config.graph ?? {};
  if (!positions || Object.keys(positions).length !== ids.size || [...ids].some(id =>
    !Array.isArray(positions[id]) || positions[id].length !== 2 || !positions[id].every(Number.isFinite))) {
    throw new Error('流程图节点坐标遗漏或无效');
  }
  const connections = new Set();
  if (!Array.isArray(edges) || !edges.length) throw new Error('流程图缺少连线');
  for (const edge of edges) {
    const key = `${edge.source}/${edge.target}`;
    if (!ids.has(edge.source) || !ids.has(edge.target) || edge.source === edge.target || connections.has(key)
      || edge.kind && !['fixed', 'pending'].includes(edge.kind)
      || edge.sourceHandle && !/^out-(bottom|left|right)$/.test(edge.sourceHandle)
      || edge.targetHandle && !/^in-(top|left|right)$/.test(edge.targetHandle)) throw new Error('流程图连线无效：' + key);
    connections.add(key);
  }
  return {...config,versionId:version.id};
}

export function workflowGraphData(workflow) {
  return {
    title: workflow.title, graph: workflow.graph,
    steps: workflow.steps.map(({id,number,title,summary,status,kind,href}) => ({id,number,title,summary,status,kind,href}))
  };
}

export function renderWorkflow(workflow,esc,{client,loader,stylesheet,data,fallback}) {
  const legend = workflow.legend ?? [
    {class:'vision',label:'视觉模块：首批3例验证'},
    {class:'fixed',label:'固定挡物：13例实测'},
    {class:'pending',label:'灰色虚线：待接入'}
  ];
  return `<section class="workflow" id="workflow" aria-label="${esc(workflow.title)}"><p class="workflow-intro">${esc(workflow.intro)}</p>
    <div class="workflow-legend">${legend.map(item=>`<span class="legend-${['vision','fixed','pending'].includes(item.class)?item.class:'pending'}">${esc(item.label)}</span>`).join('')}</div>
    <div class="workflow-widget" data-workflow-widget data-client="${esc(client)}" data-stylesheet="${esc(stylesheet)}" data-source="${esc(data)}">
      <div class="workflow-static" data-workflow-fallback>${fallback}</div>
      <div class="workflow-canvas" data-workflow-canvas hidden></div>
    </div><p class="workflow-load-state" data-workflow-state role="status">正在准备交互画布；图中的节点可直接打开步骤页面。</p>
    <p class="workflow-note">${esc(workflow.note)}</p><a class="workflow-vector" href="${esc(workflow.diagram)}" target="_blank" rel="noopener">打开矢量原图 ↗</a>
    <script type="module" src="${esc(loader)}"></script></section>`;
}

export function workflowMarkdown(workflow) {
  return workflow.steps.map(s=>`- [${s.number} ${s.title}](${s.href})：${s.summary}（${s.status}）`).join('\n');
}
