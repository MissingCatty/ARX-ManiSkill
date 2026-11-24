import { existsSync, readFileSync, realpathSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, resolve, sep } from 'node:path';
import { createHash } from 'node:crypto';
import { resources, snapshotFile } from './published-resources.mjs';
import { sourcePath } from './project-paths.mjs';

export function readUpgradePlan(root, settings, markdown) {
  const cache=resolve(resources,'metadata/roadmap.json');
  if (!existsSync(resolve(root,settings.source))) {
    const saved=JSON.parse(readFileSync(cache,'utf8'));
    return {...saved,files:new Map(saved.files.map(([name,path])=>[name,resolve(resources,path)]))};
  }
  const source = realpathSync(resolve(root, settings.source));
  const mapRoot = realpathSync(resolve(root, '../modules/reachability'));
  const docsRoot = realpathSync(resolve(root, '../modules/maniskill/docs'));
  const known = [
    {source, title:'未来升级计划', url:'/roadmap/'},
    {source:resolve(mapRoot,'实验/升降联合规划_20261120_01/实施方案.md'), title:'升降联合规划离线验证', url:'/roadmap/lift-joint/'},
    {source:resolve(mapRoot,'docs/可达性地图/下游位姿修正网络必要性评估.md'), title:'抓取位姿修正的已有分析', url:'/roadmap/grasp-refinement/'},
    {source:resolve(mapRoot,'docs/可达性地图/抓取位姿修正调研.md'), title:'可达性地图引导抓取位姿修正调研', url:'/roadmap/grasp-refinement-research/'},
    {source:resolve(mapRoot,'docs/可达性地图/抓取位姿局部修正首轮结果.md'), title:'抓取位姿局部修正首轮离线结果', url:'/roadmap/grasp-refinement-pilot/'},
    {source:resolve(mapRoot,'docs/可达性地图/前方精细地图与新旧流程评估.md'), title:'前方精细地图与新旧流程评估', url:'/roadmap/grasp-refinement-assessment/'},
    {source:resolve(mapRoot,'docs/可达性地图/先IK后地图修正流程.md'), title:'先 IK 后地图修正流程', url:'/roadmap/grasp-refinement-ik-first/'},
    {source:resolve(mapRoot,'docs/可达性地图/关节空间投影首轮验证.md'), title:'关节空间投影首轮离线验证', url:'/roadmap/joint-projection-pilot/'},
    {source:resolve(mapRoot,'docs/可达性地图/选定高度抓取对照实验.md'), title:'固定高度对照结果', url:'/roadmap/fixed-height/'},
    {source:resolve(docsRoot,'GraspCL示教兼容修复与160场景对照.md'), title:'GraspCL 示教兼容与160场景对照', url:'/roadmap/graspcl-compat/'}
  ];
  const documents = known.map(d=>{
    const actual=realpathSync(sourcePath(d.source));
    return {...d,source:actual,body:readFileSync(actual,'utf8')};
  });
  const routes = new Map(documents.map(d=>[d.source,d.url]));
  const files = new Map();
  const safeRoots = [mapRoot,docsRoot,realpathSync(resolve(root,'../tmp/reachability')),realpathSync(resolve(root,'../tmp/maniskill/docs_resources'))];
  function fileLink(file) {
    const actual=realpathSync(file);
    if (!safeRoots.some(folder=>actual.startsWith(folder+sep))
      || !['.md','.json','.csv','.png','.jpg','.pdf'].includes(extname(actual))
      || statSync(actual).size>10*1024*1024) throw new Error('升级计划引用超出允许范围：'+file);
    const hash=createHash('sha256').update(readFileSync(actual)).digest('hex').slice(0,12);
    const name=hash+'-'+basename(actual);
    files.set(name,actual);
    return '/roadmap/files/'+encodeURIComponent(name);
  }
  function resolveLink(document, href, summary=false) {
    if (!href || /^(?:https?:|mailto:|\/)/.test(href)) return href;
    if (href.startsWith('#')) return summary ? '/roadmap/'+href : href;
    const hashAt=href.indexOf('#');
    const path=hashAt<0?href:href.slice(0,hashAt), fragment=hashAt<0?'':href.slice(hashAt);
    const target=sourcePath(resolve(dirname(document.source),decodeURIComponent(path)));
    if (!existsSync(target)) throw new Error('升级计划引用不存在：'+target);
    const actual=realpathSync(target);
    return (routes.get(actual) ?? fileLink(actual))+fragment;
  }
  const body=documents[0].body;
  const table=body.match(/^\|[^\n]+\|\r?\n\|\s*:?-[-| :]+\|\r?\n(?:\|[^\n]+\|\r?\n?)+/m)?.[0];
  if (!table) throw new Error('未来升级计划缺少阶段进度表');
  const env=(document,summary=false)=>({
    resolveUpgradeLink:href=>resolveLink(document,href,summary),
    upgradeAnchors:{'阶段与状态':'stages','可达性引导的抓取位姿局部修正':'grasp-refinement','头部初始感知与腕部近场复核':'head-wrist','地图引导修正的具体方法':'map-guided-method'}
  });
  const summary=markdown.render(table,env(documents[0],true));
  const pages=documents.map(d=>({
    ...d, html:markdown.render(d.body.replace(/^# [^\n]+\n+/,''),env(d)), download:fileLink(d.source)
  }));
  const savedFiles=[...files].map(([name,path])=>{
    snapshotFile('attachments/'+name,path);
    return [name,'attachments/'+name];
  });
  writeFileSync(cache,JSON.stringify({summary,pages,files:savedFiles,source},null,2)+'\n');
  return {summary,pages,files,source};
}
