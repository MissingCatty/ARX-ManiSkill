// Exercise the real canvas in Chromium over its local debugging pipe.
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const out=resolve(root,'数据/验收/并列主线_20261122');
const binary=process.env.ARX_BROWSER ?? '/home/zhaoyechong/.cache/ms-playwright/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell';
const browser=spawn(binary,['--no-sandbox','--disable-gpu','--hide-scrollbars','--window-size=1400,1100','--remote-debugging-pipe'],{stdio:['ignore','ignore','pipe','pipe','pipe']});
let next=0,buffer=Buffer.alloc(0),errors='';const pending=new Map();
browser.stderr.on('data',data=>{errors+=data.toString();});
browser.stdio[4].on('data',chunk=>{buffer=Buffer.concat([buffer,chunk]);let end;while((end=buffer.indexOf(0))>=0){const message=JSON.parse(buffer.subarray(0,end).toString());buffer=buffer.subarray(end+1);if(message.id&&pending.has(message.id)){const p=pending.get(message.id);pending.delete(message.id);clearTimeout(p.timer);message.error?p.reject(new Error(JSON.stringify(message.error))):p.resolve(message.result);}}});
browser.on('exit',code=>{for(const p of pending.values()){clearTimeout(p.timer);p.reject(new Error(`Browser exited ${code}: ${errors.slice(-600)}`));}pending.clear();});
function rpc(method,params={},sessionId){const id=++next;return new Promise((resolvePromise,reject)=>{const timer=setTimeout(()=>{pending.delete(id);reject(new Error('CDP timeout '+method));},12000);pending.set(id,{resolve:resolvePromise,reject,timer});browser.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+'\0');});}
const pause=ms=>new Promise(r=>setTimeout(r,ms));

mkdirSync(out,{recursive:true});
const checks=[];
const expectedVersions=JSON.parse(readFileSync(resolve(root,'内容/网站.json'),'utf8')).versions.filter(v=>!v.experimentOwner&&!v.archived);
let todo=null,sessionId;
const navigate=async path=>{
  await rpc('Page.navigate',{url:'http://127.0.0.1:8786'+path},sessionId);
  for(let i=0;i<60;i++){
    await pause(100);
    const ok=await evaluate(`location.pathname===${JSON.stringify(path)} && document.readyState==='complete'`);
    if(ok)return;
  }
  throw new Error('Navigation timeout '+path);
};
const evaluate=async expression=>{
  const r=await rpc('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true},sessionId);
  if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
};
try{
  const target=await rpc('Target.createTarget',{url:'about:blank'});
  ({sessionId}=await rpc('Target.attachToTarget',{targetId:target.targetId,flatten:true}));
  await rpc('Page.enable',{},sessionId);
  await navigate('/');
  assert.equal(await evaluate("document.querySelectorAll('.tree-node').length"),expectedVersions.length);
  const roots=await evaluate("[...document.querySelectorAll('.tree-node-main')].map(e=>({id:e.dataset.version,parent:e.dataset.parent,y:e.getBoundingClientRect().top}))");
  assert.deepEqual(roots.map(r=>r.id),['baseline','sam3-graspgenx']);
  assert(roots.every(r=>r.parent===''));
  assert(Math.abs(roots[0].y-roots[1].y)<2);
  assert.equal(await evaluate("document.querySelectorAll('[data-version=graspgen]').length"),0);
  assert.equal(await evaluate("document.querySelector('[data-version=\"pointcloud-models\"]').dataset.parent"),'sam3-graspgenx');
  checks.push('首页节点与登记一致、2个主根同层、点云实验归属正确，无旧GraspGenX实验节点');
  await evaluate("document.querySelector('#iterations').scrollIntoView({block:'center'})");
  await pause(150);
  let screenshot=await rpc('Page.captureScreenshot',{format:'png'},sessionId);
  writeFileSync(resolve(out,'并列主线版本树.png'),Buffer.from(screenshot.data,'base64'));
  await evaluate("document.querySelector('[data-version=\"sam3-graspgenx\"]').click()");
  for(let i=0;i<60;i++){
    await pause(100);
    if(await evaluate("location.pathname==='/iterations/sam3-graspgenx/' && Boolean(document.querySelector('[data-workflow-widget]'))"))break;
  }
  await evaluate("document.querySelector('[data-workflow-widget]').scrollIntoView({block:'center'})");
  for(let i=0;i<60;i++){
    await pause(100);
    if(await evaluate("location.pathname==='/iterations/sam3-graspgenx/' && Boolean(document.querySelector('[data-workflow-canvas]:not([hidden])'))"))break;
  }
  assert.equal(await evaluate("location.pathname"),'/iterations/sam3-graspgenx/');
  assert.equal(await evaluate("document.querySelectorAll('[data-workflow-step]').length"),6);
  assert(!await evaluate("document.querySelector('.workflow-legend').textContent.includes('13例')"));
  await evaluate("document.querySelector('[data-workflow-step=segmentation]').click()");
  await pause(200);
  assert.match(await evaluate("document.querySelector('#workflow-detail-title')?.textContent"),/SAM3/);
  assert(await evaluate("document.querySelector('.workflow-dialog').open"));
  await evaluate("document.querySelector('[aria-label=\"关闭步骤详情\"]').click()");
  checks.push('新主线独立页面、6个交互步骤、SAM3详情弹窗及独立进度');
  await evaluate("document.querySelector('[data-workflow-widget]').scrollIntoView({block:'center'});[...document.querySelectorAll('.workflow-tools button')].find(e=>e.textContent==='全图').click()");
  await pause(400);
  screenshot=await rpc('Page.captureScreenshot',{format:'png'},sessionId);
  writeFileSync(resolve(out,'新主线交互流程.png'),Buffer.from(screenshot.data,'base64'));
  const links=await evaluate("[...document.querySelectorAll('.version-papers a')].map(e=>e.getAttribute('href'))");
  for(const p of ['/papers/sam3/','/papers/graspgenx/','/papers/curobo/'])assert(links.includes(p));
  assert.equal(await evaluate("document.querySelector('[data-todo-version]').dataset.todoVersion"),'sam3-graspgenx');
  const added=await evaluate("fetch('/api/todos',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({version:'sam3-graspgenx',text:'自动验收临时项 '+Date.now()})}).then(async r=>({status:r.status,data:await r.json()}))");
  assert.equal(added.status,201);todo=added.data;
  const changed=await evaluate(`fetch('/api/todos',{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({version:'sam3-graspgenx',id:${todo.id},completed:true})}).then(async r=>({status:r.status,data:await r.json()}))`);
  assert.equal(changed.status,200);assert.equal(changed.data.completed,1);
  const removed=await evaluate(`fetch('/api/todos',{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify({version:'sam3-graspgenx',id:${todo.id}})}).then(r=>r.status)`);
  assert.equal(removed,200);todo=null;
  checks.push('新主线论文链接及独立待办增、改、删通过，临时项已删除');
  assert.equal((await fetch('http://127.0.0.1:8786/iterations/graspgen/')).status,404);
  for(const p of ['/iterations/baseline/','/iterations/claro/','/iterations/v0.4/'])assert.equal((await fetch('http://127.0.0.1:8786'+p)).status,200);
  for(const p of ['/iterations/joint9/','/iterations/reachability-improvement/versions/V4/'])assert.equal((await fetch('http://127.0.0.1:8786'+p)).status,404);
  checks.push('旧实验入口404，原主线及既有实验页面保留');
  await navigate('/iterations/pointcloud-models/');
  assert.match(await evaluate("document.querySelector('h1').textContent"),/点云生成模型/);
  assert.equal(await evaluate("document.querySelector('.parent-version a').getAttribute('href')"),'/iterations/sam3-graspgenx/');
  assert.equal(await evaluate("document.querySelector('[data-todo-version]').dataset.todoVersion"),'pointcloud-models');
  assert(await evaluate("document.querySelector('.markdown').textContent.includes('实际运行0次')"));
  checks.push('点云生成评测独立实验页、父版本链接、待办与未运行状态通过');
  await rpc('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true},sessionId);
  for(const path of ['/','/iterations/sam3-graspgenx/','/iterations/pointcloud-models/','/papers/graspgenx/']){
    await navigate(path);
    assert(await evaluate('document.documentElement.scrollWidth<=window.innerWidth'),path+' mobile overflow');
  }
  checks.push('首页、新主线及新论文页手机布局通过');
  writeFileSync(resolve(out,'浏览器验收.json'),JSON.stringify({passed:true,checks,temporary_todo_removed:todo===null},null,2)+'\n');
  console.log('并列主线浏览器验收通过',checks.length);
}finally{
  if(todo&&sessionId)await evaluate(`fetch('/api/todos',{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify({version:'sam3-graspgenx',id:${todo.id}})})`).catch(()=>{});
  await rpc('Browser.close').catch(()=>{});browser.kill();
}
