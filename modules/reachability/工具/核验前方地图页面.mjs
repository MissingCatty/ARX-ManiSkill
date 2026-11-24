// Exercise the real canvas in Chromium over its local debugging pipe.
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const out=resolve(root,'docs/可达性地图/resources/前方精细_01');
const binary=process.env.ARX_BROWSER ?? '/home/zhaoyechong/.cache/ms-playwright/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell';
const browser=spawn(binary,['--no-sandbox','--disable-gpu','--hide-scrollbars','--window-size=1400,1100','--remote-debugging-pipe'],{stdio:['ignore','ignore','pipe','pipe','pipe']});
let next=0,buffer=Buffer.alloc(0),errors='';const pending=new Map();
browser.stderr.on('data',data=>{errors+=data.toString();});
browser.stdio[4].on('data',chunk=>{buffer=Buffer.concat([buffer,chunk]);let end;while((end=buffer.indexOf(0))>=0){const message=JSON.parse(buffer.subarray(0,end).toString());buffer=buffer.subarray(end+1);if(message.id&&pending.has(message.id)){const p=pending.get(message.id);pending.delete(message.id);clearTimeout(p.timer);message.error?p.reject(new Error(JSON.stringify(message.error))):p.resolve(message.result);}}});
browser.on('exit',code=>{for(const p of pending.values()){clearTimeout(p.timer);p.reject(new Error(`Browser exited ${code}: ${errors.slice(-600)}`));}pending.clear();});
function rpc(method,params={},sessionId){const id=++next;return new Promise((resolvePromise,reject)=>{const timer=setTimeout(()=>{pending.delete(id);reject(new Error('CDP timeout '+method));},12000);pending.set(id,{resolve:resolvePromise,reject,timer});browser.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+'\0');});}
const pause=ms=>new Promise(r=>setTimeout(r,ms));
const checks=[];
try{
  const target=await rpc('Target.createTarget',{url:'about:blank'});
  const {sessionId}=await rpc('Target.attachToTarget',{targetId:target.targetId,flatten:true});
  const evaluate=async expression=>{const r=await rpc('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true},sessionId);if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  await rpc('Page.enable',{},sessionId);
  const paths=[['website','http://127.0.0.1:8786/media/reachability-front-map.html']];
  for(const [label,url] of paths){
    await rpc('Page.navigate',{url},sessionId);
    let ready=false;for(let i=0;i<40;i++){await pause(100);ready=await evaluate("Boolean(document.getElementById('count')?.textContent.includes('9,536'))");if(ready)break;}
    assert(ready,label+' did not render');
    assert(await evaluate("document.fonts.ready.then(()=>document.fonts.check('15px MapChinese'))"),label+' Chinese font failed');
    checks.push(label+':真实浏览器加载及默认9536格显示通过');
    const initial=await evaluate("document.getElementById('canvas').toDataURL()");
    await evaluate("document.getElementById('top').click()");await pause(100);
    assert.notEqual(await evaluate("document.getElementById('canvas').toDataURL()"),initial);
    checks.push(label+':俯视按钮改变画面');
    await evaluate("const s=document.getElementById('metric');s.value='6';s.dispatchEvent(new Event('change'))");await pause(100);
    assert.match(await evaluate("document.getElementById('scale-title').textContent"),/限位余量/);
    const at30=await evaluate("document.getElementById('count').textContent");
    await evaluate("const h=document.getElementById('height');h.value='10';h.dispatchEvent(new Event('input'))");await pause(100);
    assert.notEqual(await evaluate("document.getElementById('count').textContent"),at30);
    checks.push(label+':着色指标和高度过滤可交互');
    await evaluate("document.getElementById('height').value='30';document.getElementById('height').dispatchEvent(new Event('input'));document.getElementById('metric').value='4';document.getElementById('metric').dispatchEvent(new Event('change'));document.getElementById('reset').click()");await pause(100);
    const hit=await evaluate(`(()=>{const c=document.getElementById('canvas'),r=c.getBoundingClientRect(),im=c.getContext('2d').getImageData(0,0,c.width,c.height);for(let y=30;y<c.height-100;y+=2)for(let x=30;x<c.width-30;x+=2){const k=(y*c.width+x)*4;if(im.data[k]<100&&im.data[k+1]>75&&im.data[k+2]>60&&im.data[k+2]<180)return {x:r.left+x*r.width/c.width,y:r.top+y*r.height/c.height};}return null;})()`);
    assert(hit,'No colored voxel in canvas');
    await rpc('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...hit},sessionId);
    await rpc('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...hit},sessionId);await pause(100);
    assert.match(await evaluate("document.getElementById('detail').textContent"),/平均限位余量/);
    checks.push(label+':点击体素显示坐标及质量数值');
    assert(await evaluate("document.querySelector('#regions')===null && document.querySelector('#roi')===null"));
    checks.push('新版不含桌面筛选或桌面区域列表');
    if(label==='website'){
      const screenshot=await rpc('Page.captureScreenshot',{format:'png'},sessionId);
      writeFileSync(resolve(out,'交互地图预览.png'),Buffer.from(screenshot.data,'base64'));
    }
  }
  const page=await fetch('http://127.0.0.1:8786/configuration/');assert.equal(page.status,200);
  assert((await page.text()).includes('reachability-front-map.html'));
  checks.push('网站配置页入口及静态资源HTTP访问通过');
  const reference=JSON.parse(readFileSync(resolve(out,'展示依据.json')));
  writeFileSync(resolve(out,'页面核验.json'),JSON.stringify({passed:true,checks,physical_attempts:0,html_bytes:reference.html_bytes},null,2)+'\n');
  console.log('BROWSER_CHECKS_PASSED',checks.length);
}finally{await rpc('Browser.close').catch(()=>{});browser.kill();}
