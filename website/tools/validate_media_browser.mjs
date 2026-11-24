// Inspect real browser-selected image URLs and video loading over a CDP pipe.
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import assert from 'node:assert/strict';

const base = process.argv[2] ?? 'http://127.0.0.1:8786';
const output = resolve(process.argv[3] ?? '../tmp/website/media_optimization_20261124/browser');
const binary = process.env.ARX_BROWSER;
if (!binary) throw new Error('Set ARX_BROWSER to a Chromium headless-shell executable');
const browser = spawn(binary, ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--no-first-run', '--hide-scrollbars', '--window-size=1400,1100', '--remote-debugging-pipe'],
  {stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe']});
let next = 0, buffer = Buffer.alloc(0), errors = '', session;
const pending = new Map();
browser.stderr.on('data', data => { errors += data; });
browser.stdio[3].on('error', () => {});
browser.stdio[4].on('error', () => {});
browser.stdio[4].on('data', chunk => {
  buffer = Buffer.concat([buffer, chunk]);
  let end;
  while ((end = buffer.indexOf(0)) >= 0) {
    const message = JSON.parse(buffer.subarray(0, end)); buffer = buffer.subarray(end + 1);
    const item = pending.get(message.id);
    if (!item) continue;
    pending.delete(message.id); clearTimeout(item.timer);
    if (message.error) item.reject(new Error(JSON.stringify(message.error)));
    else item.resolve(message.result);
  }
});
browser.on('exit', code => {
  for (const item of pending.values()) { clearTimeout(item.timer); item.reject(new Error(`Browser exited ${code}: ${errors.slice(-500)}`)); }
  pending.clear();
});
function rpc(method, params = {}, target) {
  const id = ++next;
  return new Promise((accept, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error('CDP timeout ' + method + ': ' + errors.slice(-600))); }, 20000);
    pending.set(id, {resolve: accept, reject, timer});
    browser.stdio[3].write(JSON.stringify({id, method, params, ...(target ? {sessionId: target} : {})}) + '\0');
  });
}
const pause = ms => new Promise(accept => setTimeout(accept, ms));
async function evaluate(expression) {
  const response = await rpc('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true}, session);
  if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
}
async function navigate(path) {
  await rpc('Page.navigate', {url: base + path}, session);
  for (let i = 0; i < 100; i++) {
    await pause(100);
    if (await evaluate(`location.pathname===${JSON.stringify(path)} && document.readyState==='complete'`)) return;
  }
  throw new Error('Page did not load: ' + path);
}

mkdirSync(output, {recursive: true});
const checks = [];
try {
  const {targetId} = await rpc('Target.createTarget', {url: 'about:blank'});
  session = (await rpc('Target.attachToTarget', {targetId, flatten: true})).sessionId;
  await rpc('Page.enable', {}, session);
  for (const [path, selector, name] of [
    ['/iterations/pointcloud-models/', '.markdown table:has(img[src$="pointcloud-comparison-c001-thumb.jpg"])', 'pointcloud'],
    ['/configuration/scene-gallery/', '.scene-montage', 'scene-gallery']
  ]) {
    await navigate(path);
    assert(await evaluate(`Boolean(document.querySelector(${JSON.stringify(selector)}))`));
    await evaluate(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({block:'start'})`);
    await pause(700);
    const images = await evaluate(`Promise.all([...document.querySelector(${JSON.stringify(selector)}).querySelectorAll('img')].map(async image=>{
      image.loading='eager'; await image.decode();
      return {src:image.getAttribute('src'), selected:image.currentSrc, width:image.naturalWidth, height:image.naturalHeight};
    }))`);
    assert(images.length > 0 && images.every(image => image.width > 0));
    assert(images.some(image => image.selected.includes('/optimized/') && image.selected.endsWith('.webp')));
    if (name === 'pointcloud') {
      const layout = await evaluate(`({display:getComputedStyle(document.querySelector(${JSON.stringify(selector)}+' tr')).display,
        images:[...document.querySelector(${JSON.stringify(selector)}).querySelectorAll('img')].map(e=>({w:e.getBoundingClientRect().width,h:e.getBoundingClientRect().height}))})`);
      assert.equal(layout.display, 'grid');
      checks.push({page: path, images, layout});
    } else {
      const link = await evaluate("document.querySelector('.scene-hotspots a').href");
      assert(link.includes('/media/') && link.endsWith('.jpg'));
      const response = await evaluate(`fetch(${JSON.stringify(link)},{method:'HEAD'}).then(r=>r.status)`);
      assert.equal(response, 200);
      checks.push({page: path, images, original_link: link, original_status: response});
    }
    const screenshot = await rpc('Page.captureScreenshot', {format: 'png'}, session);
    writeFileSync(resolve(output, name + '.png'), Buffer.from(screenshot.data, 'base64'));
  }
  await navigate('/iterations/mpc/versions/V1/');
  const videos = await evaluate("[...document.querySelectorAll('video')].map(v=>({preload:v.preload,networkState:v.networkState}))");
  assert(videos.length > 0 && videos.every(video => video.preload === 'none'));
  const earlyVideoTransfers = await evaluate("performance.getEntriesByType('resource').filter(e=>e.name.endsWith('.mp4')).length");
  assert.equal(earlyVideoTransfers, 0);
  checks.push({page: '/iterations/mpc/versions/V1/', videos, earlyVideoTransfers});
  writeFileSync(resolve(output, 'checks.json'), JSON.stringify({passed: true, checks}, null, 2) + '\n');
  console.log('Browser media checks passed');
} finally {
  browser.kill('SIGTERM');
}
