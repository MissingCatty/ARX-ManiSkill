import { readFileSync } from 'node:fs';
import { join, basename } from 'node:path';

export function readExperimentGallery(content, version) {
  if (!version.gallery) return null;
  if (basename(version.gallery) !== version.gallery) throw new Error('实验图集文件名无效');
  const data=JSON.parse(readFileSync(join(content,version.gallery),'utf8'));
  if (!data.cases?.length || data.cases.length!==data.columns*data.rows
    || data.columns!==5 || data.rows!==2) throw new Error('实验图集应有十个场景');
  for (const item of data.cases) {
    if (!item.id || !item.caption || !/^\/media\/scene-[a-z0-9]+-[a-f0-9]{12}\.jpg$/.test(item.url)) {
      throw new Error('实验图集缺少可追溯的大图：'+item.id);
    }
  }
  return data;
}

export function renderExperimentGallery(data, esc) {
  return `<div class="scene-gallery experiment-gallery">
    <p class="experiment-gallery-hint">十个代表场景：点击单格，在新标签页查看初始画面和两条链路的实际结束画面。</p>
    <div class="scene-scroll" tabindex="0" role="region" aria-label="十个代表场景的缩略拼图"><div class="scene-montage experiment-montage">
      <img src="${esc(data.sheet.url)}" alt="${esc(data.sheet.alt)}" loading="lazy" decoding="async" width="${data.sheet.width}" height="${data.sheet.height}">
      <div class="scene-hotspots experiment-hotspots">${data.cases.map(item=>`<a href="${esc(item.url)}" target="_blank" rel="noopener" data-scene-title="${esc(item.id)}" data-scene-caption="${esc(item.caption)}" aria-label="在新标签页放大查看 ${esc(item.id)}：${esc(item.caption)}" title="${esc(item.id)}：${esc(item.caption)} · 在新标签页查看原图"></a>`).join('')}</div>
    </div></div>
  </div>`;
}
