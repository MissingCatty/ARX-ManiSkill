import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { brotliCompressSync, constants, gzipSync } from 'node:zlib';

const hash = file => createHash('sha256').update(readFileSync(file)).digest('hex');
const escape = value => value.replaceAll('&', '&amp;').replaceAll('"', '&quot;');
const urlFor = file => '/' + file.split('/').map(encodeURIComponent).join('/');

export function prepareMedia(root) {
  const resources = join(root, 'resources');
  const policyHash = hash(join(root, 'media-policy.json'));
  const manifestFile = join(resources, 'optimized/manifest.json');
  const sources = new Set(['media', 'attachments'].flatMap(folder => readdirSync(join(resources, folder))
    .filter(name => /\.(?:jpe?g|png|mp4)$/i.test(name)).map(name => folder + '/' + name)));
  const load = () => {
    const manifest = existsSync(manifestFile) ? JSON.parse(readFileSync(manifestFile, 'utf8')) : {files: {}};
    const active = Object.entries(manifest.files).filter(([key]) => sources.has(key));
    // A cached video must never restore media deliberately removed from publication.
    if (active.length !== Object.keys(manifest.files).length) {
      manifest.files = Object.fromEntries(active);
      manifest.source_count = sources.size;
      writeFileSync(manifestFile, JSON.stringify(manifest, null, 2) + '\n');
    }
    return manifest;
  };
  const complete = manifest => [...sources].every(key => {
    const record = manifest.files[key];
    return record?.source_sha256 === hash(join(resources, key)) && record.policy_sha256 === policyHash
      && record.files.every(item => existsSync(join(resources, item.file)) && hash(join(resources, item.file)) === item.sha256);
  });
  let manifest = load();
  if (!complete(manifest)) {
    const candidates = [process.env.ARX_MEDIA_PYTHON, resolve(root, '../environments/simulation/bin/python'), 'python3'].filter(Boolean);
    const python = candidates.find(candidate => spawnSync(candidate, ['-c', 'from PIL import features; assert features.check("webp")'], {stdio: 'ignore'}).status === 0);
    if (!python) throw new Error('媒体压缩需要支持 WebP 的 Pillow；请设置 ARX_MEDIA_PYTHON。已发布的网站可直接运行。');
    const run = spawnSync(python, [join(root, 'tools/optimize_media.py'), '--root', root], {stdio: 'inherit'});
    if (run.status !== 0) throw new Error('媒体压缩失败，保留现有发布页面。');
    manifest = load();
    if (!complete(manifest)) throw new Error('媒体压缩缓存核验失败。');
  }
  return manifest;
}

function imageRecord(url, manifest) {
  const path = decodeURIComponent(url.split(/[?#]/, 1)[0]);
  const key = path.startsWith('/roadmap/files/') ? path.replace('/roadmap/files/', 'attachments/') : path.slice(1);
  return manifest.files[key]?.kind === 'image' ? manifest.files[key] : null;
}

export function optimizeHtml(html, manifest) {
  return html.replace(/<img\b[^>]*>/g, tag => {
    const source = tag.match(/\bsrc="([^"]+)"/)?.[1];
    if (!source) return tag;
    const record = imageRecord(source, manifest);
    if (!record || /\bsrcset=/.test(tag)) return tag;
    const variants = record.variants.map(item => `${urlFor(item.file)} ${item.width}w`).join(', ');
    // Keep src intact: gallery styles and links use its original filename.
    let additions = ` srcset="${escape(variants)}"`;
    if (!/\bsizes=/.test(tag)) additions += ' sizes="auto, (max-width: 800px) 100vw, 1100px"';
    if (!/\bwidth=/.test(tag)) additions += ` width="${record.width}"`;
    if (!/\bheight=/.test(tag)) additions += ` height="${record.height}"`;
    if (!/\bloading=/.test(tag)) additions += ' loading="lazy"';
    if (!/\bdecoding=/.test(tag)) additions += ' decoding="async"';
    return tag.replace(/\s*\/?>(?=$)/, additions + '>');
  }).replace(/<video\b[^>]*>/g, tag => {
    tag = tag.replace(/\s+preload="[^"]*"/, '');
    tag = tag.replace(/\bposter="([^"]+)"/, (original, url) => {
      const record = imageRecord(url, manifest);
      if (!record) return original;
      const variant = record.variants.find(item => item.width >= 960) ?? record.variants.at(-1);
      return `poster="${escape(urlFor(variant.file))}"`;
    });
    return tag.replace(/>$/, ' preload="none">');
  });
}

export function copyOptimizedMedia(manifest, resources, output) {
  for (const [key, record] of Object.entries(manifest.files)) {
    if (record.kind === 'video') {
      const destination = join(output, key);
      mkdirSync(dirname(destination), {recursive: true});
      cpSync(join(resources, record.published.file), destination);
    } else for (const item of record.files) {
      const destination = join(output, item.file);
      mkdirSync(dirname(destination), {recursive: true});
      cpSync(join(resources, item.file), destination);
    }
  }
}

export function compressText(output, policy) {
  let files = 0, before = 0, gzipBytes = 0, brotliBytes = 0;
  function walk(folder) {
    for (const item of readdirSync(folder, {withFileTypes: true})) {
      const file = join(folder, item.name);
      if (item.isDirectory()) { walk(file); continue; }
      if (!/\.(?:html|css|mjs|json|svg|csv|md|txt)$/.test(file) || statSync(file).size < policy.minimum_bytes) continue;
      const data = readFileSync(file);
      const gz = gzipSync(data, {level: policy.gzip_level});
      const br = brotliCompressSync(data, {params: {[constants.BROTLI_PARAM_QUALITY]: policy.brotli_quality}});
      if (gz.length >= data.length || br.length >= data.length) continue;
      writeFileSync(file + '.gz', gz); writeFileSync(file + '.br', br);
      files++; before += data.length; gzipBytes += gz.length; brotliBytes += br.length;
    }
  }
  walk(output);
  return {files, before, gzip_bytes: gzipBytes, brotli_bytes: brotliBytes};
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const manifest = prepareMedia(dirname(fileURLToPath(import.meta.url)));
  console.log('媒体缓存已就绪：' + Object.keys(manifest.files).length + ' 项');
}
