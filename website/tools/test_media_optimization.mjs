import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { brotliDecompressSync, gunzipSync } from 'node:zlib';
import { prepareMedia, optimizeHtml, copyOptimizedMedia, compressText } from '../media-optimization.mjs';
import { preferredEncodings } from '../http-encoding.mjs';

const image = {kind: 'image', width: 1920, height: 1080, variants: [
  {file: 'optimized/media/image-small.webp', width: 480},
  {file: 'optimized/media/image-full.webp', width: 1920}
], files: [{file: 'optimized/media/image-small.webp'}]};
const manifest = {files: {'media/chart.jpg': image, 'attachments/图.png': image}};

test('responsive images preserve gallery CSS selectors and original links', () => {
  const input = '<a href="/media/chart.jpg"><img src="/media/chart.jpg" alt="TCP" width="1100"></a>';
  const html = optimizeHtml(input, manifest);
  assert.match(html, /href="\/media\/chart.jpg"/);
  assert.match(html, /src="\/media\/chart.jpg"/);
  assert.match(html, /srcset="\/optimized\/media\/image-small.webp 480w, \/optimized\/media\/image-full.webp 1920w"/);
  assert.equal([...html.matchAll(/\bwidth=/g)].length, 1);
  assert.match(html, /loading="lazy"/);
  assert.equal(optimizeHtml(html, manifest), html);
});

test('encoded attachment images and video posters use the same cache', () => {
  const input = '<img src="/roadmap/files/%E5%9B%BE.png"><video preload="metadata" poster="/media/chart.jpg"></video>';
  const html = optimizeHtml(input, manifest);
  assert.match(html, /srcset=/);
  assert.match(html, /poster="\/optimized\/media\/image-full.webp"/);
  assert.match(html, /preload="none"/);
  assert.doesNotMatch(html, /preload="metadata"/);
  assert.equal(optimizeHtml('<img src="/media/diagram.svg">', manifest), '<img src="/media/diagram.svg">');
});

test('text encoding negotiation honors explicit opt-outs and quality', () => {
  assert.deepEqual(preferredEncodings('br, gzip'), ['br', 'gzip']);
  assert.deepEqual(preferredEncodings('br;q=0, gzip'), ['gzip']);
  assert.deepEqual(preferredEncodings('br;q=0, gzip;q=0'), []);
  assert.deepEqual(preferredEncodings('br;q=0.3, gzip;q=0.9'), ['gzip', 'br']);
  assert.deepEqual(preferredEncodings('br;q=0, *;q=1'), ['gzip']);
  assert.deepEqual(preferredEncodings('br;q=0.5, identity;q=0.8'), []);
  assert.deepEqual(preferredEncodings(), []);
});

test('published video and webp are independent real copies; text compression roundtrips', () => {
  const temp = mkdtempSync(join(tmpdir(), 'arx-media-test-'));
  try {
    const resources = join(temp, 'resources'), output = join(temp, 'site');
    mkdirSync(join(resources, 'optimized/media'), {recursive: true});
    mkdirSync(join(resources, 'optimized/video'), {recursive: true});
    writeFileSync(join(resources, 'optimized/media/image-small.webp'), 'webp-copy');
    writeFileSync(join(resources, 'optimized/video/video.mp4'), 'mp4-copy');
    copyOptimizedMedia({files: {'media/chart.jpg': image,
      'media/demo.mp4': {kind:'video', published:{file:'optimized/video/video.mp4'}}}}, resources, output);
    rmSync(resources, {recursive:true});
    assert.equal(readFileSync(join(output, 'media/demo.mp4'), 'utf8'), 'mp4-copy');
    assert.equal(readFileSync(join(output, 'optimized/media/image-small.webp'), 'utf8'), 'webp-copy');
    const text = '<p>compressed example</p>'.repeat(300);
    writeFileSync(join(output, 'index.html'), text);
    const result = compressText(output, {minimum_bytes: 1024, gzip_level: 9, brotli_quality: 8});
    assert.equal(result.files, 1);
    assert.equal(gunzipSync(readFileSync(join(output, 'index.html.gz'))).toString(), text);
    assert.equal(brotliDecompressSync(readFileSync(join(output, 'index.html.br'))).toString(), text);
  } finally { rmSync(temp, {recursive:true, force:true}); }
});

test('removing a publication source cannot restore it from the media cache', () => {
  const temp = mkdtempSync(join(tmpdir(), 'arx-media-removed-'));
  try {
    for (const folder of ['media', 'attachments', 'optimized']) mkdirSync(join(temp, 'resources', folder), {recursive: true});
    writeFileSync(join(temp, 'media-policy.json'), '{}');
    const file = join(temp, 'resources/optimized/manifest.json');
    writeFileSync(file, JSON.stringify({files: {'media/deleted.mp4': {kind: 'video'}}}));
    assert.deepEqual(prepareMedia(temp).files, {});
    assert.deepEqual(JSON.parse(readFileSync(file, 'utf8')).files, {});
  } finally { rmSync(temp, {recursive: true, force: true}); }
});
