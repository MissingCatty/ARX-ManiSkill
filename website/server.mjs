import { createReadStream } from 'node:fs';
import { realpath, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { isIPv4 } from 'node:net';
import { networkInterfaces } from 'node:os';
import { extname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import multicastDns from 'multicast-dns';
import { handleTodos } from './待办.mjs';
import { preferredEncodings } from './http-encoding.mjs';

const root = fileURLToPath(new URL('./resources/site', import.meta.url));
const port = Number(process.env.CLARO_PORT ?? 8786);
const host = process.env.CLARO_HOST ?? '0.0.0.0';
const domain = (process.env.CLARO_MDNS_NAME ?? 'claro-arx.local').toLowerCase();
if (!isIPv4(host) || !Number.isInteger(port) || port < 1024 || port > 65535 || !/^[a-z0-9][a-z0-9-]*\.local$/.test(domain)) {
  throw new Error('Invalid CLARO_HOST, CLARO_PORT or CLARO_MDNS_NAME');
}

const types = new Map([
  ['.html','text/html; charset=utf-8'], ['.css','text/css; charset=utf-8'],
  ['.mjs','text/javascript; charset=utf-8'], ['.md','text/plain; charset=utf-8'], ['.txt','text/plain; charset=utf-8'],
  ['.json','application/json; charset=utf-8'], ['.csv','text/csv; charset=utf-8'],
  ['.mp4','video/mp4'], ['.jpg','image/jpeg'], ['.png','image/png'], ['.webp','image/webp'],
  ['.svg','image/svg+xml'], ['.woff','font/woff']
]);
const ipv4 = address => address.split('.').reduce((value, component) => (value << 8) | Number(component), 0);
const privateAddress = address => /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(address);
const localInterfaces = () => Object.values(networkInterfaces()).flat()
  .filter(item => item.family === 'IPv4' && !item.internal && privateAddress(item.address));
function sameSubnet(remote, iface) {
  const mask = ipv4(iface.netmask);
  return isIPv4(remote) && (ipv4(remote) & mask) === (ipv4(iface.address) & mask);
}
function send(res, status, body, headers = {}) {
  res.writeHead(status, { 'Content-Type': 'text/plain; charset=utf-8', ...headers });
  res.end(body);
}
function parseRange(range, size) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(range);
  if (!match || !size || (!match[1] && !match[2])) return null;
  const start = match[1] ? Number(match[1]) : Math.max(0, size - Number(match[2]));
  const end = match[1] && match[2] ? Math.min(size - 1, Number(match[2])) : size - 1;
  return Number.isSafeInteger(start) && Number.isSafeInteger(end) && start <= end
    ? { start, end } : null;
}

const http = createServer(async (req, res) => {
  const remote = req.socket.remoteAddress;
  if (remote !== '127.0.0.1' && !localInterfaces().some(item => sameSubnet(remote, item))) {
    return send(res, 403, '请从服务器所在的局域网访问。\n');
  }
  try {
    const url = new URL(req.url, 'http://localhost');
    const pathname = decodeURIComponent(url.pathname);
    if (pathname === '/api/todos') return await handleTodos(req, res, url);
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      return send(res, 405, '请求方法不受支持。\n', { Allow: 'GET, HEAD' });
    }
    if (pathname === '/healthz') return send(res, 200, 'ok\n');
    if (pathname.includes('\0') || pathname.includes('\\') || pathname.split('/').some(p => p.startsWith('.'))) {
      return send(res, 404, '页面不存在。\n');
    }
    const file = resolve(root, '.' + pathname);
    if (file !== root && !file.startsWith(root + sep)) return send(res, 404, '页面不存在。\n');
    let target = file;
    if ((await stat(target)).isDirectory()) {
      if (!pathname.endsWith('/')) {
        return send(res, 308, '', { Location: url.pathname + '/' + url.search });
      }
      target = join(target, 'index.html');
    }
    let actual = await realpath(target);
    const mime = types.get(extname(actual));
    if (!actual.startsWith(root + sep) || !mime) return send(res, 404, '页面不存在。\n');
    const bundledAsset = /^\/assets\/(?:diagram|loader)-[A-Z0-9]+\.(?:mjs|css)$/.test(pathname);
    let encoding = null;
    const compressible = /\.(?:html|css|mjs|json|svg|csv|md|txt)$/.test(actual);
    if (compressible && !req.headers.range) {
      for (const candidate of preferredEncodings(req.headers['accept-encoding'])) {
        try {
          const zipped = await realpath(actual+(candidate === 'br' ? '.br' : '.gz'));
          if (zipped.startsWith(root+sep)) { actual=zipped; encoding=candidate; break; }
        } catch (error) { if (error.code !== 'ENOENT') throw error; }
      }
    }
    const { size, mtimeMs } = await stat(actual);
    const etag = `W/"${size.toString(16)}-${mtimeMs.toString(16)}"`;
    const galleryImage = /^\/media\/(?:scene|sheet)-[a-z0-9]+-[a-f0-9]{12}\.jpg$/.test(pathname);
    const headers = {
      'Content-Type': mime,
      'Accept-Ranges': 'bytes',
      'Cache-Control': galleryImage || bundledAsset || pathname.startsWith('/optimized/') ? 'private, max-age=31536000, immutable' : 'no-cache',
      ETag: etag,
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'no-referrer',
      'Content-Security-Policy': "default-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"
    };
    if (compressible) headers.Vary = 'Accept-Encoding';
    if (encoding) headers['Content-Encoding'] = encoding;
    if (!req.headers.range && req.headers['if-none-match'] === etag) {
      res.writeHead(304, headers);
      return res.end();
    }
    const range = req.headers.range ? parseRange(req.headers.range, size) : null;
    if (req.headers.range && !range) {
      return send(res, 416, '请求的字节范围不可用。\n', { 'Content-Range': `bytes */${size}` });
    }
    if (range) headers['Content-Range'] = `bytes ${range.start}-${range.end}/${size}`;
    headers['Content-Length'] = range ? range.end - range.start + 1 : size;
    res.writeHead(range ? 206 : 200, headers);
    if (req.method === 'HEAD') return res.end();
    const stream = createReadStream(actual, range ?? undefined);
    stream.on('error', error => res.destroy(error));
    res.on('close', () => stream.destroy());
    stream.pipe(res);
  } catch (error) {
    if (error.code === 'ENOENT' || error.code === 'ENOTDIR' || error instanceof URIError) {
      return send(res, 404, '页面不存在。\n');
    }
    console.error('HTTP', error.code ?? error.name);
    send(res, 500, '暂时无法读取页面。\n');
  }
});

const responders = new Map();
function advertise(iface) {
  const mdns = multicastDns({ interface: iface.address, bind: '0.0.0.0', reuseAddr: true });
  const answer = { name: domain, type: 'A', ttl: 60, data: iface.address, flush: true };
  mdns.on('error', error => {
    console.error('mDNS', iface.address, error.code ?? error.message);
    responders.delete(iface.address);
    try { mdns.destroy(); }
    catch (closeError) {
      if (closeError.code !== 'ERR_SOCKET_DGRAM_NOT_RUNNING') throw closeError;
    }
  });
  mdns.on('ready', () => {
    mdns.respond([answer]);
    console.log(`mDNS ${domain} → ${iface.address}`);
  });
  mdns.on('query', (query, peer) => {
    if (!sameSubnet(peer.address, iface)) return;
    const matches = query.questions.filter(q =>
      q.name.replace(/\.$/, '').toLowerCase() === domain && (q.type === 'A' || q.type === 'ANY'));
    if (!matches.length) return;
    if (peer.port === 5353) mdns.respond([answer]);
    else mdns.respond({ id: query.id, questions: matches, answers: [{ ...answer, flush: false }] }, peer);
  });
  return mdns;
}

function refreshNetwork() {
  const active = localInterfaces();
  for (const [ip, responder] of responders) {
    if (active.some(item => item.address === ip)) continue;
    responder.destroy();
    responders.delete(ip);
  }
  for (const item of active) {
    if (!responders.has(item.address)) responders.set(item.address, advertise(item));
  }
}

await stat(join(root, 'index.html'));
http.on('error', error => {
  console.error('HTTP listener', error.code ?? error.message);
  process.exit(1);
});
http.listen(port, host, () => {
  console.log(`CLARO-ARX http://${domain}:${port}/`);
  refreshNetwork();
});
const timer = setInterval(refreshNetwork, 15000);
process.once('SIGTERM', () => {
  clearInterval(timer);
  for (const responder of responders.values()) responder.destroy();
  http.close();
});
