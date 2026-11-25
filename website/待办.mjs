import { mkdirSync, readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import { fileURLToPath } from 'node:url';
import { initializeTodos } from './todo-snapshot.mjs';

const registry = new URL('./内容/网站.json', import.meta.url);
const storage = new URL('./resources/state/', import.meta.url);
let statements;

function openStore() {
  mkdirSync(storage, { recursive: true, mode: 0o700 });
  const db = new DatabaseSync(fileURLToPath(new URL('待办.sqlite', storage)));
  try {
    initializeTodos(db, new URL('todos.snapshot.json', storage));
  } catch (error) {
    db.close();
    throw error;
  }
  return {
    pending: db.prepare('SELECT * FROM todos WHERE completed = 0 ORDER BY id'),
    version: db.prepare('SELECT * FROM todos WHERE version = ? ORDER BY completed, id'),
    add: db.prepare('INSERT INTO todos (version, text) VALUES (?, ?) RETURNING *'),
    update: db.prepare('UPDATE todos SET completed = ? WHERE id = ? AND version = ? RETURNING *'),
    remove: db.prepare('DELETE FROM todos WHERE id = ? AND version = ?'),
  };
}

function json(res, status, payload) {
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  });
  res.end(JSON.stringify(payload));
}

async function readBody(req) {
  let size = 0;
  const chunks = [];
  for await (const chunk of req.iterator({ destroyOnReturn: false })) {
    size += chunk.length;
    if (size > 4096) {
      req.resume();
      return null;
    }
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    return null;
  }
}

export async function handleTodos(req, res, url) {
  if (req.method === 'GET' || req.method === 'HEAD') {
    const store = statements ??= openStore();
    const version = url.searchParams.get('version');
    return json(res, 200, version === null ? store.pending.all() : store.version.all(version));
  }
  if (req.method !== 'POST' && req.method !== 'PATCH' && req.method !== 'DELETE') {
    res.setHeader('Allow', 'GET, HEAD, POST, PATCH, DELETE');
    return json(res, 405, { error: '请求方法不受支持。' });
  }
  const sameHttpOrigin = req.headers.origin === `http://${req.headers.host}`;
  const sameHttpsOrigin = req.socket.remoteAddress === '127.0.0.1'
    && req.headers.origin === `https://${req.headers.host}`;
  if (!sameHttpOrigin && !sameHttpsOrigin) {
    return json(res, 403, { error: '请从项目网站提交待办。' });
  }
  if (req.headers['content-type']?.split(';')[0].trim() !== 'application/json') {
    return json(res, 415, { error: '提交格式不正确。' });
  }
  const body = await readBody(req);
  const configuration = JSON.parse(readFileSync(registry, 'utf8'));
  const versions = [...configuration.versions,
    ...(configuration.configurationRecords ?? []).filter(record => record.todos)];
  if (!body || !versions.some(version => version.id === body.version)) {
    return json(res, 400, { error: '无法识别这个版本。' });
  }
  const store = statements ??= openStore();
  if (req.method === 'POST') {
    const text = typeof body.text === 'string' ? body.text.trim() : '';
    if (!text || text.length > 250 || /[\u0000-\u001f\u007f]/.test(text)) {
      return json(res, 400, { error: '请输入 1 至 250 字的待办内容。' });
    }
    return json(res, 201, store.add.get(body.version, text));
  }
  if (!Number.isSafeInteger(body.id) || body.id < 1) {
    return json(res, 400, { error: '待办编号无效。' });
  }
  if (req.method === 'DELETE') {
    store.remove.run(body.id, body.version);
    return json(res, 200, { id: body.id });
  }
  if (typeof body.completed !== 'boolean') {
    return json(res, 400, { error: '待办状态无效。' });
  }
  const updated = store.update.get(Number(body.completed), body.id, body.version);
  return updated ? json(res, 200, updated) : json(res, 404, { error: '这条待办不存在。' });
}
