import { DatabaseSync } from 'node:sqlite';
import { closeSync, existsSync, fsyncSync, mkdirSync, openSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { randomUUID } from 'node:crypto';
import { captureSnapshot, restoreSnapshot, validateSnapshot } from '../todo-snapshot.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const { positionals, values } = parseArgs({allowPositionals: true, options: {
  db: { type: 'string', default: join(root, 'resources/state/待办.sqlite') },
  snapshot: { type: 'string', default: join(root, 'resources/state/todos.snapshot.json') },
  replace: { type: 'boolean', default: false },
}});
if (positionals.length !== 1 || !['export', 'restore'].includes(positionals[0])) {
  throw new Error('用法：node tools/todo_snapshot.mjs export|restore [--db 文件] [--snapshot 文件] [--replace]');
}
const file = resolve(values.db), snapshotFile = resolve(values.snapshot);
if (file === snapshotFile) throw new Error('数据库与JSON快照必须是不同文件');

function atomicJSON(path, value) {
  mkdirSync(dirname(path), {recursive: true});
  const pending = path+'.'+randomUUID()+'.pending';
  const fd = openSync(pending, 'wx', 0o600);
  try { writeFileSync(fd, JSON.stringify(value, null, 2)+'\n'); fsyncSync(fd); }
  finally { closeSync(fd); }
  renameSync(pending, path);
}

if (positionals[0] === 'export') {
  const db = new DatabaseSync(file, {readOnly: true});
  try {
    const snapshot = captureSnapshot(db);
    atomicJSON(snapshotFile, snapshot);
    console.log(JSON.stringify({action:'export', count:snapshot.items.length,
      completed:snapshot.items.filter(item=>item.completed).length,
      sequence:snapshot.sequence, snapshot:snapshotFile}));
  } finally { db.close(); }
} else {
  const snapshot = validateSnapshot(JSON.parse(readFileSync(snapshotFile, 'utf8')));
  mkdirSync(dirname(file), {recursive: true});
  const existed = existsSync(file);
  const db = new DatabaseSync(file);
  try {
    db.exec('PRAGMA busy_timeout = 5000');
    const table = db.prepare("SELECT name FROM sqlite_schema WHERE type = 'table' AND name = 'todos'").get();
    if (table && !values.replace) throw new Error('已有待办数据库；请用 --replace 进行有备份的迁移');
    let backup = null;
    if (existed && table) {
      backup = join(dirname(file), 'backups', 'todos-'+Date.now()+'-'+randomUUID()+'.sqlite');
      mkdirSync(dirname(backup), {recursive:true, mode:0o700});
      db.exec("VACUUM INTO '"+backup.replaceAll("'", "''")+"'");
    }
    restoreSnapshot(db, snapshot, {replace: values.replace});
    const restored = captureSnapshot(db);
    if (JSON.stringify(restored.items) !== JSON.stringify([...snapshot.items].sort((a,b)=>a.id-b.id))
      || restored.sequence !== snapshot.sequence) throw new Error('恢复后的待办与快照不一致');
    console.log(JSON.stringify({action:'restore', count:restored.items.length, sequence:restored.sequence,
      database:file, previous_database_backup:backup}));
  } finally { db.close(); }
}
