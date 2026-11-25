import { existsSync, readFileSync } from 'node:fs';

export const FORMAT = 'arx_website_todos_v1';
export const schema = `CREATE TABLE todos (
  id INTEGER PRIMARY KEY AUTOINCREMENT, version TEXT NOT NULL, text TEXT NOT NULL,
  completed INTEGER NOT NULL DEFAULT 0 CHECK (completed IN (0, 1))
) STRICT;`;

export function validateSnapshot(value) {
  if (value?.format !== FORMAT || !Array.isArray(value.items)
    || !Number.isSafeInteger(value.sequence) || value.sequence < 0) {
    throw new Error('待办快照格式无效');
  }
  const ids = new Set();
  for (const item of value.items) {
    if (!item || !Number.isSafeInteger(item.id) || item.id < 1 || ids.has(item.id)
      || typeof item.version !== 'string' || !item.version.trim()
      || typeof item.text !== 'string' || !item.text.trim() || item.text.length > 250
      || /[\u0000-\u001f\u007f]/.test(item.text)
      || ![0, 1].includes(item.completed) || item.id > value.sequence) {
      throw new Error('待办快照包含无效或重复条目');
    }
    ids.add(item.id);
  }
  return {...value, items:value.items.map(({id, version, text, completed}) =>
    ({id, version, text, completed})).sort((a,b)=>a.id-b.id)};
}

export function captureSnapshot(db) {
  db.exec('BEGIN');
  try {
    const items = db.prepare('SELECT id, version, text, completed FROM todos ORDER BY id').all();
    const hasSequence = db.prepare("SELECT name FROM sqlite_schema WHERE name = 'sqlite_sequence'").get();
    const sequence = hasSequence ? db.prepare("SELECT seq FROM sqlite_sequence WHERE name = 'todos'").get()?.seq : 0;
    const value = validateSnapshot({format: FORMAT, exportedAt: new Date().toISOString(),
      sequence: items.reduce((maximum,item)=>Math.max(maximum,item.id),sequence ?? 0), items});
    db.exec('COMMIT');
    return value;
  } catch (error) {
    db.exec('ROLLBACK');
    throw error;
  }
}

function insertSnapshot(db, value) {
  value = validateSnapshot(value);
  const insert = db.prepare('INSERT INTO todos (id, version, text, completed) VALUES (?, ?, ?, ?)');
  for (const item of value.items) insert.run(item.id, item.version, item.text, item.completed);
  db.prepare("DELETE FROM sqlite_sequence WHERE name = 'todos'").run();
  db.prepare("INSERT INTO sqlite_sequence (name, seq) VALUES ('todos', ?)").run(value.sequence);
}

export function initializeTodos(db, snapshotFile) {
  db.exec('BEGIN IMMEDIATE');
  try {
    const table = db.prepare("SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = 'todos'").get();
    if (!table) {
      db.exec(schema);
      if (existsSync(snapshotFile)) insertSnapshot(db, JSON.parse(readFileSync(snapshotFile, 'utf8')));
    } else if (!table.sql.includes('AUTOINCREMENT')) {
      db.exec(`ALTER TABLE todos RENAME TO todos_before_delete;
        ${schema}
        INSERT INTO todos (id, version, text, completed)
          SELECT id, version, text, completed FROM todos_before_delete;
        DROP TABLE todos_before_delete;`);
    }
    db.exec('COMMIT');
  } catch (error) {
    db.exec('ROLLBACK');
    throw error;
  }
}

export function restoreSnapshot(db, value, { replace = false } = {}) {
  value = validateSnapshot(value);
  db.exec('BEGIN IMMEDIATE');
  try {
    const table = db.prepare("SELECT name FROM sqlite_schema WHERE type = 'table' AND name = 'todos'").get();
    if (table && !replace) throw new Error('已有待办数据库；迁移时请先备份并显式使用 --replace');
    db.exec(`DROP TABLE IF EXISTS todos; ${schema}`);
    insertSnapshot(db, value);
    db.exec('COMMIT');
  } catch (error) {
    db.exec('ROLLBACK');
    throw error;
  }
}
