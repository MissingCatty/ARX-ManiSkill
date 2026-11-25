import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';
import { test } from 'node:test';
import { captureSnapshot, initializeTodos, restoreSnapshot, schema } from '../todo-snapshot.mjs';

const snapshot = {format:'arx_website_todos_v1', sequence:9, items:[
  {id:2, version:'claro', text:'检查挡物候选', completed:1},
  {id:7, version:'gripper-configuration', text:'复核夹爪扫掠区域', completed:0},
]};

test('restoration preserves IDs, completion, gaps and deleted highest ID', () => {
  const db = new DatabaseSync(':memory:');
  try {
    restoreSnapshot(db, snapshot);
    assert.equal(captureSnapshot(db).sequence, 9);
    assert.equal(JSON.stringify(captureSnapshot(db).items), JSON.stringify(snapshot.items));
    assert.equal(db.prepare("INSERT INTO todos(version,text) VALUES('claro','下一项') RETURNING id").get().id,10);
  } finally { db.close(); }
});

test('fresh bootstrap loads the snapshot and does not resurrect deleted tasks on restart', () => {
  const folder=mkdtempSync(join(tmpdir(),'arx-todos-')), file=join(folder,'todos.snapshot.json');
  const db=new DatabaseSync(':memory:');
  try {
    writeFileSync(file,JSON.stringify(snapshot));
    initializeTodos(db,file);
    assert.equal(db.prepare('SELECT count(*) AS n FROM todos').get().n,2);
    db.exec('DELETE FROM todos');
    initializeTodos(db,file);
    assert.equal(db.prepare('SELECT count(*) AS n FROM todos').get().n,0);
  } finally { db.close();rmSync(folder,{recursive:true,force:true}); }
});

test('invalid import leaves existing data intact and explicit replace is required', () => {
  const db=new DatabaseSync(':memory:');
  try {
    restoreSnapshot(db,snapshot);
    assert.throws(()=>restoreSnapshot(db,{...snapshot,items:[]}),/已有待办/);
    assert.throws(()=>restoreSnapshot(db,{...snapshot,items:[snapshot.items[0],snapshot.items[0]]},{replace:true}),/重复/);
    assert.equal(JSON.stringify(captureSnapshot(db).items),JSON.stringify(snapshot.items));
  } finally { db.close(); }
});

test('legacy databases migrate without losing existing todos', () => {
  const db=new DatabaseSync(':memory:');
  try {
    db.exec("CREATE TABLE todos(id INTEGER PRIMARY KEY,version TEXT,text TEXT,completed INTEGER); INSERT INTO todos VALUES(4,'claro','已有待办',1)");
    initializeTodos(db,'/no/snapshot/file');
    assert.equal(db.prepare('SELECT completed FROM todos WHERE id=4').get().completed,1);
    assert.equal(db.prepare("INSERT INTO todos(version,text) VALUES('claro','下一项') RETURNING id").get().id,5);
  } finally { db.close(); }
});

test('CLI export and replacement create a recoverable backup of the previous remote store', () => {
  const folder=mkdtempSync(join(tmpdir(),'arx-todos-cli-'));
  const dbFile=join(folder,'todos.sqlite'), file=join(folder,'todos.snapshot.json');
  const script=fileURLToPath(new URL('./todo_snapshot.mjs',import.meta.url));
  function run(...args) {
    const result=spawnSync(process.execPath,[script,...args,'--db',dbFile,'--snapshot',file],{encoding:'utf8'});
    assert.equal(result.status,0,result.stderr);
    return JSON.parse(result.stdout);
  }
  try {
    let db=new DatabaseSync(dbFile);db.exec(schema);db.exec("INSERT INTO todos(version,text) VALUES('claro','remote old task')");db.close();
    const exported=run('export');assert.equal(exported.count,1);
    writeFileSync(file,JSON.stringify(snapshot));
    const restored=run('restore','--replace');assert.equal(restored.count,2);
    db=new DatabaseSync(restored.previous_database_backup,{readOnly:true});
    assert.equal(db.prepare('SELECT text FROM todos').get().text,'remote old task');db.close();
    db=new DatabaseSync(dbFile,{readOnly:true});
    assert.equal(JSON.stringify(captureSnapshot(db).items),JSON.stringify(snapshot.items));db.close();
    assert.equal(JSON.parse(readFileSync(file,'utf8')).sequence,9);
  } finally {rmSync(folder,{recursive:true,force:true});}
});
