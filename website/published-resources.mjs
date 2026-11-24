import { copyFileSync, existsSync, mkdirSync, readFileSync, realpathSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { sourcePath } from './project-paths.mjs';

export const resources = fileURLToPath(new URL('./resources/', import.meta.url));
const receipts = new Map();
const hash = file => createHash('sha256').update(readFileSync(file)).digest('hex');

// Publication owns actual copies. No links back into experiment output folders.
export function snapshotFile(name, source) {
  source = sourcePath(source);
  const target = resolve(resources, name);
  if (!target.startsWith(resolve(resources) + '/') || name.split('/').includes('..')) throw new Error('Invalid published resource');
  if (existsSync(source) && realpathSync(source) !== target) {
    mkdirSync(dirname(target), {recursive:true});
    const linked=existsSync(target) && statSync(source).dev===statSync(target).dev && statSync(source).ino===statSync(target).ino;
    if (!existsSync(target) || linked || hash(source) !== hash(target)) {
      copyFileSync(source,target+'.pending'); renameSync(target+'.pending',target);
    }
  }
  if (!existsSync(target)) throw new Error('Missing published resource: '+name);
  if (realpathSync(target) !== target) throw new Error('Published resources require real files: '+name);
  receipts.set(name,{file:name,sha256:hash(target)});
  return target;
}

export function snapshotJson(name, source) {
  return JSON.parse(readFileSync(snapshotFile('metadata/'+name+'.json',source),'utf8'));
}

export function saveManifest() {
  const optimized=join(resources,'optimized/manifest.json');
  if (existsSync(optimized)) receipts.set('optimized/manifest.json',{file:'optimized/manifest.json',sha256:hash(optimized)});
  writeFileSync(join(resources,'manifest.json'),JSON.stringify({generatedAt:new Date().toISOString(),files:[...receipts.values()]},null,2)+'\n');
}
