import { existsSync, readFileSync } from 'node:fs';
import { dirname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const aliasFile = resolve(project, 'configs/path_aliases.json');
const aliases = existsSync(aliasFile)
  ? Object.entries(JSON.parse(readFileSync(aliasFile, 'utf8')).aliases).sort((a, b) => b[0].length - a[0].length)
  : [];

// Archived documents can retain their original links without restoring old folders.
// A standalone website uses its published copies and needs no project alias file.
export function sourcePath(file) {
  let current = resolve(file);
  const seen = new Set();
  while (!seen.has(current)) {
    seen.add(current);
    const alias = aliases.find(([from]) => current === from || current.startsWith(from + '/'));
    if (!alias) return current;
    const [from, to] = alias;
    const updated = (isAbsolute(to) ? to : resolve(project, to)) + current.slice(from.length);
    if (updated === current) return current;
    current = updated;
  }
  throw new Error('Cyclic archived source path');
}
