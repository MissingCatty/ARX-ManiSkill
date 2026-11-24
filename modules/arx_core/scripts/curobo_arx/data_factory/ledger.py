"""SQLite WAL single-claim ledger; no expired-heartbeat execution takeover."""
import json
import copy
from pathlib import Path
import sqlite3
import time
import uuid
import numpy as np
from .common import canonical, digest, read, sha
from .geometry import grasp_equal, pose_equal


def visual_evidence_files(episode,quality):
    """Bind visual certificates to every actual movement, preserving legacy rules."""
    episode=Path(episode)
    if not (episode/'execution_roles.json').exists():
        return ('visual_receipt.json','pregrasp_visual_validation.json','approach_visual_validation.json','lift_visual_validation.json')
    if 'execution_roles.json' not in quality.get('files',{}):
        raise ValueError('Chronological role map missing from durable quality hashes')
    requests=read(episode/'requests.json');roles=read(episode/'execution_roles.json')['sequence']
    if [r['label'] for r in requests]!=[label for _,label in roles]:
        raise ValueError('Successful role map does not cover every dispatched action')
    for semantic,label in roles:
        if semantic!=(label if label in ('pregrasp','approach','close','lift','hold') else 'approach'):
            raise ValueError('Changed action semantics in admission')
    return ['visual_receipt.json']+[label+'_visual_validation.json' for semantic,label in roles if semantic not in ('close','hold')]


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS objects(id TEXT PRIMARY KEY, record TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS goals(object_id TEXT, id TEXT, record TEXT NOT NULL,
          PRIMARY KEY(object_id,id), FOREIGN KEY(object_id) REFERENCES objects(id));
        CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY, object_id TEXT NOT NULL,
          goal_id TEXT NOT NULL, distance INTEGER NOT NULL, sample INTEGER NOT NULL,
          ordinal INTEGER NOT NULL, owner TEXT NOT NULL, env TEXT NOT NULL,
          state TEXT NOT NULL, started REAL NOT NULL, physical INTEGER NOT NULL DEFAULT 0,
          outcome TEXT, episode TEXT, content_id TEXT UNIQUE, actual TEXT, receipt TEXT,
          UNIQUE(object_id,goal_id,distance,sample,ordinal));
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, attempt_id TEXT,
          event TEXT NOT NULL, value TEXT NOT NULL, unix REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS object_leases(object_id TEXT PRIMARY KEY,
          owner TEXT UNIQUE NOT NULL, acquired REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS exhausted_objects(object_id TEXT PRIMARY KEY,
          reason TEXT NOT NULL, evidence TEXT NOT NULL, unix REAL NOT NULL);
        ''')

    def close(self):
        self.db.close()

    def _placement(self, asset):
        from .placements import available_placement
        goals=[json.loads(r[0]) for r in self.db.execute('SELECT record FROM goals WHERE object_id=?',(asset['object_id'],))]
        rows=[dict(r) for r in self.db.execute('SELECT * FROM attempts WHERE object_id=?',(asset['object_id'],))]
        events=[{'event':r[0],'value':json.loads(r[1])} for r in self.db.execute(
            "SELECT event,value FROM events WHERE event IN ('object_acquired','placement_exhausted') ORDER BY seq")]
        return available_placement(asset,goals,rows,events,self.quotas)

    def _available_object(self, excluded=()):
        from collections import Counter
        q=self.quotas;rows=self.rows()
        if sum(r['physical'] for r in rows)>=q['max_physical_attempts'] or len(rows)>=q['max_total_dispatches']:return None
        busy={r[0] for r in self.db.execute('SELECT object_id FROM object_leases')}
        exhausted={r[0] for r in self.db.execute('SELECT object_id FROM exhausted_objects')}
        counts=Counter((r['object_id'],r['goal_id'],r['distance']) for r in rows if r['state']=='committed')
        dispatches=Counter(r['object_id'] for r in rows)
        options=[]
        for index,row in enumerate(self.db.execute('SELECT id,record FROM objects ORDER BY rowid')):
            name=row['id'];asset=json.loads(row['record'])
            if not asset.get('available',True) or not asset.get('collection_enabled',True) or name in busy|exhausted|set(excluded):continue
            goals=[r[0] for r in self.db.execute('SELECT id FROM goals WHERE object_id=?',(name,))]
            full=sum(all(counts[(name,g,d)]>=q['successes_per_bin'] for d in q['distances_mm']) for g in goals)
            if asset.get('collection_placements'):
                from .placements import active_asset
                placement=self._placement(asset)
                if placement is None:continue
                asset=active_asset(asset,placement)
            elif full>=q['goals_per_object']:continue
            prior=q.get('prior_dispatch_counts',{}).get(name,0)
            visits=0
            if asset.get('collection_placements'):
                visits=sum(json.loads(r[0]).get('object_id')==name for r in self.db.execute(
                    "SELECT value FROM events WHERE event='object_acquired'"))
            priority=(dispatches[name]+prior+visits,int(name in q.get('prior_candidate_screened_objects',[])))
            options.append((priority,0 if name==q.get('first_object') else 1,index,asset))
        return min(options,key=lambda x:x[:3])[3] if options else None

    def available_object(self,excluded=()):
        return self._available_object(excluded)

    def acquire_object(self,owner,object_id=None,renderer=None):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            held=self.db.execute('SELECT o.record FROM objects o JOIN object_leases l ON o.id=l.object_id WHERE l.owner=?',(owner,)).fetchone()
            if held:
                asset=json.loads(held[0])
                if asset.get('collection_placements'):
                    from .placements import active_asset
                    event=next((json.loads(r[0]) for r in self.db.execute(
                        "SELECT value FROM events WHERE event='object_acquired' ORDER BY seq DESC")
                        if json.loads(r[0]).get('owner')==owner),None)
                    if not event or not event.get('collection_placement'):
                        raise RuntimeError('Held object has no registered placement acquisition evidence')
                    asset=active_asset(asset,event['collection_placement'])
                if object_id is not None and asset['object_id']!=object_id or renderer is not None and asset['renderer']!=renderer:
                    raise RuntimeError('Object reservation differs from registered environment group')
                self.db.execute('COMMIT');return asset
            asset=self._available_object()
            if asset is None or object_id is not None and asset['object_id']!=object_id or renderer is not None and asset['renderer']!=renderer:
                self.db.execute('COMMIT');return None
            self.db.execute('INSERT INTO object_leases VALUES (?,?,?)',(asset['object_id'],owner,time.time()))
            self.event(None,'object_acquired',{'owner':owner,'object_id':asset['object_id'],
                'placement_id':asset.get('collection_placement',{}).get('placement_id'),
                'collection_placement':asset.get('collection_placement')})
            self.db.execute('COMMIT');return asset
        except BaseException:
            self.db.execute('ROLLBACK');raise

    def release_object(self,owner,*,exhausted_reason=None,evidence=None):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            lease=self.db.execute('SELECT object_id FROM object_leases WHERE owner=?',(owner,)).fetchone()
            if lease:
                if self.db.execute("SELECT 1 FROM attempts WHERE owner=? AND state IN ('claimed','executing','writing','interrupted')",(owner,)).fetchone():
                    raise RuntimeError('Cannot release object while an attempt is unresolved')
                if exhausted_reason:
                    record=json.loads(self.db.execute('SELECT record FROM objects WHERE id=?',(lease[0],)).fetchone()[0])
                    if record.get('collection_placements'):
                        acquisition=next(json.loads(r[0]) for r in self.db.execute(
                            "SELECT value FROM events WHERE event='object_acquired' ORDER BY seq DESC")
                            if json.loads(r[0]).get('owner')==owner)
                        self.event(None,'placement_exhausted',{'object_id':lease[0],
                            'placement_id':acquisition['placement_id'],'reason':exhausted_reason,'evidence':evidence})
                    else:
                        self.db.execute('INSERT OR REPLACE INTO exhausted_objects VALUES (?,?,?,?)',
                            (lease[0],exhausted_reason,canonical(evidence or {}).decode(),time.time()))
                self.db.execute('DELETE FROM object_leases WHERE owner=?',(owner,))
                self.event(None,'object_released',{'owner':owner,'object_id':lease[0],'reason':exhausted_reason})
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK');raise

    def initialize(self, catalog, quotas):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            old = self.db.execute("SELECT value FROM meta WHERE key='registration'").fetchone()
            identity = digest([catalog, quotas])
            if old:
                if old[0] != identity:
                    raise ValueError('Existing production ledger belongs to another registration')
            else:
                self.db.execute('INSERT INTO meta VALUES (?,?)', ('registration', identity))
                self.db.execute('INSERT INTO meta VALUES (?,?)', ('quotas', canonical(quotas).decode()))
                for obj in catalog['objects']:
                    self.db.execute('INSERT INTO objects VALUES (?,?)', (obj['object_id'], canonical(obj).decode()))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    @property
    def quotas(self):
        return json.loads(self.db.execute("SELECT value FROM meta WHERE key='quotas'").fetchone()[0])

    def event(self, attempt, event, value):
        self.db.execute('INSERT INTO events(attempt_id,event,value,unix) VALUES (?,?,?,?)',
                        (attempt, event, canonical(value).decode(), time.time()))

    def add_goals(self, object_id, pool):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            existing=[(r['id'],json.loads(r['record'])) for r in self.db.execute('SELECT id,record FROM goals WHERE object_id=? ORDER BY rowid',(object_id,))]
            asset=json.loads(self.db.execute('SELECT record FROM objects WHERE id=?',(object_id,)).fetchone()[0])
            registered={p['placement_id'] for p in asset.get('collection_placements',[])}
            resolved=[];rejected=[]
            for goal in pool['candidates']:
                if registered and goal.get('placement_id') not in registered:
                    raise ValueError('Goal has no registered collection placement')
                current=copy.deepcopy(goal)
                match=next(((ident,old) for ident,old in existing
                    if goal.get('placement_id')==old.get('placement_id') and grasp_equal(goal,old)),None)
                if match:
                    current.update(source_goal_id=goal['goal_id'],goal_id=match[0],canonical_grasp_world=match[1]['grasp_world'])
                else:
                    scoped=sum(old.get('placement_id')==goal.get('placement_id') for _,old in existing)
                    if scoped>=self.quotas.get('candidate_reserve',64):
                        rejected.append({'goal_id':goal['goal_id'],'reason':'global_distinct_goal_reserve_full'})
                        continue
                    self.db.execute('INSERT INTO goals VALUES (?,?,?)',(object_id,goal['goal_id'],canonical(goal).decode()))
                    existing.append((goal['goal_id'],goal))
                current['goal_equivalence']=('contact_center_5mm_10deg_parallel_jaw_v2' if 'pad_center_eef_m' in goal
                    else 'global_5mm_10deg_parallel_jaw_v1')
                if not any(g['goal_id']==current['goal_id'] for g in resolved):resolved.append(current)
            self.event(None, 'goal_pool', {'object_id': object_id, 'pool_sha256': digest(pool), 'candidates': len(pool['candidates']),
                'raw_network_candidates':pool.get('raw_candidates'),
                'fixed_G_planning_rejections':len(pool.get('planning_pool_rejections',[])),
                'resolved_actual_goals':resolved,'global_reserve_rejections':rejected})
            self.db.execute('COMMIT')
            return {**pool,'candidates':resolved,'global_reserve_rejections':rejected,
                'goal_equivalence':'Global 5mm/10degree/parallel-jaw classes; actual fresh G retained per attempt'}
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def claim(self, object_id, owner, env, active_pool=None):
        """Prioritize deficient direction/distance bins, and reserve unique samples."""
        q = self.quotas
        self.db.execute('BEGIN IMMEDIATE')
        try:
            record = self.db.execute('SELECT record FROM objects WHERE id=?',(object_id,)).fetchone()
            if record is None:
                raise ValueError('Unknown collection object: '+object_id)
            asset = json.loads(record[0])
            if not asset.get('available',True) or not asset.get('collection_enabled',True):
                raise ValueError('Collection disabled for object: '+object_id)
            if asset.get('collection_placements') and active_pool is None:
                raise ValueError('Diverse placements require the fresh qualified active pool')
            if asset.get('collection_placements') and active_pool:
                lease=self.db.execute('SELECT owner FROM object_leases WHERE object_id=?',(object_id,)).fetchone()
                if not lease or lease[0]!=owner:raise ValueError('Placement claim has no owned object lease')
                acquisition=next(json.loads(r[0]) for r in self.db.execute(
                    "SELECT value FROM events WHERE event='object_acquired' ORDER BY seq DESC")
                    if json.loads(r[0]).get('owner')==owner)
                if any(g.get('placement_id')!=acquisition['placement_id'] for g in active_pool):
                    raise ValueError('Fresh goal pool differs from the current object placement')
            total = self.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
            physical = self.db.execute('SELECT COALESCE(SUM(physical),0) FROM attempts').fetchone()[0]
            active = self.db.execute("SELECT COUNT(*) FROM attempts WHERE state IN ('claimed','executing','writing') AND physical=0").fetchone()[0]
            if total >= q['max_total_dispatches'] or physical+active >= q['max_physical_attempts']:
                self.db.execute('COMMIT')
                return None
            if self.db.execute("SELECT 1 FROM attempts WHERE env=? AND state IN ('claimed','executing','writing','interrupted')", (env,)).fetchone():
                raise RuntimeError('Environment is already claimed or has unresolved interrupted execution')
            goals = self.db.execute('SELECT id,record FROM goals WHERE object_id=? ORDER BY rowid', (object_id,)).fetchall()
            fresh={g['goal_id']:g for g in active_pool} if active_pool is not None else None
            if fresh is not None:goals=[g for g in goals if g['id'] in fresh]
            successes = self.db.execute("SELECT goal_id,distance,COUNT(*) n FROM attempts WHERE object_id=? AND state='committed' GROUP BY goal_id,distance", (object_id,)).fetchall()
            counts = {(r[0], r[1]): r[2] for r in successes}
            full = {g['id'] for g in goals if all(counts.get((g['id'], d), 0) >= q['successes_per_bin'] for d in q['distances_mm'])}
            if len(full) >= q['goals_per_object']:
                self.db.execute('COMMIT')
                return None
            options = []
            for gi, goal in enumerate(goals):
                for d in q['distances_mm']:
                    rows = self.db.execute('SELECT physical,sample,state FROM attempts WHERE object_id=? AND goal_id=? AND distance=?', (object_id, goal['id'], d)).fetchall()
                    n = counts.get((goal['id'], d), 0)
                    pending = sum(r['state'] in ('claimed','executing','writing') for r in rows)
                    physical_bin = sum(r['physical'] for r in rows)
                    rejects = sum(r['state']=='rejected' for r in rows)
                    if n+pending >= q['successes_per_bin'] or physical_bin+pending >= q['physical_per_bin'] or rejects >= q['planning_rejections_per_bin']:
                        continue
                    # A direction round-robin before raw network rank; sample zero is always scheduled.
                    priority=((len(rows)//2, gi, d) if q.get('sampling_policy')=='paired_nominal_perturbed_rounds_v1'
                        else (n+pending, len(rows), gi, d))
                    options.append((priority, goal, max([r['sample'] for r in rows], default=-1)+1))
            if not options:
                self.db.execute('COMMIT')
                return None
            # Work on up to the required number of live Gs. Reserves replace
            # exhausted goals; partial success across all 64 must not masquerade
            # as sixteen completed grasp quotas.
            eligible=[]
            for goal in goals:
                if goal['id'] not in full and any(item[1]['id']==goal['id'] for item in options):
                    eligible.append(goal['id'])
            active_goals=set(eligible[:max(0,q['goals_per_object']-len(full))])
            options=[item for item in options if item[1]['id'] in active_goals]
            (priority, goal, sample) = min(options, key=lambda item: item[0])
            d = priority[-1]
            ident = uuid.uuid4().hex
            self.db.execute('INSERT INTO attempts(id,object_id,goal_id,distance,sample,ordinal,owner,env,state,started) VALUES (?,?,?,?,?,0,?,?,?,?)',
                            (ident, object_id, goal['id'], d, sample, owner, env, 'claimed', time.time()))
            actual_goal=fresh[goal['id']] if fresh is not None else json.loads(goal['record'])
            self.event(ident, 'claimed', {'owner': owner, 'env': env,'actual_selected_goal':actual_goal})
            self.db.execute('COMMIT')
            return {**dict(self.db.execute('SELECT * FROM attempts WHERE id=?', (ident,)).fetchone()), 'goal':actual_goal}
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def transition(self, ident, owner, state, *, physical=False, outcome=None, episode=None):
        allowed = {'claimed': {'executing','rejected','fault','interrupted'},
                   'executing': {'writing','failed','fault','interrupted'},
                   'writing': {'failed','fault','interrupted','visual_rejected'}, 'interrupted': {'recovered_interrupted'}}
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT * FROM attempts WHERE id=?', (ident,)).fetchone()
            if not row or row['owner'] != owner or state not in allowed.get(row['state'], set()):
                raise RuntimeError('Illegal/stale attempt transition')
            self.db.execute('UPDATE attempts SET state=?, physical=MAX(physical,?),outcome=?,episode=COALESCE(?,episode) WHERE id=?',
                            (state, int(physical), outcome, str(episode) if episode else None, ident))
            self.event(ident, state, {'physical': physical, 'outcome': outcome, 'episode': str(episode) if episode else None})
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def commit(self, ident, owner, episode):
        """Only durable physical + data receipts can atomically add one success."""
        episode = Path(episode)
        physical, quality = read(episode/'physical_receipt.json'), read(episode/'quality_receipt.json')
        manifest = read(episode/'episode.json')
        if physical.get('pass') is not True or quality.get('pass') is not True or manifest['attempt_id'] != ident or not manifest['write_complete']:
            raise ValueError('Physical failure/incomplete write cannot increase successful demonstrations')
        if manifest.get('visual_geometry_required'):
            for name in visual_evidence_files(episode,quality):
                if name not in quality.get('files',{}) or not read(episode/name).get('pass'):
                    raise ValueError('Visual geometry evidence is missing or failed; cannot commit training data')
        from .common import verify_files
        verify_files({str(episode/p): s for p, s in quality['files'].items()})
        if quality['physical_receipt_sha256'] != sha(episode/'physical_receipt.json'):
            raise ValueError('Physical receipt changed after data acceptance')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT * FROM attempts WHERE id=?', (ident,)).fetchone()
            if not row or row['state'] != 'writing' or row['owner'] != owner:
                raise RuntimeError('Success must finish its sole live claim; cannot count twice')
            previous = self.db.execute("SELECT actual FROM attempts WHERE object_id=? AND goal_id=? AND state='committed'", (row['object_id'],row['goal_id'])).fetchall()
            actual = manifest['actual_entry']['snapshot']['link_world']['right_eef_link']
            duplicate = any(pose_equal(actual, json.loads(r[0])) for r in previous)
            duplicate_content = self.db.execute('SELECT 1 FROM attempts WHERE content_id=?', (quality['content_id'],)).fetchone()
            state = 'duplicate_excluded' if duplicate or duplicate_content else 'committed'
            self.db.execute('UPDATE attempts SET state=?,outcome=?,episode=?,content_id=?,actual=?,receipt=? WHERE id=?',
                (state, 'duplicate_actual_entry_or_recording' if state!='committed' else 'success', str(episode),
                 None if duplicate_content else quality['content_id'], canonical(actual).decode(),
                 canonical({'physical_sha256':sha(episode/'physical_receipt.json'), 'quality_sha256':sha(episode/'quality_receipt.json')}).decode(), ident))
            self.event(ident, state, {'physical_pass': True, 'quality_pass': True})
            self.db.execute('COMMIT')
            return state
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def recover(self, owner, proof):
        """Explicit post-mortem only; never reclaim by wall-clock heartbeat."""
        if proof.get('old_process_exited') is not True or proof.get('environments_idle') is not True or not proof.get('evidence'):
            raise ValueError('Recovery requires process-exit AND environment-idle evidence')
        rows = self.db.execute("SELECT id FROM attempts WHERE owner=? AND state IN ('claimed','executing','writing')", (owner,)).fetchall()
        for row in rows:
            self.transition(row[0], owner, 'interrupted', outcome='unknown_after_interruption')
        for row in self.db.execute("SELECT id FROM attempts WHERE owner=? AND state='interrupted'", (owner,)).fetchall():
            self.transition(row[0], owner, 'recovered_interrupted', outcome='original_attempt_retained_unknown')
        self.release_object(owner)
        self.event(None, 'recovery', proof)

    def rows(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM attempts ORDER BY started,id')]
