"""Register a new, bounded SDG production campaign and its official candidate bank."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT,read,sha,write,canonical,digest
from scripts.curobo_arx.data_factory.prepare import prepare as base_prepare
from .sampler import SDGSampler,frozen_pad_center


IMAGE='localhost/arx-s02-isaac:6.0.1'
SDK_PATH='/isaac-sim/exts/isaacsim.replicator.grasping'


def copy_installed_sdk(destination):
    name='arx-sdg-sdk-'+uuid.uuid4().hex[:10]
    subprocess.run(['podman','create','--name',name,'--entrypoint','/bin/true',IMAGE],check=True,capture_output=True)
    try:subprocess.run(['podman','cp',name+':'+SDK_PATH,str(destination)],check=True,capture_output=True)
    finally:subprocess.run(['podman','rm',name],check=True,capture_output=True)
    import tomllib
    version=tomllib.loads((destination/'config/extension.toml').read_text())['package']['version']
    source=destination/'isaacsim/replicator/grasping/sampler_utils.py'
    return version,source


def prepare(out,*,extra=None,wall_hours=24,envs=2,max_attempts=2048,block=8,
            reserve=128,goals=32,successes=8,candidates=4096,orientations=24,first_object='raisin_box',candidate_bank_from=None,
            source_fix_parent=None,object_policy='common_household',placement_profile='active',only_objects=None):
    out=Path(out).resolve()
    continuation=None
    if source_fix_parent:
        if only_objects is not None:raise ValueError('Changed validation object subset requires a new collection')
        import sqlite3
        from scripts.curobo_arx.data_factory.resume import exit_proof
        parent=Path(source_fix_parent).resolve();old=read(parent/'registration.json')
        if old.get('stage')!='sdg_stage1':raise ValueError('Source fix requires an SDG parent')
        if old.get('object_policy','all')!=object_policy:
            raise ValueError('Changed object selection requires a new collection, not a source-fix continuation')
        if placement_profile!='historical':
            raise ValueError('A placement-diversity change requires a new collection; source fixes must explicitly retain historical placements')
        pid=read(parent/'coordinator_process.json')['pid'];proc=Path('/proc')/str(pid)/'cmdline'
        if proc.exists() and str(parent).encode() in proc.read_bytes():raise ValueError('SDG parent coordinator still exists')
        proof=exit_proof(parent,reconcile_late_shutdown=True)
        deadline=old.get('deadline_unix',read(parent/'launch.json')['stop_unix'])
        if time.time()>=deadline:raise ValueError('Original production deadline expired')
        db=sqlite3.connect('file:'+str(parent/'ledger.sqlite3')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        try:
            rows=[dict(r) for r in db.execute('SELECT * FROM attempts')]
            screened=set(old['quotas'].get('prior_candidate_screened_objects',[]))
            screened.update(json.loads(value)['object_id'] for (value,) in db.execute("SELECT value FROM events WHERE event='object_acquired'"))
        finally:db.close()
        if any(r['state'] in ('claimed','executing','writing','interrupted') for r in rows):
            raise ValueError('Parent has unresolved execution records')
        prior=old.get('source_fix',{})
        total=prior.get('initial_physical_budget',old['quotas']['max_physical_attempts'])
        used=prior.get('consumed_prior_physical_attempts',0)+sum(r['physical'] for r in rows)
        if total-used<1:raise ValueError('Original physical budget exhausted')
        visits=Counter(old['quotas'].get('prior_dispatch_counts',{}));visits.update(r['object_id'] for r in rows)
        continuation={'parent':str(parent),'old':old,'proof':proof,'deadline':deadline,'visits':dict(visits),'screened':sorted(screened),
            'initial_physical_budget':total,'consumed_prior_physical_attempts':used,
            'parent_physical_successes':sum(r['state']=='committed' for r in rows),
            'parent_ledger_sha256':sha(parent/'ledger.sqlite3')}
        max_attempts=total-used;wall_hours=old['wall_hours']
        candidate_bank_from=parent
    reg=base_prepare(out,extra=extra,envs=envs,wall_hours=wall_hours,block=block,
        max_attempts=max_attempts,first_object=first_object,candidate_backend='sdg',
        object_policy=object_policy,
        placement_profile=placement_profile,
        only_objects=only_objects,
        sampling_policy='paired_nominal_perturbed_rounds_v1',
        quota_overrides={'candidate_reserve':reserve,'goals_per_object':goals,'successes_per_bin':successes})
    reg.update(production_ready=False,stage='sdg_stage1',entry_sampler='per_object_G_seed_v1',
        scene_transport='arx_geometry_files_live_pose_v1',reach_avoid_target=True,
        plan='research_plan/docs/sdg_stage1_main_steps_20261031.md',
        data_source='Full ARX physical execution, candidates from official NVIDIA SDG; old166 not counted as newly generated',
        deliverables=['LiLo-native RGB-D/state/action episodes','D_requested and D_actual offline entry index',
            'per-object, surface/direction/G/distance/perturbation coverage and all failed attempts'])
    from scripts.curobo_arx.data_factory.visual_geometry import POLICY
    reg['visual_geometry_policy']=POLICY
    write(out/'registration.json',reg)
    write(out/'state.json',{'status':'preparing_official_sdg_bank','successes':0,'physical_attempts':0})
    try:
        from .asset_contract import source_contract
        for asset in reg['catalog']['objects'][reg['catalog']['initial_count']:]:
            if asset.get('available') and source_contract(asset['usd_path'])!=asset.get('physics_contract'):
                raise ValueError('Additional asset source contract is missing or differs from current USD: '+asset['object_id'])
        version,sampler=copy_installed_sdk(out/'official_sdg_source')
        urdf=ET.parse(reg['model_reference']+'/model/arx.urdf').getroot()
        aperture=sum(float(j.find('limit').get('upper')) for j in urdf.findall('joint')
            if j.get('name') in ('right_joint27','right_joint28'))
        params={'num_candidates':candidates,'num_orientations':orientations,
            'gripper_maximum_aperture':aperture,'gripper_standoff_fingertips':0.,
            'gripper_approach_direction':[0,0,1],'grasp_align_axis':[0,1,0],
            'orientation_sample_axis':[0,1,0],'lateral_sigma':0.,'verbose':False}
        reg['sdg']={'extension_version':version,'image':IMAGE,'sampler_file':str(sampler),
            'sampler_sha256':sha(sampler),'sampler_parameters':params,
            'pad_center_eef_m':frozen_pad_center().tolist(),
            'contact_symmetry':'contact_center_5mm_10deg_parallel_jaw_v2',
            'tool_convention':'SDK contact frame at surface-pair midpoint; G_tcp = T_contact @ Translate(-PAD_CENTER_EEF), closing+Y/approach+Z',
            'aperture_scope':'Conservative candidate bound from the original two .044m prismatic travels; actual full frozen collision/contact checks remain',
            'sampler_execution':'CPU, exact installed unmodified official Python source',
            'SDK_evaluate_grasp_poses_used':False,'SDK_gripper_teleport_used':False,'gravity_changed':False}
        for f in (out/'official_sdg_source').rglob('*'):
            if f.is_file():reg['dependencies'][str(f)]=sha(f)
        backend=SDGSampler(reg);bank=out/'candidate_bank';bank.mkdir()
        prior=None
        if candidate_bank_from:
            parent=Path(candidate_bank_from).resolve();prior=read(parent/'registration.json')
            if prior['sdg']['sampler_sha256']!=reg['sdg']['sampler_sha256'] or prior['sdg']['sampler_parameters']!=params or prior['seed']!=reg['seed']:
                raise ValueError('Candidate-bank source/configuration differs')
            reg['sdg']['bank_reuse']={'campaign':str(parent),'registration_sha256':sha(parent/'registration.json'),
                'scope':'Immutable geometric candidates only; no old trajectories counted as new'}
        receipts=[]
        for i,asset in enumerate(reg['catalog']['objects']):
            if not asset.get('collection_enabled',True):
                receipts.append({'object_id':asset['object_id'],'available':asset['available'],
                    'collection_enabled':False,'reason':asset['collection_exclusion_reason']});continue
            if not asset['available']:
                receipts.append({'object_id':asset['object_id'],'available':False,'issues':asset['issues']});continue
            started=time.monotonic()
            path=bank/(asset['object_id']+'.npy')
            if prior is not None:
                old_asset=next(a for a in prior['catalog']['objects'] if a['object_id']==asset['object_id'])
                if old_asset['asset_sha256']!=asset['asset_sha256'] or old_asset['geometry_group']!=asset['geometry_group']:
                    raise ValueError('Cached candidate asset differs: '+asset['object_id'])
                old_bank=Path(prior['sdg']['candidate_bank']);evidence=read(old_bank/(asset['object_id']+'.json'))
                if sha(old_bank/(asset['object_id']+'.npy'))!=evidence['poses_sha256']:
                    raise ValueError('Cached candidate bank was modified')
                shutil.copy2(old_bank/(asset['object_id']+'.npy'),path)
                samples=np.load(path,allow_pickle=False)
            else:
                samples,evidence=backend.local_candidates(asset)
                np.save(path,np.asarray(samples,dtype=float).reshape(-1,4,4))
            evidence.update(asset_sha256=asset['asset_sha256'],poses_sha256=sha(path),
                object_id=asset['object_id'],raw_candidates=len(samples),wall_seconds=time.monotonic()-started)
            receipt=bank/(asset['object_id']+'.json');write(receipt,evidence)
            reg['dependencies'].update({str(path):sha(path),str(receipt):sha(receipt)})
            receipts.append({'object_id':asset['object_id'],'available':True,'raw_candidates':len(samples),
                'parts':len(evidence['mesh_parts']),'seconds':evidence['wall_seconds']})
            print('SDG_BANK',i+1,len(reg['catalog']['objects']),asset['object_id'],len(samples),flush=True)
        write(bank/'manifest.json',{'kind':'Geometric candidate poses only; zero demonstrations or physical evaluations',
            'objects':receipts,'extension_version':version,'sampler_sha256':sha(sampler)})
        reg['dependencies'][str(bank/'manifest.json')]=sha(bank/'manifest.json')
        reg['sdg']['candidate_bank']=str(bank)
        if continuation:
            c=continuation;old=c['old'];parent=Path(c['parent'])
            reg['deadline_unix']=c['deadline']
            reg['campaign_started_unix']=old.get('campaign_started_unix',read(parent/'launch.json')['started_unix'])
            reg['wall_budget_scope']='Original SDG deadline retained across source fixes'
            reg['lineage']=[*old.get('lineage',[]),{'campaign':str(parent),
                'registration_sha256':sha(parent/'registration.json'),'state_sha256':sha(parent/'state.json'),
                'status':read(parent/'state.json')['status']}]
            reg['source_fix']={k:c[k] for k in ('parent','initial_physical_budget','consumed_prior_physical_attempts',
                'parent_physical_successes','parent_ledger_sha256')}
            reg['source_fix']['data_scope']='Clean corrected-source ledger; prior source results preserved separately, not copied or counted as new production data'
            reg['quotas']['prior_dispatch_counts']=c['visits']
            reg['quotas']['prior_candidate_screened_objects']=c['screened']
            from scripts.curobo_arx.data_factory.ledger import Ledger
            ledger=Ledger(out/'ledger.sqlite3')
            try:
                if ledger.rows():raise ValueError('Source fix must initialize an unexecuted ledger')
                ledger.db.execute('BEGIN IMMEDIATE')
                ledger.db.execute("UPDATE meta SET value=? WHERE key='quotas'",(canonical(reg['quotas']).decode(),))
                ledger.db.execute("UPDATE meta SET value=? WHERE key='registration'",(digest([reg['catalog'],reg['quotas']]),))
                ledger.db.execute('COMMIT')
            finally:ledger.close()
            write(out/'exit_proof.json',c['proof']);write(out/'source_fix_receipt.json',reg['source_fix'])
            if sha(parent/'ledger.sqlite3')!=c['parent_ledger_sha256']:raise ValueError('Parent ledger changed during source-fix preparation')
        reg['production_ready']=True
        write(out/'registration.json',reg)
        write(out/'state.json',{'status':'prepared','successes':0,'new_successes':0,'physical_attempts':0,'quota_complete':False})
        return reg
    except BaseException:
        write(out/'PREPARATION_FAULT.json',{'error':traceback.format_exc(),'wall':time.time(),'physical_attempts':0})
        write(out/'state.json',{'status':'preparation_fault','physical_attempts':0,'successes':0})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--extra-catalog')
    p.add_argument('--wall-hours',type=float,default=24);p.add_argument('--envs-per-gpu',type=int,default=2)
    p.add_argument('--max-attempts',type=int,default=2048);p.add_argument('--object-block-attempts',type=int,default=8)
    p.add_argument('--reserve',type=int,default=128);p.add_argument('--goals-per-object',type=int,default=32)
    p.add_argument('--successes-per-bin',type=int,default=8);p.add_argument('--candidates',type=int,default=4096)
    p.add_argument('--orientations',type=int,default=24);p.add_argument('--first-object',default='raisin_box')
    p.add_argument('--candidate-bank-from',help='Reuse matching hash-verified local geometric candidates only')
    p.add_argument('--source-fix-parent',help='Require exited parent; clean source version, original deadline/remaining budget and prior object visits')
    p.add_argument('--object-policy',choices=('common_household','all'),default='common_household')
    p.add_argument('--placement-profile',default='active',help='Active collection profile, explicit profile path, or historical (no placement changes)')
    p.add_argument('--only-object',action='append',help='Explicit bounded validation subset; all other physical scene objects are retained')
    a=p.parse_args();result=prepare(a.out,extra=a.extra_catalog,wall_hours=a.wall_hours,envs=a.envs_per_gpu,
        max_attempts=a.max_attempts,block=a.object_block_attempts,reserve=a.reserve,goals=a.goals_per_object,
        successes=a.successes_per_bin,candidates=a.candidates,orientations=a.orientations,first_object=a.first_object,
        candidate_bank_from=a.candidate_bank_from,source_fix_parent=a.source_fix_parent,object_policy=a.object_policy,
        placement_profile=a.placement_profile,only_objects=a.only_object)
    print('PREPARED',a.out,len(result['catalog']['objects']))
