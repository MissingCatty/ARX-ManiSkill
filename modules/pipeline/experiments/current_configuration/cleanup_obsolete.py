"""User-authorized retirement of obsolete mainline configurations and products."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=ROOT.parent
SITE=WORKSPACE/'claro/进度网站'
RECORD=Path(__file__).resolve().parent/'records/cleanup_20261124_01'
CURRENT_DATASET=ROOT/'experiments/diverse_object_scenes/datasets/diverse200_geometry_20261122_01'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def candidates():
    paths=[]
    for experiment in ['repaired_grasp_benchmark','edge_distance_probe','full_failure_distance_probe',
        'reachability_distance_comparison','bimanual_workspace_comparison','offline_failure_diagnosis','fingertip_table_tolerance']:
        folder=ROOT/'experiments'/experiment
        if (folder/'runs').exists():paths.extend((folder/'runs').iterdir())
        paths += [folder/'active_run.json',folder/'implementation_status.json']
        paths += list(folder.glob('*.log'))
    paths += [ROOT/'experiments/diverse_grasp_benchmark/active_run.json']
    for name in ['workspace_development_20261122_01','diverse200_20261122_01','diverse200_workspace_20261122_01']:
        paths.append(ROOT/'experiments/diverse_object_scenes/datasets'/name)
    paths += [ROOT/'docs/diverse_grasp_benchmark',ROOT/'docs/mainline_versions/history',
        ROOT/'docs/mainline_versions/V1_当前链路与200场景结果.md',
        ROOT/'docs/mainline_versions/V2_距离问题修正方案.md',
        ROOT/'docs/mainline_versions/V2_夹指近桌接触余量实施结果.md']
    for name in ['sam3-v1','sam3-v2','edge-distance','full-failure-distance','reachability-distance','bimanual-workspace','升降高度']:
        paths.append(SITE/'内容/资源'/name)
    paths += [SITE/'内容/资源/夹爪TCP前后对比.png',SITE/'tools/publish_sam3_v1.py']
    fitting=WORKSPACE/'maniskill/outputs/arx_collision_spheres_20261119_01'
    paths += [*fitting.glob('*_paired.npz'),fitting/'paired_checks.json',fitting/'paired_checks.log',
        fitting/'endpoint_contact_pairs.json',fitting/'ARX碰撞球改前改后对比.png',fitting/'双视角视频']
    return sorted({p.absolute() for p in paths if p.exists() or p.is_symlink()},key=str)


def inventory(path):
    # Never walk symlinks into shared weights, assets or other projects.
    if path.is_symlink():return {'files':1,'bytes':path.lstat().st_size,'symlink':True}
    if path.is_file():return {'files':1,'bytes':path.stat().st_size,'sha256':sha(path)}
    count=size=0
    for base,dirs,files in os.walk(path,followlinks=False):
        for name in files:
            child=Path(base)/name;count+=1;size+=child.lstat().st_size
    return {'files':count,'bytes':size}


def main():
    if (RECORD/'completion.json').exists():raise SystemExit('Cleanup already completed; read its receipt.')
    from configuration import load_current
    load_current(verify_dataset=True)
    protected=[CURRENT_DATASET,ROOT/'weights',ROOT/'envs',ROOT/'third_party',
        ROOT/'experiments/pointcloud_benchmark/runs/five_methods_50_20261122_01',
        WORKSPACE/'maniskill/assets',WORKSPACE/'arx_core/asset',WORKSPACE/'research_dependencies',
        WORKSPACE/'reachability/产物/前方精细_20261121_01']
    paths=candidates();rows=[]
    for path in paths:
        if not any(path.is_relative_to(root) for root in [ROOT,SITE,WORKSPACE/'maniskill/outputs/arx_collision_spheres_20261119_01']):
            raise ValueError('Out-of-scope removal: '+str(path))
        for keep in protected:
            if path==keep or keep.is_relative_to(path) or path.is_relative_to(keep):raise ValueError('Protected path: '+str(path))
        rows.append({'path':str(path),'reason':'obsolete scenes, collision configuration, dependent grasp/diagnostic products or publication entry',**inventory(path)})
    receipt={'at':datetime.now().astimezone().isoformat(),
        'authorization':'用户要求场景、夹爪等统一采用最新配置，并删除旧配置与相应产物，以免污染后续实验。',
        'entries':rows,'protected_paths':list(map(str,protected)),
        'current_dataset_freeze_sha256':sha(CURRENT_DATASET/'frozen_dataset.json')}
    save(RECORD/'manifest.json',receipt)
    for row in rows:
        path=Path(row['path'])
        if path.is_symlink() or path.is_file():path.unlink()
        else:shutil.rmtree(path)
    assert all(not Path(r['path']).exists() and not Path(r['path']).is_symlink() for r in rows)
    load_current(verify_dataset=True)
    save(RECORD/'completion.json',{'completed':True,'at':datetime.now().astimezone().isoformat(),
        'paths_removed':len(rows),'files_removed':sum(r['files'] for r in rows),
        'bytes_removed':sum(r['bytes'] for r in rows),'manifest_sha256':sha(RECORD/'manifest.json'),
        'current_dataset_unchanged':True,'current_collision_profile':'arx_refined_871_v1','new_physical_grasps':0})
    print((RECORD/'completion.json').read_text())


if __name__=='__main__':main()
