"""Single current configuration for new SAM3/GraspGenX experiments."""
from arx_project.assets import load_object_catalog
from pathlib import Path
import hashlib
import json
import sys
from arx_project.layout import ROOT, module_dir

PROJECT = module_dir('pipeline')
WORKSPACE = PROJECT.parent
ARX = module_dir('arx_core')
CONFIG = ROOT/'configs/experiments/active_experiment.json'
sys.path.insert(0,str(ARX/'scripts/curobo_arx'))
from collision_profile import load_profile, require_current_profile


from arx_project.layout import read_json as read
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_current(verify_dataset=False):
    import yaml
    config=read(CONFIG);profile=load_profile()
    if config['collision_profile_id']!=profile['profile_id']:raise ValueError('Stale experiment collision profile')
    if config['tcp_eef_z_m']!=.043 or config['lift_joint_m']!=.452:raise ValueError('Stale TCP or lift configuration')
    if config['contact_offset_m']!=.001 or config['graspgenx_virtual_origin_back_m']!=.100:raise ValueError('Stale contact or GraspGenX geometry settings')
    from modules.curobo.fingertip_contact import load_profile as contact_profile
    contact = contact_profile()
    if config.get('table_contact_policy_id') != contact.profile_id:
        raise ValueError('Stale fingertip/table policy registration')
    if config['table_penetration_allowance_m'] != (contact.maximum_penetration_m if contact.enabled else 0.):
        raise ValueError('Stale local fingertip/table allowance')
    for name,digest in config['configuration_hashes'].items():
        if sha(name)!=digest:raise ValueError('Active configuration changed: '+name)
    for path in config['planning_models'].values():require_current_profile(yaml.safe_load(Path(path).read_text()))
    dataset=Path(config['dataset']).resolve();pointer=read(ROOT/'configs/experiments/scene_dataset.json')
    if Path(pointer['dataset']).resolve()!=dataset:raise ValueError('Current dataset pointer mismatch')
    if sha(dataset/'frozen_dataset.json')!=config['dataset_freeze_sha256']:raise ValueError('Dataset freeze changed')
    camera=read(ROOT/'simulation/cameras/camera_profiles.json')['profiles']['head']
    mount=read(ROOT/'simulation/cameras/head_camera_mount.json')
    if camera['model']!='Intel RealSense D405' or mount['downward_pitch_degrees']!=45.:raise ValueError('Stale head camera configuration')
    if verify_dataset:
        frozen=read(dataset/'frozen_dataset.json')
        entries=frozen.get('files',frozen.get('sha256',{}))
        if isinstance(entries,list):
            entries={r['path']:r['sha256'] for r in entries}
        if not entries:raise ValueError('No frozen scene files')
        for name,digest in entries.items():
            path=Path(name);path=path if path.is_absolute() else dataset/path
            if sha(path)!=digest:raise ValueError('Frozen scene file changed: '+str(path))
    return config


def runtime():
    """Use current catalog, calibrated down-looking head camera and 1mm contact."""
    config=load_current()
    sys.path[:0]=[str(PROJECT/'experiments/diverse_object_scenes'),str(WORKSPACE/'maniskill')]
    from pipeline import runtime as catalog_runtime
    result=catalog_runtime(load_object_catalog(Path(config['dataset'])/'object_catalog.json')['models'])
    environment=result[0]
    if environment.INITIAL_JOINTS['joint4']!=config['lift_joint_m']:raise ValueError('Runtime lift default differs')
    return result


if __name__=='__main__':
    cfg=load_current(verify_dataset=True)
    print(json.dumps({'passed':True,'profile':cfg['collision_profile_id'],'scenes':200,'tcp_m':.043,'lift_m':.452},ensure_ascii=False))
