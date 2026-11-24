"""Register a bounded Graspness -> pregrasp -> local ARX teacher run."""
import argparse
import copy
from pathlib import Path
import subprocess
from scripts.curobo_arx.data_factory.common import ROOT,read,write,sha,space
from scripts.curobo_arx.data_factory.prepare import prepare as base_prepare
from scripts.curobo_arx.data_factory.visual_geometry import POLICY


def prepare(out,config,*,objects=None,max_attempts=12,wall_hours=2):
    config=Path(config).resolve();cfg=copy.deepcopy(read(config));source=Path(cfg['source'])
    space(Path(out).resolve().parent)
    for path in (Path(cfg['python']),Path(cfg['checkpoint']),source/'models/graspnet.py'):
        if not path.is_file():raise FileNotFoundError('Graspness prerequisite is missing: '+str(path))
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=cfg['revision']:raise ValueError('Graspness source checkout differs from pinned revision')
    receipt=read(ROOT/cfg['checkpoint_receipt'])
    checkpoint_sha=sha(cfg['checkpoint'])
    if checkpoint_sha!=receipt['sha256']:raise ValueError('Graspness pretrained weight receipt differs')
    cfg.update(checkpoint_sha256=checkpoint_sha,weight_provenance=receipt)
    placement_profile=Path(cfg.get('placement_profile','configs/lilo_vla/graspness_placement_diversity.json'))
    if not placement_profile.is_absolute():placement_profile=ROOT/placement_profile
    reg=base_prepare(out,candidate_backend='graspness',max_attempts=max_attempts,envs=2,wall_hours=wall_hours,
        block=2,only_objects=objects,object_policy='common_household',first_object=cfg.get('first_object'),
        placement_profile=str(placement_profile),
        distances_mm=[cfg['pregrasp_distance_mm']],quota_overrides={'candidate_reserve':8,'goals_per_object':2,'successes_per_bin':1})
    out=Path(out).resolve()
    reg.update(stage='graspness_teacher',production_ready=False,graspness=cfg,
        entry_sampler='per_object_G_seed_v1',qualified_pregrasp=True,
        qualification_limit=cfg['qualified_goal_limit'],visual_geometry_policy=POLICY,
        scene_transport='arx_geometry_files_live_pose_v1',reach_avoid_target=True,
        render_camera_clipping=cfg.get('render_camera_clipping',{}),
        plan='scripts/curobo_arx/graspness_teacher/README.md',
        data_source='Real Graspness neural proposals and original ARX physical execution; no AnyGrasp/SDG substitution',
        workflow='Actual head depth -> Graspness G -> H=G translated -100mm along approach -> actual cuRobo reach -> wrist recording at actual H -> approach/close/lift/hold')
    write(out/'registration.json',reg)
    reg['collection_mode']=cfg.get('collection_mode','bounded_validation')
    if cfg.get('bulk_storage_root'):
        from scripts.curobo_arx.data_factory.storage import register_bulk_storage
        reg['bulk_storage']=register_bulk_storage(out,cfg['bulk_storage_root'])
    dependencies=reg['dependencies'];dependencies[str(config)]=sha(config)
    dependencies[str(ROOT/cfg['checkpoint_receipt'])]=sha(ROOT/cfg['checkpoint_receipt'])
    dependencies[cfg['checkpoint']]=checkpoint_sha
    for folder in (source,ROOT.parent/'research_dependencies/anygrasp_MinkowskiEngine'):
        for path in folder.rglob('*'):
            if path.is_file() and '.git' not in path.parts and '__pycache__' not in path.parts and path.suffix in ('.py','.so','.h','.hpp','.cuh','.cu','.cpp'):
                if 'build' in path.parts and path.suffix!='.so':continue
                dependencies[str(path)]=sha(path)
    reg['production_ready']=True
    write(out/'registration.json',reg)
    write(out/'method_receipt.json',{'algorithm':'Graspness / GSNet','weight_provenance':receipt,
        'robot_camera_calibration_physics_and_success_criteria_changed':False,
        'renderer_only_correction':cfg.get('render_camera_clipping',{}),
        'recording_begins':'Actual verified pregrasp H, before local approach command',
        'neural_proposals_are_not_demonstrations':True,'physical_attempt_limit':max_attempts})
    return reg


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--config',default=str(ROOT/'configs/lilo_vla/graspness_teacher.json'))
    parser.add_argument('--only-object',action='append');parser.add_argument('--max-attempts',type=int,default=12)
    parser.add_argument('--wall-hours',type=float,default=2)
    args=parser.parse_args();reg=prepare(args.out,args.config,objects=args.only_object,max_attempts=args.max_attempts,wall_hours=args.wall_hours)
    print('PREPARED_GRASPNESS',args.out,reg['bounded_validation_objects'])
