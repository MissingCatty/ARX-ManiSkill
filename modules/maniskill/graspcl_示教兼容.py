"""Explicit demonstration TCP metadata for new runs, without editing frozen sources."""
from copy import deepcopy
import ast
import inspect
from pathlib import Path
import textwrap

import numpy as np

TCP=.043
INSTALLED=False


def prepare_recall(candidate):
    meta=candidate['metadata']
    source=float(meta.get('contact_storage_tcp_m',.040))
    if source not in (.040,.043):raise ValueError('Unknown demonstration contact-frame definition')
    shift=TCP-source;pose=np.array(candidate['pose'],dtype=float,copy=True)
    delta=shift*pose[:3,2];pose[:3,3]+=delta;candidate['pose']=pose
    for key in ['point1','point2']:
        if key in candidate:candidate[key]=np.array(candidate[key])+delta
    meta.update(source_contact_storage_tcp_m=source,contact_storage_tcp_m=TCP,
        storage_tcp_conversion_m=shift,demonstration_tcp40_to_tcp43_translation_m=shift)


def install():
    global INSTALLED
    if INSTALLED:return
    import table_truncation_model as model
    original=model.make_pool
    def factory(base):
        cls=original(base)
        # The physical-table adapter compiles its methods dynamically, so their
        # inspect source locations are synthetic. Start from the frozen AST and
        # apply the same registered physical-table metadata transformation.
        parsed=ast.parse(Path(model.__file__).read_text())
        make=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='make_pool')
        definition=next(n for n in make.body if isinstance(n,ast.ClassDef) and n.name=='TruncationPool')
        method=next(n for n in definition.body if isinstance(n,ast.FunctionDef) and n.name=='refine')
        import sys
        sys.path.insert(0,str(Path(model.__file__).parent/'tmp'))
        from physical_table_pair_20261118_01 import transform_tree
        physical=transform_tree(ast.unparse(method),'pool')
        class Converter(ast.NodeTransformer):
            changes=0
            def visit_If(self,node):
                if ast.unparse(node.test)=="candidate['metadata']['origin'].startswith('demonstration_')":
                    assert len(node.body)==4
                    assert ast.unparse(node.body[-1]).endswith('= 0.003')
                    node.body=ast.parse('compat_prepare_recall(candidate)').body;self.changes+=1
                    return node
                return self.generic_visit(node)
        converter=Converter();tree=ast.fix_missing_locations(converter.visit(physical))
        assert converter.changes==1,'Frozen recall conversion no longer matches'
        scope={**cls.refine.__globals__,'compat_prepare_recall':prepare_recall}
        exec(compile(tree,__file__+':refine','exec'),scope)
        cls.refine=scope['refine']
        return cls
    model.make_pool=factory;INSTALLED=True


def validate_trial(directory,reg):
    import graspcl_pipeline as common
    directory=Path(directory);path=directory/'graspcl/candidate_pool.json'
    if not path.exists():return {'passed':True,'grasp_inference_performed':False}
    pool=common.read(path);before=common.read(directory/'graspcl/unrefined_candidate_pool.json')
    refinement=common.read(directory/'truncation.json')
    assert pool['refinement']==refinement
    assert refinement['mode']=='fixed_orientation' and refinement['table_closed_margin_m']==0.
    assert refinement['nominal_closed_precheck'] is False
    assert refinement['retained']==len(pool['candidates'])
    assert len(pool['candidates'])+len(refinement['rejected'])==len(before['candidates'])
    recall_count=0
    for c in pool['candidates']:
        m=c['metadata'];src=before['candidates'][m['source_pool_index']];a=np.asarray(src['pose']);b=np.asarray(c['pose'])
        recall=m['origin'].startswith('demonstration_');stored=float(src['metadata'].get('contact_storage_tcp_m',.040))
        shift=TCP-stored if recall else 0.;retraction=m['insertion_retraction_m']
        assert np.allclose(b[:3,:3],a[:3,:3],atol=1e-9,rtol=0)
        assert np.allclose(b[:3,3],a[:3,3]+(shift-retraction)*a[:3,2],atol=1e-9,rtol=0)
        assert 0<=retraction<=src['metadata']['source_network_depth_m']+1e-9
        assert m['gripper_table_clearance_after_m']>=0 and m['observed_target_points_in_closing_sweep']>0
        if recall:
            assert stored==TCP,'The new bank must author TCP43 directly'
            assert m['storage_tcp_conversion_m']==0.,'TCP43 demonstration received duplicate compensation'
            recall_count+=1
    scored=reg['pipeline_profile']=='graspcl_200'
    if scored:
        assert pool['refined_candidates_rescored'] is True
        rescore=common.read(directory/'refined_rescore.json');assert rescore['success'] and rescore['bank_updates']==0
        memory=common.read(directory/'frozen_memory_rescore.json');assert memory['before_sha256']==memory['after_sha256']
    receipt={'passed':True,'contact_storage_tcp_m':TCP,'execution_tcp_m':TCP,
        'duplicate_compensation':False,'retained_recall_candidates':recall_count,'refined_candidates_rescored':scored,
        'candidate_pool_sha256':common.sha(path),'test_feedback_added':0}
    common.save(directory/'示教兼容核验.json',receipt)
    return receipt


def self_test():
    from scipy.spatial.transform import Rotation
    checks=[]
    for stored in [.040,.043]:
        for angles in [[0,0,0],[.3,-.7,1.2]]:
            eef=np.eye(4);eef[:3,:3]=Rotation.from_euler('xyz',angles).as_matrix();eef[:3,3]=[.2,-.1,.4]
            pose=eef.copy();pose[:3,3]+=stored*pose[:3,2]
            c={'pose':pose,'metadata':{'contact_storage_tcp_m':stored}}
            prepare_recall(c);got=c['pose'].copy();got[:3,3]-=TCP*got[:3,2]
            assert np.allclose(got,eef,atol=1e-12)
            previous=c['pose'].copy();prepare_recall(c);assert np.array_equal(previous,c['pose'])
            checks.append({'stored_tcp_m':stored,'rotated':angles!=[0,0,0],'eef_invariant':True,'idempotent':True})
    try:prepare_recall({'pose':np.eye(4),'metadata':{'contact_storage_tcp_m':.042}})
    except ValueError:pass
    else:raise AssertionError('Unknown TCP definition accepted')
    return {'passed':True,'physical_grasps':0,'tests':checks,'unknown_tcp_rejected':True}
