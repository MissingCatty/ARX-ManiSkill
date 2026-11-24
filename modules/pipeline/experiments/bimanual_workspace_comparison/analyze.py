"""Compare single-arm density with dual-arm positional workspace overlap."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
WORKSPACE=PROJECT.parent
ROOT=HERE/'runs/dual_arm_20261123_01'
ATLAS=WORKSPACE/'reachability/产物/前方精细_20261121_01'
FULL=WORKSPACE/'maniskill/assets/arx/robot.urdf'
REFERENCE=WORKSPACE/'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf'
DISTANCE=PROJECT/'experiments/reachability_distance_comparison/runs/map_distance_20261123_01'
PRIMARY=PROJECT/'experiments/full_failure_distance_probe/runs/all53_20261123_01'
BAND=(-.25,-.10)


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024**2):h.update(b)
    return h.hexdigest()


def save(p,v):
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=encode)+'\n')


def origin(joint):
    o=joint.find('origin');t=np.eye(4)
    if o is not None:
        t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
        t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return t


class Arm:
    def __init__(self,path,side):
        root=ET.parse(path).getroot();parents={j.find('child').get('link'):j for j in root.findall('joint')}
        link=side+'_eef_link';self.chain=[]
        while link!='body4':
            j=parents[link];self.chain.append(j);link=j.find('parent').get('link')
        self.chain.reverse();self.joints=[];self.offsets=[];self.axes=[];self.limits=[]
        acc=np.eye(4)
        for j in self.chain:
            acc=acc@origin(j)
            if j.get('type')=='revolute':
                self.joints.append(j.get('name'));self.offsets.append(acc)
                self.axes.append(np.fromstring(j.find('axis').get('xyz'),sep=' '))
                self.limits.append([float(j.find('limit').get(n)) for n in ['lower','upper']]);acc=np.eye(4)
            else:assert j.get('type')=='fixed'
        self.tail=acc;self.limits=np.array(self.limits)
        assert len(self.joints)==6

    def fk(self,q):
        t=np.eye(4);i=0
        for j in self.chain:
            t=t@origin(j)
            if j.get('type')=='revolute':
                axis=np.fromstring(j.find('axis').get('xyz'),sep=' ')
                turn=np.eye(4);turn[:3,:3]=Rotation.from_rotvec(axis*q[i]).as_matrix();t=t@turn;i+=1
        return t


def lookup(ids,query):
    i=np.searchsorted(ids,query);good=i<len(ids);i=np.minimum(i,len(ids)-1)
    good &= ids[i]==query
    return good,i


def encode(cells):
    c=cells.astype(np.int64)+2048
    assert ((c>=0)&(c<4096)).all()
    return (c[:,0]*4096+c[:,1])*4096+c[:,2]


def metrics(cells,counts,resolution,orientation=None):
    ids=encode(cells);assert (np.diff(ids)>0).all()
    shift=int(round(.5/resolution));assert abs(shift*resolution-.5)<1e-12
    lids=ids+shift*4096
    common,index=lookup(lids,ids);lcounts=np.where(common,counts[index],0)
    centers=(cells+.5)*resolution
    scopes={}
    for name,mask in [('all_heights',np.ones(len(ids),bool)),('table_height_band',(centers[:,2]>=BAND[0])&(centers[:,2]<=BAND[1]))]:
        threshold=float(np.quantile(counts[mask],.75));high=mask&(counts>=threshold)
        # Translation changes only Y; the height-selected population and its
        # density threshold are identical for the two arms.
        bothhigh=high&common&(lcounts>=threshold)
        nh=int(high.sum());nc=int((common&mask).sum());n=int(mask.sum())
        scopes[name]={'right_occupied_cells':n,'left_occupied_cells':n,'shared_cells':nc,
            'shared_fraction_of_right_volume':nc/n,'right_high_density_threshold':threshold,
            'right_high_density_cells':nh,'right_high_in_left_cells':int((high&common).sum()),
            'right_high_in_left_fraction':float(common[high].mean()),
            'right_high_also_left_high_cells':int(bothhigh.sum()),
            'right_high_also_left_high_fraction':float(bothhigh.sum()/nh),
            'high_density_iou':float(bothhigh.sum()/(2*nh-bothhigh.sum())),
            'right_high_sample_mass_in_left_fraction':float(counts[high&common].sum()/counts[high].sum()),
            'right_volume_m3':n*resolution**3,'shared_volume_m3':nc*resolution**3}
        if orientation is not None:
            ot=float(np.quantile(orientation[mask],.75));oh=mask&(orientation>=ot)
            oleft=np.where(common,orientation[index],0)
            scopes[name]['orientation_coverage_high']={'threshold_bins':ot,'right_cells':int(oh.sum()),
                'right_high_in_left_fraction':float(common[oh].mean()),
                'right_high_also_left_high_fraction':float((common[oh]&(oleft[oh]>=ot)).mean())}
    return {'resolution_m':resolution,'left_translation_cells_y':shift,'scopes':scopes}


def main():
    assert not ROOT.exists();ROOT.mkdir(parents=True)
    meta=read(ATLAS/'地图元数据.json');done=read(ATLAS/'完成.json');assert done['passed']
    inputs={str(ATLAS/n):h for n,h in done['file_sha256'].items()}
    for p in [FULL,REFERENCE,DISTANCE/'rows.json',PRIMARY/'summary.json']:
        inputs[str(p)]=sha(p)
    save(ROOT/'registration.json',{'at':datetime.now().astimezone().isoformat(),
        'source_sha256':{str(Path(__file__)):sha(__file__)},'input_sha256':inputs,
        'question':'Do the single-right-arm high-density cells lie in the spatial coverage intersection of both arms?',
        'frame':'body4/action_tcp43','space':'front x>=0, positional occupancy with each arm free to choose orientation',
        'primary_resolution_m':.05,'sensitivity_resolution_m':.02,'height_band_body4_m':BAND,
        'high_density_definition':'occupied-cell sample-count 75th percentile, including tied counts, computed separately for whole space and height band',
        'method':'Validate actual left/right URDF chains and joint limits; use an exact chain relation if structurally proved, not a mirror assumption',
        'new_IK_queries':0,'physical_attempts':0,'robot_motion_commands':0,'atlas_modified':False})
    for p,h in inputs.items():assert sha(p)==h,p
    left,right,reference=Arm(FULL,'left'),Arm(FULL,'right'),Arm(REFERENCE,'right')
    assert np.array_equal(left.limits,right.limits)
    assert np.max(np.abs(reference.limits-right.limits))<1e-12
    delta=left.offsets[0][:3,3]-right.offsets[0][:3,3]
    assert np.allclose(delta,[0,.5,0],atol=1e-12)
    assert np.array_equal(left.offsets[0][:3,:3],right.offsets[0][:3,:3])
    for l,r,ref in zip(left.offsets[1:],right.offsets[1:],reference.offsets[1:]):
        assert np.max(np.abs(l-r))<1e-12 and np.max(np.abs(r-ref))<1e-12
    assert np.max(np.abs(right.offsets[0]-reference.offsets[0]))<1e-12
    assert np.max(np.abs(left.tail-right.tail))<1e-12 and np.max(np.abs(right.tail-reference.tail))<1e-12
    assert all(np.array_equal(l,r) and np.array_equal(r,ref) for l,r,ref in zip(left.axes,right.axes,reference.axes))
    bank=np.memmap(ATLAS/meta['witness_file'],dtype=meta['dtype'],mode='r',shape=(meta['witnesses'],len(meta['fields'])))
    tcp=np.array(meta['T_eef_action']);rng=np.random.default_rng(20261123)
    p_error=[];r_error=[];translation_error=[];witnesses=[]
    for i in rng.choice(len(bank),512,replace=False):
        b=np.array(bank[i]);q=b[:6].astype(float)
        assert ((q>=left.limits[:,0])&(q<=left.limits[:,1])).all()
        tl,tr=left.fk(q)@tcp,right.fk(q)@tcp
        rotation=Rotation.from_quat(b[9:13][[1,2,3,0]]).as_matrix()
        p_error.append(float(np.linalg.norm(tr[:3,3]-b[6:9])))
        r_error.append(float(Rotation.from_matrix(tr[:3,:3]@rotation.T).magnitude()))
        translation_error.append(float(np.max(np.abs(tl[:3,3]-tr[:3,3]-delta))))
        assert np.max(np.abs(tl[:3,:3]-tr[:3,:3]))<1e-12
        witnesses.append({'map_row':int(i),'q':q,'right_tcp_body4':tr,'left_tcp_body4':tl})
    assert max(p_error)<2e-5 and max(r_error)<2e-4 and max(translation_error)<1e-12
    save(ROOT/'model_alignment.json',{'passed':True,'actual_left_joint_names':left.joints,'actual_right_joint_names':right.joints,
        'left_right_limits_equal':True,'right_reference_chain_equal':True,
        'left_from_right_translation_body4_m':delta,'orientation_transform':'identical for matching joint vectors',
        'structural_equality_verified':True,'independent_fk_pairs':512,
        'maximum_map_position_error_m':max(p_error),'maximum_map_rotation_error_rad':max(r_error),
        'maximum_translation_relation_error_m':max(translation_error),'physical_attempts':0})
    save(ROOT/'paired_fk_witnesses.json',witnesses)
    coarse=dict(np.load(ATLAS/meta['coarse']['file'],allow_pickle=False))
    results=[metrics(coarse['cells'],coarse['sample_count'],.05,coarse['orientation_count'])]
    fine=dict(np.load(ATLAS/meta['fine']['file'],allow_pickle=False))
    c2=np.floor_divide(fine['cells'].astype(np.int64),4);ids2=encode(c2)
    unique,inverse=np.unique(ids2,return_inverse=True)
    counts2=np.bincount(inverse,weights=fine['sample_count']).astype(np.int64)
    cells2=np.column_stack([unique//4096//4096-2048,unique//4096%4096-2048,unique%4096-2048])
    assert counts2.sum()==meta['witnesses']
    results.append(metrics(cells2,counts2,.02))
    np.savez_compressed(ROOT/'occupancy_2cm.npz',cells_right=cells2,sample_count=counts2,
        cells_left=cells2+[0,25,0],translation_body4_m=delta)
    ids=coarse['voxel_id'];lids=ids+40960;membership=[]
    for row in read(DISTANCE/'rows.json'):
        if row['cohort']!='all53':continue
        p=np.array(row['tcp_mean_body4']);cell=np.floor(p/.05).astype(np.int64)
        key=encode(cell[None])[0];rc,ri=lookup(ids,np.array([key]));lc,li=lookup(lids,np.array([key]))
        membership.append({'id':row['id'],'source_id':row['source_id'],'planning_success':row['planning_success'],
            'point_kind':'mean_native_candidate_TCP','point_body4_m':p,'right_covered':bool(rc[0]),'left_covered':bool(lc[0]),
            'both_covered':bool(rc[0] and lc[0])})
    selected=[]
    source_summary=read(PRIMARY/'summary.json')['cases']
    for r in source_summary:
        if not r['planning_success']:continue
        state=read(PRIMARY/'cases'/r['id']/'state.json')
        t=np.linalg.inv(np.array(state['links']['body4']))@np.array(r['selected']['grasp_world'])@tcp
        cell=np.floor(t[:3,3]/.05).astype(np.int64);key=encode(cell[None])[0]
        rc,_=lookup(ids,np.array([key]));lc,_=lookup(lids,np.array([key]))
        selected.append({'id':r['id'],'selected_tcp_body4':t,'right_covered':bool(rc[0]),'left_covered':bool(lc[0]),
                         'both_covered':bool(rc[0] and lc[0])})
    summary={'resolutions':results,'front_right_witnesses':meta['witnesses'],
        'left_witnesses_derived_from_exact_model_relation':meta['witnesses'],
        'test_positions':318,'planned_positions':131,
        'planned_mean_points_in_both':sum(r['both_covered'] for r in membership if r['planning_success']),
        'selected_grasp_points_in_both':sum(r['both_covered'] for r in selected),
        'left_right_translation_body4_m':delta,'height_band_body4_m':BAND,
        'new_IK_queries':0,'physical_attempts':0,'source_inputs_unchanged':True}
    save(ROOT/'summary.json',summary);save(ROOT/'distance_membership.json',membership);save(ROOT/'selected_grasp_membership.json',selected)
    for p,h in inputs.items():assert sha(p)==h,p
    save(ROOT/'completion_verification.json',{'passed':True,'actual_model_structure_verified':True,
        'independent_fk_pairs':512,'atlas_modified':False,'source_inputs_unchanged':True,'physical_attempts':0,'new_IK_queries':0})
    print(json.dumps(summary,ensure_ascii=False,default=lambda x:x.tolist()),flush=True)


if __name__=='__main__':main()
