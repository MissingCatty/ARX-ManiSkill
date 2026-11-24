"""Use the unmodified installed NVIDIA antipodal sampler on composed object geometry.

Only candidate poses are generated here. Neither SDK evaluation/teleportation nor
any physics setter is used. All accepted demonstrations require full ARX execution.
"""
import copy
from collections import defaultdict
import importlib.util
from pathlib import Path
import time
import numpy as np
from pxr import Usd,UsdGeom
import trimesh
from scripts.curobo_arx.data_factory.common import ROOT,digest,read,sha,write
from scripts.curobo_arx.data_factory.geometry import balanced_group_order,grasp_equal,rigid


def frozen_pad_center():
    import sys
    sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
    from sphere_refinement import PAD_CENTER_EEF
    return np.array(PAD_CENTER_EEF,dtype=float).copy()


def composed_mesh(path):
    """Apply every child transform, rather than sampling only the first USD mesh."""
    stage=Usd.Stage.Open(str(path))
    root=stage.GetDefaultPrim()
    if not root or abs(UsdGeom.GetStageMetersPerUnit(stage)-1.)>1e-9:
        raise ValueError('Candidate USD must have a default prim and metre units')
    cache=UsdGeom.XformCache()
    inverse=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(root)).T)
    meshes=[];sources=[]
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):continue
        imageable=UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility()==UsdGeom.Tokens.invisible:continue
        if imageable.ComputePurpose()==UsdGeom.Tokens.proxy:continue
        mesh=UsdGeom.Mesh(prim)
        points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=int)
        counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),dtype=int)
        if not len(points):continue
        transform=inverse@np.asarray(cache.GetLocalToWorldTransform(prim)).T
        vertices=(np.c_[points,np.ones(len(points))]@transform.T)[:,:3]
        faces=[];offset=0
        for n in counts:
            ids=indices[offset:offset+n];offset+=n
            faces.extend([[int(ids[0]),int(ids[i]),int(ids[i+1])] for i in range(1,n-1)])
        if len(faces):
            meshes.append(trimesh.Trimesh(vertices=vertices,faces=faces,process=False))
            sources.append({'path':str(prim.GetPath()),'mesh_to_object':transform.tolist(),
                'vertices':len(vertices),'triangles':len(faces)})
    if not meshes:raise ValueError('No visible object mesh; do not substitute a primitive')
    combined=trimesh.util.concatenate(meshes)
    if not np.isfinite(combined.vertices).all():raise ValueError('Nonfinite composed geometry')
    return combined,{'usd_sha256':sha(path),'default_prim':str(root.GetPath()),'mesh_parts':sources,
        'size_m':combined.extents.tolist(),'bounds_center_m':combined.bounds.mean(0).tolist(),
        'triangulation':'Original polygon fan, with all authored child transforms',
        'used_layers':{l.realPath:sha(l.realPath) for l in stage.GetUsedLayers() if l.realPath}}


def candidate_pool(transforms,object_world,size,maximum,pad,source,center=None):
    groups=defaultdict(list);rejected=[]
    obj=rigid(object_world)
    for index,contact_frame in enumerate(transforms):
        local=rigid(contact_frame)
        world=obj@local
        contact=world[:3,3].copy()
        world[:3,3]-=world[:3,:3]@pad
        location=(local[:3,3]-(np.zeros(3) if center is None else np.asarray(center)))/np.maximum(np.asarray(size)/2,.001)
        region=(int(np.argmax(abs(location))),int(np.sign(location[np.argmax(abs(location))])))
        direction=(int(np.argmax(abs(world[:3,2]))),int(np.sign(world[np.argmax(abs(world[:3,2])),2])))
        roll=int(np.floor((np.arctan2(world[1,1],world[0,1])%np.pi)/(np.pi/4)))
        group=str((region,direction,roll))
        row={'goal_id':digest({'sdg_G':np.round(world,7).tolist()})[:20],
            'grasp_world':world.tolist(),'target_contact_world':contact.tolist(),
            'object_local_contact_frame':local.tolist(),'pad_center_eef_m':pad.tolist(),
            'contact_symmetry':'contact_center_5mm_10deg_parallel_jaw_v2',
            'raw_index':index,'score':1.,'score_scope':'Uniform geometric proposals; no learned confidence',
            'group':group,'candidate_source':source,'closing_axis_eef':[0,1,0],'approach_axis_eef':[0,0,1]}
        groups[group].append(row)
    order=balanced_group_order(groups)
    selected=[]
    while groups and len(selected)<maximum:
        for key in order:
            if key not in groups:continue
            row=groups[key].pop(0)
            if any(grasp_equal(row,old) for old in selected):
                rejected.append({'index':row['raw_index'],'reason':'parallel_jaw_equivalent_duplicate'})
            else:selected.append(row)
            if not groups[key]:del groups[key]
            if len(selected)>=maximum:break
    return {'candidates':selected,'raw_candidates':len(transforms),'rejected':rejected,
        'candidate_source':source,'reserve_limit':maximum,
        'group_selection':'Deficient surface-region/approach-direction/roll round robin; no downward-only gate',
        'remaining_unselected':sum(map(len,groups.values()))}


class SDGSampler:
    def __init__(self,registration):
        self.registration=registration;self.config=registration['sdg']
        source=Path(self.config['sampler_file'])
        if sha(source)!=self.config['sampler_sha256']:raise RuntimeError('Registered NVIDIA sampler changed')
        spec=importlib.util.spec_from_file_location('arx_official_nvidia_sdg_sampler',source)
        self.sdk=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.sdk)
        self.meshes={};self.pad=frozen_pad_center()
        if not np.array_equal(self.pad,np.asarray(self.config['pad_center_eef_m'])):
            raise RuntimeError('Frozen ARX pad convention changed')

    def local_candidates(self,asset):
        key=asset['object_id']
        if key not in self.meshes:
            bank=self.config.get('candidate_bank')
            if bank:
                root=Path(bank)
                evidence=read(root/(key+'.json'))
                if evidence['asset_sha256']!=asset['asset_sha256']:raise RuntimeError('SDG bank asset changed')
                samples=np.load(root/(key+'.npy'),allow_pickle=False)
                if sha(root/(key+'.npy'))!=evidence['poses_sha256']:raise RuntimeError('SDG candidate bank changed')
                self.meshes[key]=(samples,evidence)
                return self.meshes[key]
            mesh,evidence=composed_mesh(asset['usd_path'])
            params=copy.deepcopy(self.config['sampler_parameters'])
            params['random_seed']=int(digest([self.registration['seed'],asset['geometry_group']])[:8],16)
            started=time.monotonic()
            samples=self.sdk.sample_antipodal(mesh,**params)
            evidence.update(sampler_parameters=params,seconds=time.monotonic()-started,
                source='isaacsim.replicator.grasping.sampler_utils.sample_antipodal',
                extension_version=self.config['extension_version'],physics_evaluation=False)
            self.meshes[key]=(samples,evidence)
        return self.meshes[key]

    def handle(self,request):
        if request['action']!='infer':raise ValueError('Unsupported SDG request')
        directory=Path(request['directory']);asset=request['asset']
        observation=read(directory/'head_observation.json')
        if observation['object_id']!=asset['object_id']:raise RuntimeError('SDG object/observation mismatch')
        world=observation['snapshot']['objects']['test_object']['world']
        samples,evidence=self.local_candidates(asset)
        source={'name':'NVIDIA Grasping SDG antipodal','version':self.config['extension_version'],
            'sampler_sha256':self.config['sampler_sha256'],'type':'ground_truth_mesh_geometric_candidate',
            'not_a_demonstration':True}
        pool=candidate_pool(samples,world,asset['size_m'],self.registration['quotas']['candidate_reserve'],self.pad,source,
            evidence['bounds_center_m'])
        write(directory/'sdg_sampling.json',evidence)
        write(directory/'pool.json',pool)
        return pool
