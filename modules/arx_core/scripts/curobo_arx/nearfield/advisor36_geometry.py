"""Bounded local-surface canonical proposal search, before physical registration."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.curobo_arx.data_factory.common import read,write,sha
from scripts.curobo_arx.sdg_stage1.sampler import composed_mesh
from .rulefree_diagnose import PAD


class Surface:
    def __init__(self,mesh):
        t=np.asarray(mesh.triangles);self.a=t[:,0];self.e=t[:,1]-t[:,0];self.f=t[:,2]-t[:,0]
        n=np.cross(self.e,self.f);size=np.linalg.norm(n,axis=1)
        self.normals=n/np.maximum(size[:,None],1e-30)

    def hit(self,o,d):
        h=np.cross(np.broadcast_to(d,self.f.shape),self.f);det=np.einsum('ij,ij->i',self.e,h)
        valid=abs(det)>1e-12;inv=np.zeros_like(det);inv[valid]=1/det[valid]
        s=o-self.a;u=inv*np.einsum('ij,ij->i',s,h);q=np.cross(s,self.e)
        v=inv*(q@d);t=inv*np.einsum('ij,ij->i',self.f,q)
        valid &= (u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(t>=0)
        ix=np.flatnonzero(valid)
        if not len(ix):return None
        i=int(ix[np.argmin(t[ix])])
        return {'point':o+t[i]*d,'normal':self.normals[i],'face':i}

    def pair(self,center,closing):
        y=np.asarray(closing);y=y/np.linalg.norm(y)
        hits=[self.hit(center+s*y,-s*y) for s in (-1,1)]
        if any(h is None for h in hits):return None
        angles=[float(np.rad2deg(np.arccos(np.clip(s*h['normal']@y,-1,1))))
                for h,s in zip(hits,(-1,1))]
        opposite=float(np.rad2deg(np.arccos(np.clip(-hits[0]['normal']@hits[1]['normal'],-1,1))))
        width=float(np.linalg.norm(hits[1]['point']-hits[0]['point']))
        return {'center':(hits[0]['point']+hits[1]['point'])/2,'closing':y,'angles':angles,
                'normal_opposition_error_degrees':opposite,'width_m':width,'hits':hits}


def pose(center,closing,historical):
    y=np.asarray(closing);y=y/np.linalg.norm(y)
    z=historical[:3,2]-y*(y@historical[:3,2]);z/=np.linalg.norm(z)
    R=np.column_stack([np.cross(y,z),y,z]);g=np.eye(4);g[:3,:3]=R;g[:3,3]=center-R@PAD
    return g


def candidates(item,mesh):
    import trimesh
    obj=item['object_id'];g=np.asarray(item['historical_G_object']);c=g[:3,3]+g[:3,:3]@PAD
    if obj in ('household_hope_milk_carton','raisin_box','cheez_it','household_ycb_spam_can'):
        if obj=='household_ycb_spam_can':
            p=Path(__file__).resolve().parents[3]/'runs/nearfield_advisor35_rulefree_20261112_01/canonical_references.json'
            prior=next(r for r in read(p) if r['object_id']==obj);fixed=np.asarray(prior['Gcanonical_object'])
            yield fixed[:3,3]+fixed[:3,:3]@PAD,fixed[:3,1],'prior_registered_Spam',3.
            return
        T,ext=trimesh.bounds.oriented_bounds(mesh,angle_digits=1);B=np.linalg.inv(T)
        axis=int(np.argmax(abs(B[:3,:3].T@g[:3,1])));y=B[:3,axis].copy();y*=np.sign(y@g[:3,1])
        p=(T@np.r_[c,1])[:3];p[axis]=0.;center=(B@np.r_[p,1])[:3]
        others=[i for i in range(3) if i!=axis]
        values=[0.,.003,-.003,.006,-.006,.012,-.012,.020,-.020]
        for a,b in sorted(((a,b) for a in values for b in values),key=lambda x:x[0]**2+x[1]**2):
            yield center+a*B[:3,others[0]]+b*B[:3,others[1]],y,'box_local_parallel_surfaces',3.
    elif obj=='ycb_056_tennis_ball':
        v=np.asarray(mesh.vertices);A=np.c_[2*v,np.ones(len(v))];fit=np.linalg.lstsq(A,(v*v).sum(1),rcond=None)[0]
        center=fit[:3];up=np.asarray(item['source_object_world'])[:3,:3].T@np.array([0.,0.,1.])
        base=g[:3,1]-up*(up@g[:3,1]);base/=np.linalg.norm(base)
        for angle in (0,5,-5,10,-10,15,-15,20,-20):
            y=Rotation.from_rotvec(up*np.deg2rad(angle)).apply(base)
            yield center,y,'sphere_fitted_center_equatorial_diameter',5.
    else:
        v=np.asarray(mesh.vertices)
        if 'masher' in obj:
            # The upper handle, not the perforated lower head or old pinch.
            zlo,zhi=mesh.bounds[:,2];lo=zlo+.55*(zhi-zlo);hi=zlo+.85*(zhi-zlo)
            handle=v[(v[:,2]>=lo)&(v[:,2]<=hi)];center=handle.mean(0)
            _,_,vh=np.linalg.svd(handle-center,full_matrices=False);axis=vh[0]
            if axis[2]<0:axis=-axis
            heights=np.linspace(np.quantile(handle@axis,.15),np.quantile(handle@axis,.85),9)
            kind='actual_handle_local_cross_section'
        else:
            axis=np.asarray(item['source_object_world'])[:3,:3].T@np.array([0.,0.,1.]);axis/=np.linalg.norm(axis)
            heights=(c@axis)+np.array([0.,-.005,.005,-.010,.010,-.015,.015,-.020,.020,-.025,.025])
            kind='bottle_local_cross_section'
        y=g[:3,1]-axis*(axis@g[:3,1]);y/=np.linalg.norm(y);x=np.cross(y,axis)
        for h in heights:
            near=v[abs(v@axis-h)<.003]
            if len(near)<8:continue
            mx=(np.min(near@x)+np.max(near@x))/2;my=(np.min(near@y)+np.max(near@y))/2
            center=h*axis+mx*x+my*y
            angles=(0,5,-5,10,-10,15,-15,20,-20,30,-30,45,-45)
            for angle in angles:
                closing=Rotation.from_rotvec(axis*np.deg2rad(angle)).apply(y)
                for offset in (0.,.003,-.003):
                    yield center+offset*x,closing,kind,5.


def search(item):
    mesh,proof=composed_mesh(item['asset']['usd_path']);surface=Surface(mesh)
    g=np.asarray(item['historical_G_object']);records=[]
    for center,y,kind,limit in candidates(item,mesh):
        pair=surface.pair(center,y)
        if pair is None:continue
        # For curved surfaces, align to the actual opposing normals, then
        # re-intersect; do not rotate a reported normal to make it pass.
        if kind in ('bottle_local_cross_section','actual_handle_local_cross_section'):
            for _ in range(3):
                ny=pair['hits'][1]['normal']-pair['hits'][0]['normal']
                if np.linalg.norm(ny)<1e-8:break
                ny/=np.linalg.norm(ny)
                if ny@y<0:ny=-ny
                improved=surface.pair(pair['center'],ny)
                if improved is None or max(improved['angles'])>=max(pair['angles']):break
                pair=improved
        goal=pose(pair['center'],pair['closing'],g)
        if kind=='prior_registered_Spam':
            p=Path(__file__).resolve().parents[3]/'runs/nearfield_advisor35_rulefree_20261112_01/canonical_references.json'
            goal=np.asarray(next(r['Gcanonical_object'] for r in read(p) if r['object_id']==item['object_id']))
        angle=float(Rotation.from_matrix(g[:3,:3].T@goal[:3,:3]).magnitude()*180/np.pi)
        distance=float(np.linalg.norm(g[:3,3]-goal[:3,3]))
        valid=max(pair['angles'])<=limit and .004<pair['width_m']<.1
        entry={'index':len(records),'method':kind,'pass_local_geometry':bool(valid),
            'normal_limit_degrees':limit,'contact_normal_errors_degrees':pair['angles'],
            'normal_opposition_error_degrees':pair['normal_opposition_error_degrees'],
            'contact_width_m':pair['width_m'],'goal_position_change_m':distance,'goal_rotation_change_degrees':angle,
            'Gcanonical_object':goal.tolist(),'pad_center_object':pair['center'].tolist(),
            'hits':[{'point':h['point'].tolist(),'normal':h['normal'].tolist(),'face':h['face']} for h in pair['hits']],
            'cost':distance*1000+angle*.3}
        records.append(entry)
    good=[r for r in records if r['pass_local_geometry']]
    selected=min(good,key=lambda r:(r['cost'],max(r['contact_normal_errors_degrees']),r['index'])) if good else None
    return {'object_id':item['object_id'],'source_seed_id':item['seed_id'],'source_split':item['source_split'],
        'pass_local_geometry':selected is not None,'selected':selected,'candidates':records,
        'mesh_winding_consistent':bool(mesh.is_winding_consistent),'mesh_proof':proof,
        'physical_preflight_pass':False,'scope':'Actual triangle normals; no IK, path, physical grasp or continuum-envelope guarantee yet.'}


def local_neighbors(item,anchor):
    """Finite reselection around a failed, already normal-qualified target.

    Roll about the closing axis preserves the normal requirement. Box/contact
    center translations stay within its local tangent plane; a sphere's center
    is never shifted off its fitted equatorial diameter.
    """
    mesh,proof=composed_mesh(item['asset']['usd_path']);surface=Surface(mesh)
    g=np.asarray(anchor['Gcanonical_object']);old=np.asarray(item['historical_G_object'])
    center=g[:3,3]+g[:3,:3]@PAD;obj=item['object_id'];limit=anchor['normal_limit_degrees']
    offsets=[0.,.003,-.003,.006,-.006,.012,-.012,.020,-.020]
    if obj=='ycb_056_tennis_ball':offsets=[0.]
    rows=[]
    for a in offsets:
        for b in offsets:
            pair=surface.pair(center+a*g[:3,0]+b*g[:3,2],g[:3,1])
            if pair is None:continue
            for roll in (0.,3.,-3.,6.,-6.,9.,-9.,12.,-12.):
                oriented=g.copy();oriented[:3,:3]=g[:3,:3]@Rotation.from_rotvec([0.,np.deg2rad(roll),0.]).as_matrix()
                goal=pose(pair['center'],pair['closing'],oriented)
                distance=float(np.linalg.norm(old[:3,3]-goal[:3,3]))
                angle=float(Rotation.from_matrix(old[:3,:3].T@goal[:3,:3]).magnitude()*180/np.pi)
                valid=max(pair['angles'])<=limit and .004<pair['width_m']<.1
                rows.append({'index':10000+len(rows),'method':'local_tangent_and_closing_axis_roll_reselection',
                    'parent_method':anchor['method'],'pass_local_geometry':bool(valid),'normal_limit_degrees':limit,
                    'contact_normal_errors_degrees':pair['angles'],'normal_opposition_error_degrees':pair['normal_opposition_error_degrees'],
                    'contact_width_m':pair['width_m'],'goal_position_change_m':distance,'goal_rotation_change_degrees':angle,
                    'Gcanonical_object':goal.tolist(),'pad_center_object':pair['center'].tolist(),
                    'hits':[{'point':h['point'].tolist(),'normal':h['normal'].tolist(),'face':h['face']} for h in pair['hits']],
                    'cost':distance*1000+angle*.3,'local_tangent_translation_m':[a,b],
                    'closing_axis_roll_degrees':roll})
    return {'object_id':obj,'anchor':anchor,'candidates':rows,'mesh_proof':proof,
        'search_is_finite':True,'physical_trials':0,'normals_and_core_unchanged':True,
        'sphere_center_fixed':obj=='ycb_056_tennis_ball','source_sha256':sha(__file__)}


def main(root):
    root=Path(root).resolve();out=root/'geometry';out.mkdir(exist_ok=False)
    results=[]
    for item in read(root/'source_inventory.json'):
        result=search(item);write(out/(item['object_id']+'.json'),result);results.append(result)
        selected=result['selected'];print(item['object_id'],result['pass_local_geometry'],
            selected['contact_normal_errors_degrees'] if selected else None,
            selected['goal_position_change_m'] if selected else None,flush=True)
    write(root/'canonical_candidates.json',[{k:v for k,v in r.items() if k!='candidates'} for r in results])
    write(root/'geometry_completion.json',{'objects':len(results),'geometric_passes':sum(r['pass_local_geometry'] for r in results),
        'physical_trials':0,'source_sha256':sha(__file__),'files':{str(p):sha(p) for p in out.glob('*.json')}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
