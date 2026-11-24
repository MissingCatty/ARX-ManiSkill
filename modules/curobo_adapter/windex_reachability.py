"""Independent analytic and bounded numerical IK for the fixed Windex target."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from urdf_reference import URDFChain, NAMES

HERE=Path(__file__).resolve().parent
ROOT=HERE/'research/20261117_06'
URDF=HERE.parent/'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf'


def ry(angle):
    return Rotation.from_rotvec([0.,angle,0.]).as_matrix()


def rz(angle):
    return Rotation.from_rotvec([0.,0.,angle]).as_matrix()


def rx(angle):
    return Rotation.from_rotvec([angle,0.,0.]).as_matrix()


def turns(angle,lower,upper):
    return [angle+2*np.pi*k for k in range(-3,4) if lower-1e-10<=angle+2*np.pi*k<=upper+1e-10]


class AnalyticARX:
    """Enumerate the nonsingular branches of this authored ZYYYZX chain."""
    def __init__(self):
        joints={j.get('name'):j for j in ET.parse(URDF).getroot().findall('joint')}
        self.offset=[]
        for name in NAMES:
            self.offset.append(np.fromstring(joints[name].find('origin').get('xyz'),sep=' '))
        tool=joints['right_link26_right_eef_link_joint'].find('origin')
        self.tool=np.fromstring(tool.get('xyz'),sep=' ')
        self.tool_rotation=Rotation.from_euler('xyz',np.fromstring(tool.get('rpy'),sep=' ')).as_matrix()
        self.reference=URDFChain(URDF)
        assert abs(self.offset[2][1]+self.offset[3][1])<1e-12
        assert self.offset[1][1]==self.offset[4][1]==self.offset[5][1]==0.
        assert self.offset[2][0]<0 and np.linalg.norm(self.offset[2][[1,2]])<1e-4

    def direct(self,q):
        a,b,c,d,e,f=q
        pitch=b-c-d
        R1=rz(a);R2=R1@ry(b);R3=R1@ry(b-c);R4=R1@ry(pitch);R5=R4@rz(-e);R6=R5@rx(f)
        p=self.offset[0]+R1@self.offset[1]+R2@self.offset[2]+R3@self.offset[3]+R4@self.offset[4]+R5@self.offset[5]+R6@self.tool
        T=np.eye(4);T[:3,:3]=R6@self.tool_rotation;T[:3,3]=p
        return T

    def roots(self,goal,bounds):
        # Desired joint-26 frame; tool offset is subtracted in its own authored
        # frame, without changing the real robot calibration or action TCP.
        R=Rotation.from_matrix(goal[:3,:3]).as_matrix()@self.tool_rotation.T
        p26=goal[:3,3]-R@self.tool
        shoulder_vector=p26-self.offset[0]-self.offset[5][0]*R[:,0]
        radius=np.linalg.norm(shoulder_vector[:2])
        if radius<1e-9:
            return {'enumeration_complete':False,'reason':'shoulder_yaw_singularity','roots':[]}
        yaw0=math.atan2(shoulder_vector[1],shoulder_vector[0])
        yaw_values=turns(yaw0,*bounds[0])+turns(yaw0+np.pi,*bounds[0])
        branches=[];roots=[]
        L1=-self.offset[2][0];L2=np.linalg.norm(self.offset[3][[0,2]])
        alpha=math.atan2(self.offset[3][2],self.offset[3][0])
        for yaw in yaw_values:
            U=rz(-yaw)@R
            sine=float(np.clip(U[1,0],-1.,1.))
            psi0=math.asin(sine)
            if abs(math.cos(psi0))<1e-8:
                return {'enumeration_complete':False,'reason':'wrist_euler_singularity','roots':roots}
            for psi in [psi0,np.pi-psi0]:
                for q5 in turns(-psi,*bounds[4]):
                    cosine=math.cos(psi)
                    theta=math.atan2(-U[2,0]/cosine,U[0,0]/cosine)
                    q6base=math.atan2(-U[1,2]/cosine,U[1,1]/cosine)
                    q6values=turns(q6base,*bounds[5])
                    planar=rz(-yaw)@(p26-self.offset[0])-self.offset[1]-ry(theta)@self.offset[4]-ry(theta)@rz(-q5)@self.offset[5]
                    assert abs(planar[1])<1e-8
                    length=np.linalg.norm(planar[[0,2]])
                    cosine_elbow=(L1*L1+L2*L2-length*length)/(2*L1*L2)
                    info={'yaw':yaw,'pitch_sum':theta,'q5':q5,'q6_values':q6values,
                          'planar_radius_m':float(length),'maximum_2link_reach_m':float(L1+L2),
                          'elbow_cosine':float(cosine_elbow),'within_2link_radial_workspace':bool(abs(cosine_elbow)<=1+1e-10)}
                    branches.append(info)
                    if abs(cosine_elbow)>1+1e-10:
                        continue
                    angle=math.acos(float(np.clip(cosine_elbow,-1,1)))
                    for elbow_angle in [angle,-angle]:
                        for q3 in turns(elbow_angle-alpha,*bounds[2]):
                            complex_arm=-L1+L2*np.exp(1j*(alpha+q3))
                            q2base=float(np.angle(complex_arm)-math.atan2(planar[2],planar[0]))
                            for q2 in turns(q2base,*bounds[1]):
                                for q4 in turns(q2-q3-theta,*bounds[3]):
                                    for q6 in q6values:
                                        q=np.array([yaw,q2,q3,q4,q5,q6])
                                        own,_=self.reference.fk(q)
                                        pe=float(np.linalg.norm(own[:3,3]-goal[:3,3]));re=float(Rotation.from_matrix(own[:3,:3]@goal[:3,:3].T).magnitude())
                                        assert pe<1e-8 and re<1e-7,(q,pe,re)
                                        if not any(np.max(np.abs(q-np.asarray(x['q'])))<1e-7 for x in roots):
                                            roots.append({'q':q.tolist(),'position_error_m':pe,'orientation_error_rad':re})
        return {'enumeration_complete':True,'assumptions':'Exact target orientation and position, this nonsingular chain, original joint bounds; does not enumerate the entire tolerance region',
                'yaw_candidates':yaw_values,'orientation_branches':branches,'roots':roots}


def main():
    spec=json.loads((ROOT/'registration_v2.json').read_text());started=time.monotonic()
    import hashlib
    for name,h in spec['hashes'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    source=Path(spec['input']);state=json.loads((source/'input/state.json').read_text())
    bounds=np.asarray([state['joint_limits'][state['joint_names'].index(n)] for n in NAMES])
    measured=np.asarray([state['qpos'][state['joint_names'].index(n)] for n in NAMES])
    chain=URDFChain(URDF);analytic=AnalyticARX();rng=np.random.default_rng(spec['seed'])
    max_fk_error=0.;root_regressions=0
    for q in rng.uniform(bounds[:,0]+1e-5,bounds[:,1]-1e-5,size=(128,6)):
        actual,_=chain.fk(q);derived=analytic.direct(q)
        error=float(np.max(np.abs(actual-derived)));max_fk_error=max(max_fk_error,error);assert error<1e-12
        enumerated=analytic.roots(actual,bounds)
        if enumerated['enumeration_complete']:
            assert any(np.max(np.abs(q-np.asarray(r['q'])))<1e-7 for r in enumerated['roots']),q
            root_regressions+=1
    assert root_regressions>=120
    calls=list(map(json.loads,(source/'calls.jsonl').read_text().splitlines()))
    target=np.linalg.inv(np.asarray(state['links']['body4']))@np.asarray(calls[-1]['goal'])
    own,_=chain.fk(measured);expected=np.linalg.inv(np.asarray(state['links']['body4']))@np.asarray(state['links']['right_eef_link'])
    assert np.linalg.norm(own[:3,3]-expected[:3,3])<2e-5
    assert Rotation.from_matrix(own[:3,:3]@expected[:3,:3].T).magnitude()<2e-4
    roots=analytic.roots(target,bounds)
    (ROOT/'analytic_results.json').write_text(json.dumps({'factorization_max_error':max_fk_error,'known_q_root_regressions':root_regressions,'target_in_base':target.tolist(),**roots},indent=2))
    print('ANALYTIC',json.dumps(roots),flush=True)
    seeds=[measured,np.asarray(calls[-1]['native_phases'][-1]['optimized_joint_positions'][calls[-1]['native_phases'][-1]['best_seed']])]
    seeds+=list(rng.uniform(bounds[:,0]+1e-6,bounds[:,1]-1e-6,size=(spec['maximum_cpu_solves']-len(seeds),6)))
    records=[];witnesses=[]
    with (ROOT/'cpu_requests.jsonl').open('x') as requests,(ROOT/'cpu_calls.jsonl').open('x') as stream:
        for i,q0 in enumerate(seeds):
            assert time.monotonic()-started<spec['maximum_cpu_wall_seconds']
            requests.write(json.dumps({'id':i,'q0':q0.tolist()})+'\n');requests.flush()
            solve=least_squares(lambda q:chain.residual(q,target)[0],np.clip(q0,bounds[:,0]+1e-6,bounds[:,1]-1e-6),
                jac=lambda q:chain.residual(q,target)[1],bounds=(bounds[:,0],bounds[:,1]),max_nfev=250,
                ftol=1e-11,xtol=1e-11,gtol=1e-11,x_scale='jac')
            fk,_=chain.fk(solve.x);pe=float(np.linalg.norm(fk[:3,3]-target[:3,3]));re=float(Rotation.from_matrix(fk[:3,:3]@target[:3,:3].T).magnitude())
            row={'id':i,'q':solve.x.tolist(),'position_error_m':pe,'orientation_error_rad':re,
                 'pose_valid':pe<.002 and re<.03,'nfev':solve.nfev,'status':int(solve.status),'collision_checked':False}
            stream.write(json.dumps(row)+'\n');stream.flush();records.append(row)
            if row['pose_valid'] and not any(np.linalg.norm(solve.x-np.asarray(w['q']))<1e-4 for w in witnesses):witnesses.append(row)
    (ROOT/'cpu_results.json').write_text(json.dumps({'solves':len(records),'pose_valid_solves':sum(r['pose_valid'] for r in records),
        'distinct_witnesses':witnesses,'best_position_error_m':min(r['position_error_m'] for r in records),
        'wall_seconds':time.monotonic()-started,'no_scene_constraints_in_math_diagnostic_only':True,'physical_trials':0},indent=2))
    print('CPU',len(records),'valid',sum(r['pose_valid'] for r in records),'distinct',len(witnesses),flush=True)


if __name__=='__main__':main()
