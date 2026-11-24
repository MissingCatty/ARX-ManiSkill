"""Offline, geometry-derived comparison of the true old and proposed action TCP."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import yaml

from audit_collision_cover import link_frames
from audit_runtime_colliders import runtime_geometry

HERE=Path(__file__).resolve().parent
ARX=HERE.parent/'arx_core'
ROOT=HERE/'research/20261117_07_action_tcp'
sys.path.insert(0,str(ARX/'scripts/curobo_arx'))
import action_frame as interface


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    reg=json.loads((ROOT/'registration.json').read_text())
    for path,expected in reg['protected_hashes'].items():
        assert sha(path)==expected,path
    source=Path(reg['input'])
    state=json.loads((source/'input/state.json').read_text())
    cfg=yaml.safe_load((source/'model/arx.yml').read_text())['robot_cfg']['kinematics']
    scene=json.loads((source/'model/scene.json').read_text())
    old=np.asarray(interface.EEF_TCP).copy();old[2,3]=.040
    assert np.allclose(old[:3,3],[0,0,.040],atol=1e-12)
    proposed=old.copy();proposed[2,3]=reg['new_offset_eef_z_m']
    measured=[state['qpos'][state['joint_names'].index(f'right_joint{k}')] for k in range(21,27)]
    qtarget=json.loads((HERE/'research/20261117_06/analytic_results.json').read_text())['roots'][0]['q']
    geometry=runtime_geometry(HERE.parent/'maniskill/assets/arx')
    def frames_for(q):
        values=dict(cfg['lock_joints']);values.update(dict(zip([f'right_joint{k}' for k in range(21,27)],q)))
        return link_frames(values)
    actual_frames=frames_for(measured);eef=actual_frames['right_eef_link']
    tips={}
    for name in ['right_link27','right_link28']:
        points=np.vstack(geometry[name]);base=points@actual_frames[name][:3,:3].T+actual_frames[name][:3,3]
        local=(base-eef[:3,3])@eef[:3,:3]
        tips[name]=float(local[:,2].max())
    tip=min(tips.values())
    assert old[2,3]<proposed[2,3]<tip
    profile={'profile_id':'arx_action_tcp_43mm_v1','status':'offline_candidate',
        'relative_to':'right_eef_link','T_eef_action':proposed.tolist(),
        'translation_eef_m':proposed[:3,3].tolist(),
        'baseline_T_eef_action':old.tolist(),'forward_shift_from_true_baseline_m':float(proposed[2,3]-old[2,3]),
        'authored_tip_axial_position_eef_z_m':tips,'axial_gap_before_m':float(tip-old[2,3]),
        'axial_gap_after_m':float(tip-proposed[2,3]),'gap_definition':'Axial gap to distal tip plane, not Euclidean distance to each open finger',
        'axes':'Unchanged: action X=EEF Y, action Y=-EEF X, action Z=EEF Z',
        'source_geometry':'Original existing source convex parts, measured finger joint positions',
        'robot_kinematics_or_calibration_modified':False,'runtime_default_switched':False}
    (ROOT/'action_tcp_43mm.json').write_text(json.dumps(profile,indent=2)+'\n')
    box=scene['cuboid']['table'];p=box['pose'];rotation=Rotation.from_quat([*p[4:],p[3]]).as_matrix();half=np.asarray(box['dims'])/2
    def tabletop(points):
        v=(points-p[:3])@rotation;v[:,2]-=half[2];return v*100
    cert=json.loads((HERE/'research/20261117_06/tolerance_certificate.json').read_text())
    witness=cert['native_source_vertex_certificates'][0]
    sphere=max(cert['sphere_certificates'],key=lambda v:v['guaranteed_collision_margin_m'])
    fig,axes=plt.subplots(1,2,figsize=(14,7));all_points=[];frame_records=[]
    for ax,q,title in zip(axes,[measured,qtarget],['Actually reached after first +4 cm','Requested next +3 cm (not executed)']):
        frames=frames_for(q);eef=frames['right_eef_link']
        for link,color in [('right_link26','#888888'),('right_link27','#1f78b4'),('right_link28','#ff7f00')]:
            for i,vertices in enumerate(geometry[link]):
                v=tabletop(vertices@frames[link][:3,:3].T+frames[link][:3,3]);projection=v[:,[0,2]];all_points.append(projection)
                if len(v)>=3 and np.linalg.matrix_rank(projection-projection.mean(0))==2:
                    outline=projection[ConvexHull(projection).vertices];ax.fill(outline[:,0],outline[:,1],color=color,alpha=.13)
                    outline=np.vstack([outline,outline[0]]);ax.plot(outline[:,0],outline[:,1],color=color,lw=.7,label=link if i==0 else None)
        base=tabletop(eef[:3,3][None])[0];previous=tabletop((eef@old)[:3,3][None])[0];new=tabletop((eef@proposed)[:3,3][None])[0]
        axial_tip=tabletop((eef[:3,3]+tip*eef[:3,2])[None])[0]
        point=frames[witness['link']][:3,:3]@witness['source_vertex_link']+frames[witness['link']][:3,3];point=tabletop(point[None])[0]
        center=tabletop((eef[:3,:3]@sphere['center_eef']+eef[:3,3])[None])[0]
        ax.add_patch(Circle(center[[0,2]],sphere['radius_m']*100,fill=False,color='red',lw=1.8,label='Original collision sphere'))
        ax.plot([base[0],new[0]],[base[2],new[2]],color='gray',ls=':',lw=1)
        ax.scatter([base[0]],[base[2]],marker='x',s=55,color='#777777',label='EEF origin (not action TCP)')
        ax.scatter([previous[0]],[previous[2]],marker='o',s=90,facecolors='none',edgecolors='#e66101',lw=1.8,label='True baseline action TCP: +40 mm')
        ax.scatter([new[0]],[new[2]],marker='+',s=160,color='black',lw=2,label='New action TCP: +43 mm',zorder=9)
        ax.scatter([point[0]],[point[2]],marker='D',s=30,color='red',label='Source finger vertex')
        ax.axhspan(-3,0,color='#bdbdbd',alpha=.6);ax.axhline(0,color='black')
        ax.set(title=title,xlabel='Table-local X (cm)',ylabel='Height above table (cm)');ax.grid(alpha=.2)
        zoom=ax.inset_axes([.65,.68,.33,.28])
        zoom.scatter([previous[0]],[previous[2]],marker='o',s=75,facecolors='none',edgecolors='#e66101',lw=1.6)
        zoom.scatter([new[0]],[new[2]],marker='+',s=100,color='black',lw=1.8)
        zoom.scatter([axial_tip[0]],[axial_tip[2]],marker='.',s=35,color='#008577')
        zoom.plot([previous[0],axial_tip[0]],[previous[2],axial_tip[2]],ls=':',color='gray')
        zoom.annotate('40 mm',previous[[0,2]],xytext=(-42,10),textcoords='offset points',fontsize=8,color='#c25200')
        zoom.annotate('43 mm',new[[0,2]],xytext=(-42,-12),textcoords='offset points',fontsize=8)
        zoom.annotate('Tip plane',axial_tip[[0,2]],xytext=(-36,-27),textcoords='offset points',fontsize=7,color='#008577')
        zoom.set(xlim=(new[0]-.65,new[0]+.65),ylim=(new[2]-.65,new[2]+.65),title='TCP detail (cm)')
        zoom.tick_params(labelsize=7);zoom.title.set_fontsize(8);zoom.grid(alpha=.2);zoom.set_aspect('equal',adjustable='box')
        frame_records.append({'panel':title,'eef_origin_table_cm':base.tolist(),'true_baseline_tcp_table_cm':previous.tolist(),
            'new_tcp_table_cm':new.tolist(),'axial_tip_reference_table_cm':axial_tip.tolist(),'source_finger_vertex_table_cm':point.tolist(),
            'new_tcp_in_base':(eef@proposed).tolist(),'baseline_tcp_in_base':(eef@old).tolist()})
        if title.startswith('Requested'):
            ax.annotate('Source vertex remains 4.89 mm below table',xy=point[[0,2]],xytext=(.035,.035),textcoords='axes fraction',arrowprops={'arrowstyle':'->'},fontsize=8.5)
    merged=np.vstack(all_points)
    for ax in axes:
        ax.set_xlim(merged[:,0].min()-2,merged[:,0].max()+3);ax.set_ylim(-3,merged[:,1].max()+2);ax.set_aspect('equal',adjustable='box')
    axes[0].legend(fontsize=7.5,loc='upper left')
    fig.suptitle('Action TCP comparison: true +40 mm baseline to +43 mm near-tip reference\nSame saved robot poses and collision geometry; gray X is EEF origin, black + is the new action TCP',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(ROOT/'action_tcp_43mm_comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)
    (ROOT/'projected_points.json').write_text(json.dumps({'panels':frame_records,'units':'cm in table frame; homogeneous transforms in m'},indent=2)+'\n')
    for path,expected in reg['protected_hashes'].items():assert sha(path)==expected,path
    (ROOT/'verification.json').write_text(json.dumps({'passed':True,'profile_offset_eef_z_m':float(proposed[2,3]),
        'true_baseline_offset_eef_z_m':float(old[2,3]),'tip_plane_eef_z_m':tip,
        'old_axial_gap_mm':(tip-old[2,3])*1000,'new_axial_gap_mm':(tip-proposed[2,3])*1000,
        'original_files_unchanged':True,'robot_model_changed':False,'runtime_default_switched':False,
        'physical_trials':0,'cloud_requests':0,'figure_origin_label_corrected':True},indent=2)+'\n')
    print('NEW TCP',proposed[:3,3].tolist(),'tip gap mm',(tip-proposed[2,3])*1000,flush=True)


if __name__=='__main__':main()
