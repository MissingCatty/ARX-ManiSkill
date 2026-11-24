"""Static source-geometry illustration of the saved and rejected EEF poses."""
from pathlib import Path
import json
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
ROOT=HERE/'research/20261117_06'


def main():
    spec=json.loads((ROOT/'native_registration.json').read_text());source=Path(spec['input'])
    state=json.loads((source/'input/state.json').read_text());cfg=yaml.safe_load((source/'model/arx.yml').read_text())['robot_cfg']['kinematics'];scene=json.loads((source/'model/scene.json').read_text());box=scene['cuboid']['table'];p=box['pose'];rot=Rotation.from_quat([*p[4:],p[3]]).as_matrix();half=np.asarray(box['dims'])/2
    geometry=runtime_geometry(HERE.parent/'maniskill/assets/arx')
    root=json.loads((ROOT/'analytic_results.json').read_text())['roots'][0]['q']
    cert=json.loads((ROOT/'tolerance_certificate.json').read_text());witness=cert['native_source_vertex_certificates'][0];sphere=max(cert['sphere_certificates'],key=lambda x:x['guaranteed_collision_margin_m'])
    measured=[state['qpos'][state['joint_names'].index(f'right_joint{k}')] for k in range(21,27)]
    def tabletop(vertices):
        v=(vertices-p[:3])@rot;v[:,2]-=half[2];return v*100
    fig,axes=plt.subplots(1,2,figsize=(12,6));all_points=[]
    for ax,q,title in zip(axes,[measured,root],['Actually reached after first +4 cm','Requested next +3 cm (not executed)']):
        values=dict(cfg['lock_joints']);values.update(dict(zip([f'right_joint{k}' for k in range(21,27)],q)));frames=link_frames(values)
        for link,color in [('right_link26','#888888'),('right_link27','#1f78b4'),('right_link28','#ff7f00')]:
            for i,vertices in enumerate(geometry[link]):
                world=vertices@frames[link][:3,:3].T+frames[link][:3,3];v=tabletop(world);projection=v[:,[0,2]];all_points.append(projection)
                if len(v)>=3 and np.linalg.matrix_rank(projection-projection.mean(0))==2:
                    outline=projection[ConvexHull(projection).vertices]
                    ax.fill(outline[:,0],outline[:,1],color=color,alpha=.15)
                    outline=np.vstack([outline,outline[0]]);ax.plot(outline[:,0],outline[:,1],color=color,lw=.7,label=link if i==0 else None)
        eef=frames['right_eef_link'];tcp=tabletop(eef[:3,3][None])[0]
        point=frames[witness['link']][:3,:3]@witness['source_vertex_link']+frames[witness['link']][:3,3];point=tabletop(point[None])[0]
        center=eef[:3,:3]@sphere['center_eef']+eef[:3,3];center=tabletop(center[None])[0]
        ax.add_patch(Circle(center[[0,2]],sphere['radius_m']*100,fill=False,color='red',lw=2,label='Original collision sphere'))
        ax.scatter([tcp[0]],[tcp[2]],marker='+',s=100,color='black',label='Action TCP')
        ax.scatter([point[0]],[point[2]],marker='D',s=35,color='red',label='Source finger vertex')
        ax.axhspan(-3,0,color='#bdbdbd',alpha=.6);ax.axhline(0,color='black')
        ax.set(title=title,xlabel='Table-local X (cm)',ylabel='Height above table (cm)');ax.grid(alpha=.2)
        if title.startswith('Requested'):
            ax.annotate('Finger source vertex 4.89 mm below table',xy=point[[0,2]],xytext=(.04,.035),textcoords='axes fraction',arrowprops={'arrowstyle':'->'},fontsize=9)
    merged=np.vstack(all_points)
    for ax in axes:
        ax.set_xlim(merged[:,0].min()-2,merged[:,0].max()+3)
        ax.set_ylim(-3,merged[:,1].max()+2);ax.set_aspect('equal',adjustable='box')
    axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Fixed open gripper: endpoint/table conflict\nStatic geometry reconstruction from saved ARX state; no new robot motion')
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(ROOT/'windex_table_collision.png',dpi=170,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
