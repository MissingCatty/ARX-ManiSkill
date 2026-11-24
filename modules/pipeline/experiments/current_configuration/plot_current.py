"""Draw the active 871-sphere model and TCP43 from original convex geometry."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon
from scipy.spatial import ConvexHull
from configuration import PROJECT,WORKSPACE,load_profile,load_current

sys.path.insert(0,str(PROJECT/'experiments/fingertip_table_tolerance'))
from geometry import Geometry
OUT=WORKSPACE/'claro/进度网站/内容/资源/active-configuration'


def main():
    config=load_current();profile=load_profile()
    state=json.loads((Path(__file__).parent/'records/validation_20261124_01/state.json').read_text())
    q=np.array([state['qpos'][state['joint_names'].index('right_joint'+str(i))] for i in range(21,27)])
    geometry=Geometry(config['planning_models']['open'],{'links':{'body4':np.eye(4)}})
    frames={k:v[0] for k,v in geometry.frames(q).items()}
    parts={};balls={}
    fitting=WORKSPACE/'maniskill/outputs/arx_collision_spheres_20261119_01'
    for name,values in profile['collision_spheres'].items():
        t=frames[name]
        with np.load(fitting/(name+'.npz')) as z:
            parts[name]=[(z[k]/1000)@t[:3,:3].T+t[:3,3] for k in z.files if k.startswith('convex_')]
        s=np.array([[*b['center'],b['radius']] for b in values])
        balls[name]=np.c_[s[:,:3]@t[:3,:3].T+t[:3,3],s[:,3]]
    eye=np.array([1.,1.4,.7]);eye/=np.linalg.norm(eye)
    right=np.cross([0.,0.,1.],eye);right/=np.linalg.norm(right)
    up=np.cross(eye,right);camera=np.stack([right,up,eye],axis=1)

    def draw(ax,names,show_spheres):
        for name in names:
            for v in parts[name]:
                xy=(v@camera)[:,:2]*1000
                ax.add_patch(Polygon(xy[ConvexHull(xy).vertices],facecolor='#bdc7d0',edgecolor='#8a969f',linewidth=.4))
        if show_spheres:
            combined=np.concatenate([balls[n] for n in names]);combined[:,:3]=combined[:,:3]@camera
            for x,y,z,r in sorted(combined,key=lambda s:s[2]):
                ax.add_patch(Circle((x*1000,y*1000),r*1000,facecolor='#f5a33c',edgecolor='#b37627',linewidth=.18,alpha=.8))
        points=np.concatenate([v@camera for n in names for v in parts[n]])[:,:2]*1000
        lo,hi=points.min(0)-10,points.max(0)+10
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal')
        ax.set_xlabel('Projected horizontal (mm)');ax.set_ylabel('Projected vertical (mm)');ax.grid(alpha=.15)

    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for row,names in enumerate([list(parts),['right_link26','right_link27','right_link28']]):
        for col,overlay in enumerate([False,True]):
            draw(axes[row,col],names,overlay)
            axes[row,col].set_title(('Original physical convex geometry' if not overlay else 'Active 871-sphere collision model')+(' — wrist/fingers' if row else ' — right arm'))
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/'spheres.jpg',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained');draw(ax,['right_link26','right_link27','right_link28'],False)
    eef=frames['right_eef_link'];t0=eef[:3,3];tcp=t0+.043*eef[:3,2]
    start=(t0@camera)[:2]*1000;end=(tcp@camera)[:2]*1000
    ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#1762ac','lw':2})
    ax.scatter(*end,c='#e14532',s=65,zorder=10)
    ax.annotate('Action TCP: +43 mm\nTip-plane axial gap: 1.97 mm',xy=end,xytext=(20,-42),textcoords='offset points',fontsize=11,arrowprops={'arrowstyle':'-','color':'#e14532'})
    ax.set_title('Current action TCP43 and original gripper geometry')
    fig.savefig(OUT/'tcp.jpg',dpi=130);plt.close(fig)
    print('Current geometry figures written.')


if __name__=='__main__':main()
