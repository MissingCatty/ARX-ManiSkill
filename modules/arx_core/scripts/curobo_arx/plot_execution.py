"""Plot recorded ARX end-effector paths and the actual test obstacle."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

out=Path(__file__).resolve().parents[2]/'runs/arx_curobo'
report=json.loads((out/'plan_report.json').read_text())
fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(projection='3d')
for label,color in [('baseline','crimson'),('planned','royalblue')]:
    records=json.loads((out/f'{label}_execution.json').read_text())['records']
    xyz=np.array([r['ee_world'] for r in records])[:,:3,3]
    ax.plot(*xyz.T,label=f'ARX measured {label}',color=color,linewidth=2)
    ax.scatter(*xyz[0],color=color,marker='o')
    ax.scatter(*xyz[-1],color=color,marker='x')
center=np.array(report['obstacle_center_world']);half=np.array(report['obstacle_dims'])/2
corners=np.array([center+half*[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
faces=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
ax.add_collection3d(Poly3DCollection(corners[faces],alpha=.3,facecolor='orange',edgecolor='darkorange'))
ax.set(xlabel='World X (m)',ylabel='World Y (m)',zlabel='World Z (m)',title='ARX Isaac Sim: measured obstacle-avoidance test\ncuRobo PRM; planned execution: zero external contacts')
ax.legend();ax.set_box_aspect((1,1,1));fig.tight_layout()
fig.savefig(out/'execution_paths.png',dpi=180)
print(out/'execution_paths.png')
