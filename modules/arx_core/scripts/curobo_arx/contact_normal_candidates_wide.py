"""UNREGISTERED wider CPU proposal family (v1 zero candidates retained); no IK/checker/simulator queries."""
import copy,time
import numpy as np
from scipy.spatial.transform import Rotation
from contact_normal_quality import ObservedContactQuality
from long_axis_misalignment import proposals
from sphere_refinement import PAD_CENTER_EEF


def candidates(folder,selected,shape,max_seconds=90.):
    started=time.monotonic();quality=ObservedContactQuality(folder);axis=np.asarray(shape['horizontal_axis_world'])
    rows=[];examined=0;normal_pass=0;seen=set()
    for world,pre,lift,change in proposals(selected,shape):
        if change['world_z_pose_search_offset_m']!=0.:continue
        for axial in (0.,-.020,.020,-.040,.040):
            for roll in (0.,-15.,15.,-30.,30.,-45.,45.,-60.,60.,-75.,75.,-90.,90.):
              for height in (0.,-.003,-.006,.003,.006,.009,.012):
                  if time.monotonic()-started>=max_seconds:
                      return sorted(rows,key=lambda row:row['rank'],reverse=True),{'examined':examined,'normal_pass':normal_pass,'quality_pass':len(rows),'timeout':True,'elapsed_s':time.monotonic()-started}
                  examined+=1;goal=world.copy();goal[:3,:3]=world[:3,:3]@Rotation.from_euler('z',roll,degrees=True).as_matrix()
                  pad=np.asarray(change['pad_target_world'])+axial*axis+np.array([0.,0.,height]);goal[:3,3]=pad-goal[:3,:3]@PAD_CENTER_EEF
                  signature=tuple(np.round(goal.flatten(),10))
                  if signature in seen:continue
                  seen.add(signature);evidence=quality.evaluate_normals(goal)
                  if not evidence['pass']:continue
                  normal_pass+=1;palm=quality.palm.evaluate(goal)
                  if not palm['pass']:continue
                  updated=copy.deepcopy(change);updated.update(hypothesis='UNREGISTERED_contact_normals',axial_contact_target_shift_m=axial,closing_roll_about_approach_deg=roll,world_z_pose_search_offset_m=height,pad_target_world=pad.tolist(),contact_normal_quality=evidence,palm_target_quality=palm)
                  pregoal=goal.copy();pregoal[2,3]+=.08
                  # Preference is mechanical and fixed: strong lower-quartile normal alignment,
                  # then median, observed support count; displacement used only as tie breaker.
                  rank=[evidence['q25_abs_surface_normal_dot_closing'],evidence['median_abs_surface_normal_dot_closing'],evidence['visible_tip_ROI_points'],-abs(axial),-abs(roll)]
                  rows.append({'world':goal.tolist(),'pre':pregoal.tolist(),'change':updated,'rank':rank})
    return sorted(rows,key=lambda row:row['rank'],reverse=True),{'examined':examined,'unique':len(seen),'normal_pass':normal_pass,'quality_pass':len(rows),'timeout':False,'elapsed_s':time.monotonic()-started}
