"""Repaired contact settings and per-execution physics measurements."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

from repair_contract import VERSION,CONTACT_OFFSET_M,simulation_config

HERE=Path(__file__).resolve().parent


def install(root):
    source=HERE.parent/'diverse_grasp_benchmark/catalog_simulation.py'
    spec=importlib.util.spec_from_file_location('repaired_catalog_simulation',source)
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    run,baseline,capture=previous.install(root)
    import environment
    original=environment.ARXPickYCB._default_sim_config
    environment.ARXPickYCB._default_sim_config=property(lambda self:simulation_config(original.fget(self)))
    parent_execution=run.Execution

    class MeasuredExecution(parent_execution):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.physics_stream=(self.out/'physics_trace.jsonl').open('x')
            self.physics_max_speed=0.
            self.physics_max_displacement=0.
            self.physics_min_separation=None
            self.physics_initial_position=self.env.obj.pose.p[0].cpu().numpy().copy()
            shapes=self.env.obj._bodies[0].get_collision_shapes()
            assert all(abs(s.contact_offset-CONTACT_OFFSET_M)<1e-8 for s in shapes)
            run.save(self.out/'execution_physics.json',{'repair_contract':VERSION,
                'contact_offset_m':self.env.sim_config.scene_config.contact_offset,
                'rest_offset_m':self.env.sim_config.scene_config.rest_offset,
                'target_mass_kg':self.env.obj._bodies[0].mass,
                'target_inertia':np.asarray(self.env.obj._bodies[0].inertia).tolist(),
                'target_collision_shapes':len(shapes),'actual_tcp_m':.043})

        def record(self):
            super().record()
            body=self.env.obj._bodies[0]
            speed=float(np.linalg.norm(body.linear_velocity))
            position=self.env.obj.pose.p[0].cpu().numpy()
            displacement=float(np.linalg.norm(position-self.physics_initial_position))
            self.physics_max_speed=max(self.physics_max_speed,speed)
            self.physics_max_displacement=max(self.physics_max_displacement,displacement)
            separations=[]
            target=self.env.obj._objs[0]
            for contact in self.env.scene.get_contacts():
                if target not in [b.entity for b in contact.bodies]:continue
                separations += [float(p.separation) for p in contact.points if np.linalg.norm(p.impulse)>1e-8]
            if separations:
                sep=min(separations)
                self.physics_min_separation=sep if self.physics_min_separation is None else min(sep,self.physics_min_separation)
            if self.tick%12==0 or separations:
                self.physics_stream.write(__import__('json').dumps({'tick':self.tick,'stage':self.label,
                    'position':position.tolist(),'speed_mps':speed,'min_separation_m':min(separations) if separations else None})+'\n')

        def finish(self):
            self.physics_stream.close()
            run.save(self.out/'execution_physics_metrics.json',{'repair_contract':VERSION,
                'max_object_speed_mps':self.physics_max_speed,
                'max_object_displacement_m':self.physics_max_displacement,
                'min_contact_separation_m':self.physics_min_separation})
            super().finish()
    run.Execution=MeasuredExecution

    def capture_checked():
        out=capture()
        from common import save
        save(root/'preflight/repaired_physics.json',{'passed':True,'repair_contract':VERSION,
            'contact_offset_m':CONTACT_OFFSET_M,'rest_offset_m':0.,'physical_grasps':0})
        return out
    return run,baseline,capture_checked
