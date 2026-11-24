"""Tapered static ARX observations with original cooked convex target pieces."""
import argparse
import copy
import json
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
from scripts.curobo_arx.data_factory.common import read,write,sha,verify_files,space
from scripts.curobo_arx.data_factory.runtime import Group as BaseGroup
from scripts.curobo_arx.data_factory.frozen import fingerprint
from .advisor38_static_capture import StaticCapture
from .advisor36_samples import require_release
from .advisor36_contract import residual,token
from .capture import SynchronizedCapture,validate_saved_capture
from .verify_validation import reset_signature
from .visual_grounding_spec import expected_reset
from training.nearfield_arx.contract import pose_error


class Group(BaseGroup):
    def check(self):
        if self.stop or (self.campaign/'STOP').exists() or time.time()>=self.launch['stop_unix']:
            raise RuntimeError('Static observation campaign stopped/deadline')
        if any(e.bridge.active or e.bridge.fault for e in self.envs):
            raise RuntimeError('Static capture requires confirmed idle healthy ARX')
        space(self.out); self.writer.check()

    def wait_for(self,job):
        start=time.monotonic()
        while True:
            self.check(); reply=self.planner.poll(job)
            if reply is not None:
                if not reply['ok']: raise RuntimeError(reply['error'])
                return reply['result']
            if time.monotonic()-start>900: raise TimeoutError('Static IK resource timeout')
            # No physics is advanced while waiting for this independent state.
            time.sleep(.05)

    def run(self):
        env=self.envs[0]; b=env.bridge; results=[]
        static=self.registration['static_observations']
        registry=read(static['canonical_registry'])
        table={r['sample_id']:r for r in (json.loads(line) for line in Path(static['candidate_table']).read_text().splitlines())}
        self.ledger.close()
        try:
            initial_capture=SynchronizedCapture(b,render_every_update=True)
            capture=StaticCapture(b,self.registration['model_reference'])
            for obj in dict.fromkeys(table[i]['object_id'] for i in self.launch['sample_ids']):
                self.check(); entry=registry['objects'][obj]
                require_release(entry['seed_results'],obj)
                verify_files(entry['physical_evidence'])
                source=entry['source']; asset=copy.deepcopy(source['asset'])
                asset.update(placement=source['placement']['placement'],collection_placement=source['placement'])
                folder=self.out/'objects'/obj;folder.mkdir(parents=True,exist_ok=False)
                env.reset(asset)
                expected=expected_reset(source['seed'],{'object_id':obj,'placement':source['placement'],
                                                       'target_delta_world_m':[0.,0.,0.]})
                reset=reset_signature(env.reset_initial)
                write(folder/'reset_verification.json',{'pass':reset==expected,'actual':reset,'expected':expected})
                if reset!=expected:raise RuntimeError('Historical object/robot initial reset changed')
                # Same initial3second settle. This is the only physical warmup
                # before independent, explicitly static joint-state observations.
                for _ in range(15): self.step(False)
                snap=b.snapshot();initial_capture.capture(snap,snap)
                frozen=fingerprint(b,read(self.registration['reference_files'][self.renderer]))
                write(folder/'frozen_before.json',frozen)
                if not frozen['pass']:raise RuntimeError('Static environment configuration mismatch')
                # Keep historical non-arm joints, and use the actually successful
                # open-finger state of the canonical center preflight.
                center=read(entry['center_closure_request'])['measured_hold_snapshot']
                capture.bind_mount(center)
                write(folder/'static_render_binding.json',capture.render_binding)
                template=np.array(env.reset_initial['joint_positions'])
                for name in ('right_joint27','right_joint28'):
                    template[snap['joint_names'].index(name)]=center['joint_positions'][center['joint_names'].index(name)]
                arm_seed=[center['joint_positions'][center['joint_names'].index(n)] for n in snap['right_arm']]
                # Establish the constant open state once as initialization.
                # A synchronization fault here is not a started observation.
                capture.set_joint_state(obj+'_open_seed',snap['joint_names'],template)
                b.context.cameras['right_wrist'].sync_pose(force=True)
                # The historical non-arm joint reset above changes measured
                # body4. Export only after that state has been established.
                scene=folder/'scene.json';write(scene,b.export_scene())
                from .advisor37_surfaces import export_geometry
                surfaces=folder/'static_collision_surfaces.json';write(surfaces,export_geometry(b))
                for ident in (i for i in self.launch['sample_ids'] if table[i]['object_id']==obj):
                    self.check(); candidate=table[ident]
                    out=self.out/'observations'/ident;out.mkdir(parents=True,exist_ok=False)
                    started=time.time(); write(out/'STARTED.json',{'sample_id':ident,'wall':started,'candidate':candidate})
                    # A consistent open gripper is part of every proposed static
                    # state, including the planner's measured model update.
                    latest=b.snapshot()
                    request={'sample_id':ident,'snapshot':latest,'scene':str(scene),'output':str(out),
                        'desired_eef_object':candidate['desired_eef_object'],'category':candidate['category'],
                        'canonical_arm_seed':arm_seed,'visual_context':env.visual_context()}
                    write(out/'IK_request.json',request)
                    ik=self.wait_for(self.planner.submit('static_ik',**request))
                    row={'sample_id':ident,'pair_id':candidate['pair_id'],'object_id':obj,
                        'split':candidate['split'],'category':candidate['category'],'candidate':candidate,
                        'started_unix':started,'physical_trajectory_executed':False,'record_integrity_pass':True}
                    if not ik['pass']:
                        row.update(pass_=False,rejected=ik['rejected'],IK=ik)
                    else:
                        q=template.copy()
                        for name,value in zip(ik['joint_names'],ik['arm_q']):q[snap['joint_names'].index(name)]=value
                        setter=capture.set_joint_state(ident,snap['joint_names'],q)
                        write(out/'static_setter.json',setter)
                        surface_request={'sample_id':ident,'snapshot':setter['after'],
                            'geometry':str(surfaces),'output':str(out)}
                        write(out/'surface_request.json',surface_request)
                        surface=self.wait_for(self.planner.submit('static_surface_check',**surface_request))
                        bundle=capture.capture_static(setter)
                        actual=b.snapshot();target=np.asarray(actual['objects']['test_object']['world'])@np.asarray(candidate['canonical_object'])
                        eef=actual['link_world']['right_eef_link']; desired=np.asarray(actual['objects']['test_object']['world'])@np.asarray(candidate['desired_eef_object'])
                        fit=pose_error(eef,desired); dp,dr=residual(eef,target)
                        core=bool(np.linalg.norm(dp)<=.004 and np.linalg.norm(dr)<=np.deg2rad(3))
                        good=bool(surface['pass'] and fit['translation_m']<=.0005 and fit['rotation_rad']<=np.deg2rad(.2) and core==(candidate['category']=='grasp'))
                        observation,rgb,depth=b.observation(self.owner+'/env_0',ident,obj,None,stage='independent_static_observation')
                        raw=out/'raw';raw.mkdir();Image.fromarray(rgb).save(raw/'000000.png');np.save(raw/'000000.npy',depth)
                        observation.update(rgb_path=str(raw/'000000.png'),depth_path=str(raw/'000000.npy'),
                            rgb_relative='raw/000000.png',depth_relative='raw/000000.npy',
                            capture_files={k:f'raw/000000_{k}.npy' for k in bundle['arrays']})
                        for k,a in bundle['arrays'].items():np.save(out/observation['capture_files'][k],a)
                        write(out/'observation.json',observation);validate_saved_capture(out,observation)
                        from scripts.curobo_arx.data_factory.visual_geometry import inspect_row
                        visual=inspect_row(env.visual_geometry,observation,depth.shape)
                        good=bool(good and visual['pass'])
                        row.update(pass_=good,rejected=None if good else ('actual_static_target_intersection' if not surface['pass'] else 'actual_static_FK_core_or_camera_geometry_rejected'),
                            target_surface_check=surface,
                            actual_FK_fit=fit,actual_translation_m=dp.tolist(),actual_rotation_degrees=np.rad2deg(dr).tolist(),
                            actual_eef_world=eef,actual_canonical_world=target.tolist(),actual_core=core,
                            output=token(eef,target,grasp=core),visual_geometry=visual,
                            observation=str(out/'observation.json'),static_setter=str(out/'static_setter.json'),
                            files={str(f):sha(f) for f in out.rglob('*') if f.is_file()})
                    row['finished_unix']=time.time();write(out/'capture_receipt.json',row);results.append(row)
                    with (self.out/'results.jsonl').open('a') as stream:
                        stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush()
                    write(self.out/'state.json',{'status':'capturing','processed':len(results),
                        'scheduled':len(self.launch['sample_ids']),'capture_pass':sum(r['pass_'] for r in results),
                        'last_sample':ident,'wall':time.time()})
                    print('ADVISOR38_STATIC',ident,row['pass_'],row['rejected'],flush=True)
                check=fingerprint(b,read(self.registration['reference_files'][self.renderer]))
                write(folder/'frozen_after.json',check)
                if not check['pass']:raise RuntimeError('Frozen configuration changed during static acquisition')
            write(self.out/'results.json',{'observations':results,'physical_grasp_attempts':0})
            write(self.out/'completion.json',{'pass':True,'registered':len(self.launch['sample_ids']),
                'processed':len(results),'capture_pass':sum(r['pass_'] for r in results),
                'observations_not_physical_trajectories':True,'training_started':False})
        except BaseException as error:
            write(self.out/'FAULT.json',{'error':traceback.format_exc(),'processed':len(results),
                'static_diagnostics':getattr(error,'static_diagnostics',None)});raise
        finally:
            self.writer.close();self.diagnostics.close()
            write(self.out/'shutdown_receipt.json',{'all_trajectories_idle':not b.active,
                'raw_writer_drained_and_closed':True,'wall':time.time()})
            self.app.close(wait_for_replicator=False,skip_cleanup=True,exit_code=1 if (self.out/'FAULT.json').exists() else 0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--launch',required=True)
    a=p.parse_args();Group(a.campaign,a.launch).run()
