import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .capture import AXES, masks_from_ids, reference_seconds, validate_bundle
from scripts.curobo_arx.data_factory.recording import Recording, Writer


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.env='/World/Envs/env_0'
        self.roles={'target':self.env+'/Task/bottle','right_link27':self.env+'/Robot/right_link27',
                    'right_link28':self.env+'/Robot/right_link28'}
        self.ids=np.array([[0,2,3],[0,2,3]],dtype=np.uint32)
        self.mapping={'0':'BACKGROUND','2':self.roles['target']+'/mesh','3':self.roles['right_link27']+'/mesh'}
        self.rgb=np.ones((2,3,3),dtype=np.uint8);self.depth=np.full((2,3),.1,dtype=np.float32)
        self.ref={'referenceTimeNumerator':2,'referenceTimeDenominator':10}
        self.params={'cameraViewTransform':np.eye(4).reshape(-1),'renderProductResolution':[3,2]}

    def bundle(self, **kwargs):
        args=dict(rgb=self.rgb,depth=self.depth,ids=self.ids,info={'idToLabels':self.mapping},
            reference_before=self.ref,reference_after=self.ref,sim_time=.2,
            camera_params=self.params,expected_optical=AXES,roles=self.roles,environment=self.env)
        args.update(kwargs);return validate_bundle(**args)

    def test_static_images_and_occluded_finger_are_valid(self):
        a=self.bundle();b=self.bundle()
        self.assertTrue(np.array_equal(a['rgb'],b['rgb']))
        self.assertEqual(int(a['arrays']['right_link28'].sum()),0)

    def test_buffers_are_copied(self):
        b=self.bundle();self.ids[:]=0;self.depth[:]=0
        self.assertEqual(int(b['arrays']['target'].sum()),2)
        self.assertGreater(b['depth'].min(),0)

    def test_array_and_metadata_wrapped_rgb_have_identical_owning_pixels(self):
        from .capture import array_data
        a=array_data(self.rgb);b=array_data({'data':self.rgb,'info':{}})
        self.assertTrue(np.array_equal(a,b))
        self.rgb[:]=0
        self.assertTrue(a.any());self.assertTrue(b.any())

    def test_unready_capture_only_allowed_within_original_warmup(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from .runtime_v2 import ActionGroup
        from .capture import CaptureUnavailable
        with tempfile.TemporaryDirectory() as tmp:
            g=ActionGroup.__new__(ActionGroup);g.out=Path(tmp)
            g.world=SimpleNamespace(current_time=1.)
            env=SimpleNamespace(bridge=SimpleNamespace(snapshot=lambda:{},active=None),
                recorder=None,state='settling',settle_until=3.)
            g.envs=[env]
            def missing(*args):raise CaptureUnavailable('empty', [{'error':'empty'}])
            g.captures=[SimpleNamespace(capture=missing)]
            with patch('scripts.curobo_arx.nearfield.runtime.LocalGroup.step'):
                g.step(False)
                g.step(True)  # A claimed task resets again before any motion.
                g.world.current_time=3.
                with self.assertRaises(CaptureUnavailable):g.step(False)
                g.world.current_time=1.;env.state='executing'
                with self.assertRaises(CaptureUnavailable):g.step(True)
                env.state='settling';env.bridge.active={'id':'motion'}
                with self.assertRaises(CaptureUnavailable):g.step(False)

    def test_empty_or_invalid_channels_fail(self):
        for kw in ({'rgb':np.zeros_like(self.rgb)},{'depth':np.zeros_like(self.depth)},
                   {'depth':np.ones((1,2))},{'ids':self.ids.astype(float)}):
            with self.subTest(kw=list(kw)),self.assertRaises(ValueError):self.bundle(**kw)

    def test_stale_and_mixed_render_times_fail(self):
        with self.assertRaises(ValueError):self.bundle(sim_time=.4)
        with self.assertRaises(ValueError):self.bundle(reference_after={'referenceTimeNumerator':3,'referenceTimeDenominator':10})
        with self.assertRaises(ValueError):reference_seconds({})

    def test_old_camera_pose_fails(self):
        e=AXES.copy();e[0,3]=.002
        with self.assertRaises(ValueError):self.bundle(expected_optical=e)

    def test_missing_instance_map_fails(self):
        with self.assertRaises(ValueError):self.bundle(info={'idToLabels':{'0':'BACKGROUND'}})

    def test_other_environment_pixels_or_roles_fail(self):
        mapping=dict(self.mapping);mapping['3']=mapping['3'].replace('env_0','env_1')
        with self.assertRaises(ValueError):self.bundle(info={'idToLabels':mapping})
        roles=dict(self.roles);roles['target']=roles['target'].replace('env_0','env_1')
        with self.assertRaises(ValueError):self.bundle(roles=roles)

    def test_finger_identity_cannot_be_swapped(self):
        roles=dict(self.roles);roles['right_link27'],roles['right_link28']=roles['right_link28'],roles['right_link27']
        with self.assertRaises(ValueError):self.bundle(roles=roles)

    def test_force_sync_does_not_skip_small_real_pose_updates(self):
        import importlib.util
        from types import SimpleNamespace
        from scripts.curobo_arx.data_factory.common import ROOT
        spec=importlib.util.spec_from_file_location('arx_camera_test',ROOT/'arx-sim/arx_sim/sensors/camera.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        c=module.IsaacRgbdCamera.__new__(module.IsaacRgbdCamera)
        pos=np.array([[4.+.00002,2.,1.]]);rot=np.array([[1.,0.,0.,0.]])
        c._last_positions=np.array([[4.,2.,1.]]);c._last_orientations=rot.copy();calls=[]
        c._source_prim=SimpleNamespace(get_world_poses=lambda:(SimpleNamespace(numpy=lambda:pos),SimpleNamespace(numpy=lambda:rot)))
        c._sensor=SimpleNamespace(authoring_object=SimpleNamespace(set_world_poses=lambda **kw:calls.append(kw)))
        c.sync_pose();self.assertEqual(len(calls),0)
        c.sync_pose(force=True);self.assertEqual(len(calls),1)

    def test_capture_trigger_cannot_write_to_original_mounted_camera(self):
        from .capture import capture_trigger
        from types import SimpleNamespace
        from unittest.mock import Mock
        stage=Mock()
        with self.assertRaises(ValueError):
            capture_trigger(SimpleNamespace(prim_path=self.env+'/Robot/wrist_camera'),stage)
        stage.GetPrimAtPath.assert_not_called()
        rate=Mock();rate.Get.return_value=30.
        stage.GetPrimAtPath.return_value.GetAttribute.return_value=rate
        with self.assertRaises(RuntimeError):
            capture_trigger(SimpleNamespace(prim_path='/World/COOLASSensors/env_0/right_wrist'),stage)

    def test_complete_channels_are_written_without_aliasing(self):
        from scripts.curobo_arx.data_factory.common import read
        with tempfile.TemporaryDirectory() as tmp:
            writer=Writer();root=Path(tmp)/'episode';task={'env':'e','id':'a'}
            rec=Recording(root,task,writer)
            row={'frame_time':.2,'capture_sim_time':.2,'env_id':'e','attempt_id':'a',
                 'snapshot':{'link_world':{'right_eef_link':np.eye(4).tolist()}},'stage':'entry',
                 'capture':self.bundle()['metadata']}
            extras=self.bundle()['arrays'];rec.add(row,self.rgb,self.depth,extras=extras)
            extras['instance_ids'][:]=0;writer.close()
            import json
            saved=json.loads((root/'samples.jsonl').read_text())
            self.assertEqual(len(saved['capture_files']),5)
            self.assertEqual(int(np.load(root/saved['capture_files']['instance_ids']).max()),3)


if __name__=='__main__':unittest.main()
