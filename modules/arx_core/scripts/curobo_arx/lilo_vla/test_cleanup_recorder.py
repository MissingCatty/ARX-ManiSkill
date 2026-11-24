"""Offline stop-ack/telemetry race tests; no simulator or GPU access."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts/curobo_arx/lilo_vla'), str(ROOT/'scripts/curobo_arx')]
from batch_runtime import Runtime, confirm_idle, UnsafeInstance
from gpu_policy import COMPUTE_GPU


class RecorderStopTests(unittest.TestCase):
    def setUp(self):
        self.clock=0.;self.wall_clock=1000.
        for target,clock in (('batch_runtime.time.monotonic',lambda:self.clock),
                             ('batch_runtime.time.time',lambda:self.wall_clock)):
            mocked=patch(target,side_effect=clock);mocked.start();self.addCleanup(mocked.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.box = self.root/'mailbox'; self.box.mkdir()
        self.trial = self.root/'trial'; self.trial.mkdir()
        self.fixture = dict(mailbox=str(self.box), gpu_uuid=COMPUTE_GPU,
                            renderer='RaytracedLighting', input_sha256={'setup':'fixed'})
        self.value = dict(version='arx_collection_status_v1', session_id='test',
            wall_time=time.time(), gpu_uuid=COMPUTE_GPU, renderer='RaytracedLighting',
            input_sha256=self.fixture['input_sha256'], active_trajectory_request_id=None,
            controller_busy=False, setup_pending=None, streaming=False,
            recorders={'teacher':None, 'video':str(self.trial/'frames')})
        self.stop_ack = False
        self.stop_frames = 0
        self.calls = []
        self.save()

    def save(self):
        (self.box/'stream_state.json').write_text(json.dumps({'collection_status':self.value}))

    def progress(self, delay):
        self.clock+=delay;self.wall_clock+=delay
        self.value['wall_time']=self.wall_clock
        if self.stop_ack:
            self.stop_frames += 1
            if self.stop_frames >= 2:
                self.value['recorders']['video'] = None
        self.save()

    def runtime(self):
        instance = Runtime({'fixture':self.fixture}, 'test')
        instance.trial = self.trial
        instance.heartbeat = lambda: None
        return instance

    def client(self, *args, **kwargs):
        owner = self
        class FakeClient:
            def request(self, action, **kwargs):
                owner.calls.append(action)
                if action != 'grasp_video_stop':
                    raise AssertionError('Unexpected command '+action)
                owner.stop_ack = True
                return {'frames':561, 'error':None}
        return FakeClient()

    def test_stop_ack_before_status_refresh_waits_and_saves_final_boundary(self):
        with patch('batch_runtime.Client', side_effect=self.client), patch('batch_runtime.time.sleep', side_effect=self.progress):
            self.runtime().cleanup(time.monotonic()+1)
        self.assertEqual(self.calls, ['grasp_video_stop'])
        self.assertGreaterEqual(self.stop_frames, 3)
        final = json.loads((self.trial/'boundary_status.json').read_text())
        self.assertEqual(final['recorders'], {'teacher':None, 'video':None})

    def test_own_recorder_that_never_stops_cannot_pass(self):
        def progress(delay):
            self.clock+=delay;self.wall_clock+=delay
            self.value['wall_time']=self.wall_clock
            self.save()
        with patch('batch_runtime.Client', side_effect=self.client), patch('batch_runtime.time.sleep', side_effect=progress):
            with self.assertRaises(UnsafeInstance):
                self.runtime().cleanup(time.monotonic()+1)
        self.assertEqual(self.calls, ['grasp_video_stop'])
        self.assertFalse((self.trial/'boundary_status.json').exists())

    def test_foreign_recorder_is_rejected_and_never_stopped(self):
        self.value['recorders']['video'] = '/another/user/frames'; self.save()
        with patch('batch_runtime.Client', side_effect=self.client):
            with self.assertRaisesRegex(UnsafeInstance, 'Unowned'):
                self.runtime().cleanup(time.monotonic()+1)
        self.assertEqual(self.calls, [])

    def test_two_distinct_cleared_statuses_are_required(self):
        active = copy.deepcopy(self.value)
        first = copy.deepcopy(active); first['wall_time'] += .001; first['recorders']['video'] = None
        second = copy.deepcopy(first); second['wall_time'] += .001
        with patch('batch_runtime.status', side_effect=[active, first, first, second]), patch('batch_runtime.time.sleep'):
            result = confirm_idle(self.box, self.fixture, 'test', time.monotonic()+1,
                self.trial, allow_owned_recorders=True, require_recorders_stopped=True)
        self.assertEqual(result['wall_time'], second['wall_time'])

    def test_foreign_recorder_during_stop_wait_still_fails(self):
        foreign = copy.deepcopy(self.value); foreign['recorders']['video'] = '/another/user/frames'
        with patch('batch_runtime.status', side_effect=[self.value, foreign]), patch('batch_runtime.time.sleep'):
            with self.assertRaisesRegex(UnsafeInstance, 'Unowned'):
                confirm_idle(self.box, self.fixture, 'test', time.monotonic()+1,
                    self.trial, allow_owned_recorders=True, require_recorders_stopped=True)


if __name__ == '__main__':
    unittest.main()
