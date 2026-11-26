"""CPU worker lifecycle, original endpoint ordering and frame conversion."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .client import GeometricClient, GeometricSequence
from .kinematics import ArmChain


class CpuWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = GeometricClient(Path(self.temp.name))
        self.addCleanup(self.client.close)

    def test_persistent_worker_solves_real_targets(self):
        chain = ArmChain(); start = chain.limits.mean(axis=1)
        target = chain.fk(start)[0]
        first = self.client.solve([target.tolist()]*16, start.tolist())
        second = self.client.solve([target.tolist()], start.tolist())
        self.assertEqual(first['id'], 1); self.assertEqual(second['id'], 2)
        self.assertEqual(len(first['answers']), 16)
        self.assertTrue(all(a['admitted_branches'] > 0 for a in first['answers']))

    def test_worker_exit_is_a_fault_without_relaunch(self):
        pid = self.client.process.pid
        self.client.process.terminate(); self.client.process.wait(timeout=5)
        with self.assertRaisesRegex(RuntimeError, 'worker exited'):
            self.client.solve([np.eye(4).tolist()], [0.]*6)
        self.assertEqual(self.client.process.pid, pid)

    def test_missing_response_times_out(self):
        self.client.timeout = .02
        with self.assertRaises(TimeoutError): self.client.read()

    def test_mismatched_response_does_not_enter_the_planner(self):
        with patch.object(self.client, 'read', return_value={'id': -1, 'answers': []}):
            with self.assertRaisesRegex(RuntimeError, 'mismatched'):
                self.client.solve([np.eye(4).tolist()], [0.]*6)


class SequenceTests(unittest.TestCase):
    def setUp(self):
        pose = np.eye(4); pose[:3, 3] = [.4, -.1, .8]
        self.candidates = [{'pose': pose.tolist(), 'score': 1-i*.01, 'index': i} for i in range(19)]
        self.before = deepcopy(self.candidates)
        self.calls = []
        def solve(targets, start):
            self.calls.append(deepcopy(targets))
            return {'answers': [{'source': len(self.calls)*100+i, 'native_solver_seconds': .00001}
                                for i in range(len(targets))], 'worker_seconds': .001, 'roundtrip_seconds': .002}
        self.context = SimpleNamespace(invbase=np.eye(4))
        self.context.invbase[:3, 3] = [-1, 2, -.5]
        self.sequence = GeometricSequence(self.candidates, self.context, {}, SimpleNamespace(solve=solve))
        array = np.zeros(6)
        value = SimpleNamespace(numpy=lambda: array)
        self.start = SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: value))

    def test_repeated_poses_preserve_each_source_and_tail(self):
        output = [self.sequence.next(self.sequence.target(c), self.start) for c in self.candidates]
        self.assertEqual([r['source'] for r in output], list(range(100, 116))+[200, 201, 202])
        self.assertEqual(list(map(len, self.calls)), [16, 3])
        self.assertEqual(self.candidates, self.before)
        self.assertEqual(self.sequence.summary()['prefetched_unconsumed'], 0)

    def test_TCP43_and_base_transform_match_original_endpoint(self):
        target = self.sequence.target(self.candidates[0])
        self.sequence.next(target, self.start)
        np.testing.assert_allclose(np.asarray(self.calls[0][0]), self.context.invbase@target,
            rtol=0, atol=1e-12)
        self.assertAlmostEqual(target[2, 3], .8-.043)

    def test_changed_order_or_pose_is_a_fault(self):
        wrong = self.sequence.target(self.candidates[0]); wrong[0, 3] += .01
        with self.assertRaisesRegex(RuntimeError, 'original endpoint'):
            self.sequence.next(wrong, self.start)
        self.assertFalse(self.calls)

    def test_prefetched_tail_cannot_change_consumed_count(self):
        self.sequence.next(self.sequence.target(self.candidates[0]), self.start)
        self.assertEqual(self.sequence.summary()['consumed_candidates'], 1)
        self.assertEqual(self.sequence.summary()['computed_candidates'], 16)
        self.assertEqual(self.sequence.summary()['prefetched_unconsumed'], 15)


if __name__ == '__main__': unittest.main()
