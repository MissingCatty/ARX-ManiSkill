"""Candidate transport, storage faults and archived-path equivalence checks."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from arx_project import artifacts
from arx_project.artifacts import (ArtifactWriteError, BackgroundWriter, candidate_transport,
    commit_candidate_transport, finish_candidate_transport, publish_candidate_pool, read_candidate_transport)
from arx_project.json_io import relocate_paths
from arx_project.layout import ROOT, relocated, write_json
from modules.curobo.candidate_io.store import (candidate_pool, candidate_scope,
    bind_candidate_view, limit_candidate_view, read_snapshot)


def payload(count=5):
    return {'protocol': 'current_native_pool', 'candidates': [
        {'score': float(count-i), 'pose': [[1.,0.,0.,i*.01],[0.,1.,0.,0.],
            [0.,0.,1.,.03],[0.,0.,0.,1.]], 'metadata': {'origin': 'graspgenx'}} for i in range(count)]}


class IOTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'tmp/candidate_io'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='check_', dir=parent)
        self.root = Path(self.temp.name)

    def tearDown(self):
        artifacts.close_owned_transports()
        self.temp.cleanup()

    def test_current_metadata_does_not_enter_legacy_path_conversion(self):
        value = payload()
        with patch('arx_project.json_io.relocated', wraps=relocated) as convert:
            actual = relocate_paths(value, [('/legacy', '/current')])
        self.assertEqual(actual, value)
        convert.assert_not_called()

    def test_legacy_paths_keys_chains_and_boundaries_match_original(self):
        aliases = [('/legacy/deep', '/alternate'), ('/legacy', '/new'),
                   ('/alternate', '/new/inside'), ('short', '/legacy')]
        value = {'/legacy/file': ['/legacy/deep/a', '/legacy-other/a', 'label', 'short/x'],
                 'score': 1.25, 'other': False}
        self.assertEqual(relocate_paths(value, aliases), relocated(value, aliases))

    def test_legacy_cycle_still_fails(self):
        with self.assertRaises(ValueError):
            relocate_paths({'file':'/a/x'}, [('/a','/b'),('/b','/a')])

    def test_one_decode_and_shared_objects_across_arms(self):
        write_json(self.root/'candidates/candidate_pool.json', payload())
        with patch('modules.curobo.candidate_io.store.json.loads', wraps=json.loads) as loads:
            with candidate_scope(self.root):
                right, left = self.root/'arms/right', self.root/'arms/left'
                bind_candidate_view(right, self.root); bind_candidate_view(left, self.root)
                root_pool = candidate_pool(self.root)
                self.assertIs(candidate_pool(right), root_pool)
                self.assertIs(candidate_pool(left), root_pool)
                self.assertIs(candidate_pool(right)['candidates'][0], root_pool['candidates'][0])
                self.assertEqual(loads.call_count, 1)
        self.assertFalse((right/'candidates/candidate_pool.json').exists())
        self.assertTrue((right/'candidates/candidate_pool_reference.json').exists())
        self.assertEqual(read_snapshot(right)[0], payload())

    def test_rank_prefix_keeps_original_order_and_other_arm_full_pool(self):
        write_json(self.root/'candidates/candidate_pool.json', payload())
        with candidate_scope(self.root):
            right, left = self.root/'arms/right', self.root/'arms/left'
            bind_candidate_view(right, self.root); bind_candidate_view(left, self.root)
            limit_candidate_view(left, 2, {'winning_candidate_before_request': 1})
            self.assertEqual(len(candidate_pool(right)['candidates']), 5)
            self.assertEqual(candidate_pool(left)['candidates'], payload()['candidates'][:2])
            self.assertEqual(candidate_pool(self.root), payload())
        self.assertEqual(read_snapshot(left)[0]['candidates'], payload()['candidates'][:2])

    def test_invalid_prefix_and_unbound_view_fail(self):
        write_json(self.root/'candidates/candidate_pool.json', payload())
        with candidate_scope(self.root):
            with self.assertRaises(RuntimeError):candidate_pool(self.root/'wrong')
            with self.assertRaises(ValueError):limit_candidate_view(self.root, 99, {})

    def test_candidate_mutation_fails_and_context_is_released(self):
        write_json(self.root/'candidates/candidate_pool.json', payload())
        with self.assertRaises(RuntimeError):
            with candidate_scope(self.root):candidate_pool(self.root)['candidates'][0]['score'] = -100.
        with candidate_scope(self.root):
            self.assertEqual(candidate_pool(self.root)['candidates'][0]['score'], 5.)

    def test_background_write_error_propagates(self):
        writer = BackgroundWriter()
        with patch('arx_project.artifacts.atomic_json', side_effect=OSError('disk full')):
            writer.submit(self.root/'file.json', {'value':1})
            with self.assertRaises(ArtifactWriteError):writer.flush()
        writer.close()

    def test_shared_data_available_while_snapshot_writer_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        original = artifacts.atomic_json

        def blocking(path, value):
            entered.set()
            if not release.wait(5):raise RuntimeError('test writer release timeout')
            return original(path, value)

        descriptor = None
        try:
            with patch('arx_project.artifacts.atomic_json', side_effect=blocking):
                descriptor = publish_candidate_pool(self.root, payload())
                self.assertTrue(entered.wait(1))
                self.assertFalse((self.root/'candidates/candidate_pool.json').exists())
                self.assertEqual(read_candidate_transport(self.root, descriptor), payload())
                release.set()
                self.assertTrue(finish_candidate_transport(self.root)['persisted'])
        finally:
            release.set()
        self.assertIsNone(candidate_transport(self.root))
        self.assertEqual(json.loads((self.root/'candidates/candidate_pool.json').read_text()), payload())

    def test_published_snapshot_freezes_values_before_background_save(self):
        original = payload(); expected = deepcopy(original)
        descriptor = publish_candidate_pool(self.root, original)
        original['candidates'][0]['score'] = 999.
        self.assertEqual(read_candidate_transport(self.root, descriptor), expected)
        finish_candidate_transport(self.root)
        self.assertEqual(json.loads((self.root/'candidates/candidate_pool.json').read_text()), expected)

    def test_corrupt_transport_fails_without_file_fallback(self):
        descriptor = publish_candidate_pool(self.root, payload())
        bad = {**descriptor, 'sha256': '0'*64}
        with self.assertRaises(ValueError):read_candidate_transport(self.root, bad)
        with self.assertRaises(ValueError):read_candidate_transport(self.root/'other', descriptor)

    def test_wrong_commit_token_cannot_release_another_snapshot(self):
        descriptor = publish_candidate_pool(self.root, payload())
        with self.assertRaises(ValueError):commit_candidate_transport(self.root, '0'*32)
        self.assertEqual(candidate_transport(self.root), descriptor)

    def test_unconsumed_snapshots_have_a_bounded_count(self):
        with patch('arx_project.artifacts.MAX_PENDING_TRANSPORTS',1):
            publish_candidate_pool(self.root,payload())
            with self.assertRaises(RuntimeError):
                publish_candidate_pool(self.root/'another',payload())

    def test_shared_transport_reference_can_be_replayed_after_commit(self):
        descriptor = publish_candidate_pool(self.root, payload())
        with candidate_scope(self.root, descriptor):
            bind_candidate_view(self.root/'arms/right', self.root)
            limit_candidate_view(self.root/'arms/right', 3, {})
        finish_candidate_transport(self.root)
        loaded, _ = read_snapshot(self.root/'arms/right')
        self.assertEqual(loaded['candidates'], payload()['candidates'][:3])

    def test_reference_detects_a_changed_canonical_snapshot(self):
        write_json(self.root/'candidates/candidate_pool.json', payload())
        with candidate_scope(self.root):bind_candidate_view(self.root/'arms/right', self.root)
        write_json(self.root/'candidates/candidate_pool.json', payload(4))
        with self.assertRaises(ValueError):read_snapshot(self.root/'arms/right')

    def test_legacy_graspcl_snapshot_directory_remains_readable(self):
        write_json(self.root/'graspcl/candidate_pool.json',payload())
        with candidate_scope(self.root):
            self.assertEqual(candidate_pool(self.root),payload())

    def test_independent_process_reads_transport_without_unlinking_owner(self):
        descriptor = publish_candidate_pool(self.root, payload())
        code = ('import json,sys; from arx_project.artifacts import read_candidate_transport; '
                'v=read_candidate_transport(sys.argv[1],json.loads(sys.argv[2])); print(len(v["candidates"]))')
        result = subprocess.run([sys.executable, '-c', code, str(self.root), json.dumps(descriptor)],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '5')
        self.assertNotIn('resource_tracker', result.stderr)
        self.assertEqual(read_candidate_transport(self.root, descriptor), payload())


if __name__ == '__main__':
    unittest.main()
