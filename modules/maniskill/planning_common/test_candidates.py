"""Offline candidate contract tests; no model, simulator or API requests."""
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch,MagicMock

import numpy as np

from common import ROOT,read,save,sha,ordinary_planner_rejection
from candidates import STAGE,rows,digest_rows,finalize_postprocessing,export_candidates,load_candidates


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.raw=self.root/'network/raw_outputs';self.raw.mkdir(parents=True)
        self.original=np.zeros((8,17),dtype=np.float64)
        self.original[:,0]=np.linspace(.9,.2,8);self.original[:,1:4]=[.07,.02,.03]
        self.original[:,4:13]=np.eye(3).reshape(1,9);self.original[:,15]=.4;self.original[:,16]=-1
        np.save(self.raw/'rgb_0.raw_grasps.npy',self.original)

    def finalize(self,accepted,final=None):
        if final is None:final=accepted
        np.save(self.raw/'rgb_0.grasp.npy',final)
        return finalize_postprocessing(self.raw,[dict(raw_rows_sha256=digest_rows(self.original),accepted_rows=accepted)])

    def test_uses_postprocessed_subset_and_preserves_refined_values(self):
        accepted=self.original[[1,4,7]].copy();accepted[:,1]=[.061,.062,.063];accepted[:,3]=.024
        final=accepted[[0,2]].copy();self.finalize(accepted,final)
        metadata=export_candidates(self.root/'network',self.root)
        actual,receipt=load_candidates(self.root)
        np.testing.assert_array_equal(actual,final)
        self.assertEqual((metadata['raw_count'],metadata['collision_pass_count'],metadata['count']),(8,3,2))
        self.assertEqual(receipt['candidate_stage'],STAGE)
        self.assertTrue(all(p.endswith('.grasp.npy') for p in metadata['source_files']))
        self.assertFalse((self.root/'raw_grasps.npy').exists())

    def test_all_collision_rejections_stay_empty(self):
        # Reproduce the native demo's fallback: its output still contains raw rows.
        receipt=self.finalize(np.empty((0,17)),self.original[:3])
        metadata=export_candidates(self.root/'network',self.root);actual,_=load_candidates(self.root)
        self.assertEqual(actual.shape,(0,17));self.assertEqual(metadata['count'],0)
        self.assertTrue(receipt['objects'][0]['all_rejected_fallback_removed'])
        self.assertEqual(rows(self.raw/'rgb_0.raw_grasps.npy').shape,(8,17))

    def test_missing_processed_file_does_not_fall_back_to_raw(self):
        with self.assertRaisesRegex(RuntimeError,'Missing native postprocessed file'):
            finalize_postprocessing(self.raw,[dict(raw_rows_sha256=digest_rows(self.original),accepted_rows=self.original)])

    def test_rejected_row_in_processed_file_is_record_fault(self):
        with self.assertRaisesRegex(RuntimeError,'lacks collision-pass evidence'):
            self.finalize(self.original[[1,2]],self.original[[0]])

    def test_missing_collision_audit_does_not_import_existing_raw_or_processed(self):
        np.save(self.raw/'rgb_0.grasp.npy',self.original[:1])
        with self.assertRaises(FileNotFoundError):export_candidates(self.root/'network',self.root)
        self.assertFalse((self.root/'postprocessed_grasps.npy').exists())

    def test_zero_reconstructed_objects_produce_empty_candidate_set(self):
        (self.raw/'rgb_0.raw_grasps.npy').unlink();finalize_postprocessing(self.raw,[])
        export_candidates(self.root/'network',self.root);actual,_=load_candidates(self.root)
        self.assertEqual(actual.shape,(0,17))

    def test_tampered_export_is_rejected_before_planning(self):
        self.finalize(self.original[:1]);export_candidates(self.root/'network',self.root)
        np.save(self.root/'postprocessed_grasps.npy',self.original)
        with self.assertRaisesRegex(RuntimeError,'planner input changed'):load_candidates(self.root)

    def test_tampered_native_output_is_rejected_before_planning(self):
        self.finalize(self.original[:1]);export_candidates(self.root/'network',self.root)
        np.save(self.raw/'rgb_0.grasp.npy',self.original)
        with self.assertRaisesRegex(RuntimeError,'source changed'):load_candidates(self.root)

    def test_historical_export_cannot_be_used_as_fresh_inference(self):
        self.finalize(self.original[:1]);export_candidates(self.root/'network',self.root,fresh_inference=False)
        with self.assertRaisesRegex(RuntimeError,'requires fresh inference'):load_candidates(self.root)

    def test_candidate_array_must_be_finite_and_17_columns(self):
        path=self.raw/'bad.grasp.npy';np.save(path,np.zeros((1,16)))
        with self.assertRaises(ValueError):rows(path)
        invalid=self.original[:1].copy();invalid[0,0]=np.nan;np.save(path,invalid)
        with self.assertRaises(ValueError):rows(path)







if __name__=='__main__':unittest.main(verbosity=2)
