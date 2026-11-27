import unittest

import numpy as np

from .profiles import allocate,enclose,merge_link,digest


class ProfileTests(unittest.TestCase):
    def test_allocation_preserves_links_and_total(self):
        counts={'arm':26,'wrist':89,'finger':258}
        for total in [3,32,128,373]:
            result=allocate(counts,total)
            self.assertEqual(sum(result.values()),total)
            self.assertTrue(all(1<=result[k]<=counts[k] for k in counts))

    def test_contained_sphere_does_not_expand_enclosure(self):
        a=np.array([0,0,0,.03]);b=np.array([.01,0,0,.005])
        np.testing.assert_array_equal(enclose(a,b),a)

    def test_merge_covers_every_original_ball_after_float32_rounding(self):
        source=[{'center':[i*.003,0,0],'radius':.002} for i in range(20)]
        result,proof=merge_link(source,5)
        self.assertEqual(len(result),5)
        self.assertGreater(proof['minimum_original_ball_cover_margin_m'],0.)
        self.assertEqual(sorted(i for group in proof['members'] for i in group),list(range(20)))
        for sphere,indices in zip(result,proof['members']):
            for i in indices:
                self.assertGreaterEqual(sphere['radius'],np.linalg.norm(np.asarray(sphere['center'])-source[i]['center'])+source[i]['radius'])

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):merge_link([{'center':[0,0,0],'radius':-1}],1)

    def test_parameter_hash_detects_radius_changes(self):
        self.assertNotEqual(digest({'radius':.001}),digest({'radius':.002}))


if __name__=='__main__':unittest.main()
