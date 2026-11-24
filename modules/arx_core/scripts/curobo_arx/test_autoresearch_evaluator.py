"""Check that reporting or incomplete trajectories cannot spoof ARX success.

These are evaluator checks against archived ARX data, not new robot trials.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from autoresearch import evaluate, ROOT


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        source = ROOT / 'runs/visual_feedback_20261019/01_bagel_baseline'
        for path in source.glob('*.json'):
            shutil.copy2(path, self.folder / path.name)

    def tearDown(self):
        self.temp.cleanup()

    def change(self, file, transform):
        path = self.folder / file
        data = json.loads(path.read_text())
        transform(data)
        path.write_text(json.dumps(data))

    def test_archived_real_arx_success(self):
        self.assertTrue(evaluate(self.folder)['pass'])

    def test_report_alone_cannot_override_fallen_object(self):
        def lower(data):
            for rec in data['records']:
                rec['objects']['test_object'][2][3] = .1
        self.change('hold_execution.json', lower)
        self.assertTrue(json.loads((self.folder / 'result.json').read_text())['success'])
        self.assertFalse(evaluate(self.folder)['pass'])

    def test_missing_lift_cannot_be_success(self):
        (self.folder / 'lift_execution.json').unlink()
        self.assertFalse(evaluate(self.folder)['pass'])

    def test_dense_collision_certificate_rejected(self):
        self.change('lift_optimization.json', lambda d: d.update(invalid_samples=1))
        value = evaluate(self.folder)
        self.assertFalse(value['pass'])
        self.assertFalse(value['guard_pass'])

    def test_unexpected_physical_contact_rejected(self):
        def contact(data):
            data['contacts'].append({'a':'/World/Robot/right_link26',
                                    'b':'/World/Table', 'num_contacts':1})
        self.change('lift_execution.json', contact)
        self.assertFalse(evaluate(self.folder)['guard_pass'])

    def test_truncated_hold_rejected(self):
        self.change('hold_execution.json', lambda d: d.update(records=d['records'][:2]))
        self.assertFalse(evaluate(self.folder)['pass'])


if __name__ == '__main__':
    unittest.main()
