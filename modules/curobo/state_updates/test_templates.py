from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from .artifacts import ModelTemplate, ReceiptWriter


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.robot = {'kinematics': {'lock_joints': {'finger1': .044, 'finger2': .044},
            'collision_spheres': {'link': [{'center': [1e-7, -.03, .02], 'radius': .004}]}},
            'other': [1, True, None, 'unchanged']}
        self.template = ModelTemplate(self.robot, ['finger1', 'finger2'])

    def test_scientific_notation_remains_a_numeric_yaml_scalar(self):
        value = yaml.safe_load(self.template.render({'finger1': 1e-7, 'finger2': 0.}).text)
        self.assertEqual(value['robot_cfg']['kinematics']['lock_joints'], {'finger1': 1e-7, 'finger2': 0.})

    def test_all_nonlock_model_values_are_unchanged(self):
        expected = deepcopy(self.robot)
        expected['kinematics']['lock_joints'] = {'finger1': .02183451, 'finger2': .02183451}
        actual = yaml.safe_load(self.template.render(expected['kinematics']['lock_joints']).text)
        self.assertEqual(actual, {'robot_cfg': expected})
        self.assertEqual(self.robot['kinematics']['lock_joints']['finger1'], .044)

    def test_rendered_model_is_written_before_writer_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            text = self.template.render({'finger1': .01, 'finger2': .02})
            writer = ReceiptWriter()
            writer.submit(temp, {'arx.yml': text})
            writer.finish()
            self.assertEqual((Path(temp) / 'arx.yml').read_text(), text.text)

    def test_nonfinite_lock_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.template.render({'finger1': float('nan'), 'finger2': .02})

    def test_wrong_lock_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'lock set'):
            self.template.render({'finger1': .01})


if __name__ == '__main__':
    unittest.main()
