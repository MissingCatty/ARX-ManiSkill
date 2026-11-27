from pathlib import Path
import tempfile
import unittest

from .binding import MeasuredStartCollision
from .runtime import memory_pool


class RejectionTests(unittest.TestCase):
    def instance(self, error):
        class Base:
            def request(self, *args, **kwargs):
                raise error
        cls = memory_pool(Base)
        return cls.__new__(cls)

    def test_native_start_collision_remains_an_ordinary_planning_failure(self):
        pool = self.instance(MeasuredStartCollision())
        with tempfile.TemporaryDirectory() as temp:
            result = pool.request(Path(temp))
            self.assertFalse(result['success'])
            self.assertEqual(result['reason'], 'measured_start_state_in_collision')
            self.assertTrue((Path(temp) / 'plan_constraint_rejection.json').exists())

    def test_unrelated_assertion_is_not_converted_to_a_planning_failure(self):
        pool = self.instance(AssertionError('configuration corruption'))
        with self.assertRaisesRegex(AssertionError, 'configuration corruption'):
            pool.request('/unused')


if __name__ == '__main__':
    unittest.main()
