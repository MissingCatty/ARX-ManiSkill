"""Independent source guards retain failures after the cloud route is retired."""
from pathlib import Path
import tempfile
import unittest
from paths import sha, guard


class GuardTests(unittest.TestCase):
    def test_source_change_stops(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'source.py'; p.write_text('original')
            registration = {'runtime_source_hashes': {str(p): sha(p)}}
            guard(registration); p.write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'Frozen runtime source changed'):
                guard(registration)

    def test_missing_source_stops(self):
        with self.assertRaises(FileNotFoundError):
            guard({'runtime_source_hashes': {'/nonexistent/arx-registered-source': '0'}})


if __name__ == '__main__':
    unittest.main()
