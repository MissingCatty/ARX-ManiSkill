"""Match the runtime package search path rather than the test runner's path."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]


class RuntimePackageImports(unittest.TestCase):
    def test_admission_imports_as_it_does_inside_grasp_execution(self):
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'scripts/curobo_arx')
        result=subprocess.run([sys.executable,'-c',
            'import lilo_vla.handoff_profile; print("package-import-ok")'],cwd=ROOT,env=env,
            capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('package-import-ok',result.stdout)


if __name__=='__main__':unittest.main()
