"""Reject partial quality checks and changed model/source payloads before load."""
from pathlib import Path
import tempfile
import unittest

from arx_project.artifacts import atomic_json
from arx_project.layout import sha256
from .service import load_profile


class ValidatedProfile(unittest.TestCase):
    def test_incomplete_and_changed_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);weight=root/'weight.bin';weight.write_bytes(b'validated model')
            source=root/'source.py';source.write_text('validated source')
            profile={'validated_cases':50,'record_integrity':True,'geometry_quality_stable':True,
                'source_hashes':{'source.py':sha256(source)},
                'weights':{'weight.bin':{'size':weight.stat().st_size,'sha256':sha256(weight)}}}
            config=root/'profile.json';atomic_json(config,profile)
            self.assertEqual(load_profile(config,project_root=root)['validated_cases'],50)
            atomic_json(config,{**profile,'validated_cases':49})
            with self.assertRaises(RuntimeError):load_profile(config,project_root=root)
            atomic_json(config,profile);weight.write_bytes(b'altered model!!')
            with self.assertRaises(RuntimeError):load_profile(config,project_root=root)

    def test_quality_failure_cannot_be_served(self):
        with tempfile.TemporaryDirectory() as folder:
            config=Path(folder)/'profile.json'
            atomic_json(config,{'validated_cases':50,'record_integrity':True,'geometry_quality_stable':False})
            with self.assertRaises(RuntimeError):load_profile(config)


if __name__=='__main__':unittest.main()
