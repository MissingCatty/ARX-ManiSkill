"""Keep automatic activation separate from the explicitly selected runtime."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from modules.curobo.gripper_sweep import runtime as regular
from . import runtime as analytic


class RuntimeTests(unittest.TestCase):
    def planner(self):return SimpleNamespace(pool=SimpleNamespace(fast_binding_enabled=False))

    def policy(self):return {'optional_runtime':{'C_YAML_binding':True,'GPU_analytic_endpoints':False}}

    def test_disabled_auto_flag_keeps_regular_factory_without_extension(self):
        with patch.object(regular,'read_json',return_value=self.policy()), \
             patch('modules.curobo.gripper_sweep.codec_adapter.create',return_value=self.planner()) as factory:
            regular.create('args');factory.assert_called_once_with('args')

    def test_explicit_analytic_entry_is_available_when_auto_use_is_disabled(self):
        with patch.object(regular,'read_json',return_value=self.policy()), \
             patch('modules.curobo.gripper_sweep.codec_adapter.create',return_value=self.planner()) as factory:
            analytic.create('args')
            self.assertIn('pool_extension',factory.call_args.kwargs)
            self.assertTrue(callable(factory.call_args.kwargs['pool_extension']))


if __name__=='__main__':unittest.main()
