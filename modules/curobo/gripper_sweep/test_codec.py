from types import SimpleNamespace
import unittest
import yaml
from .codec_adapter import attach_codec


def encode(value):return yaml.safe_dump(value,sort_keys=False)


class CodecTests(unittest.TestCase):
    def make(self):
        module=SimpleNamespace(bind_inputs=encode)
        class Pool:
            yaml_cache={}
            def bind(self,value):return module.bind_inputs(value)
        return attach_codec(Pool(),module),module

    def test_switch_preserves_values_and_restores_original_function(self):
        pool,module=self.make();value={'x':[.044,1e-7,True,None],'lock_joints':{'left':.03}}
        self.assertEqual(pool.bind(value),encode(value));self.assertEqual(pool.binding_codec.report()['dump_calls'],0)
        pool.fast_binding_enabled=True;actual=pool.bind(value)
        self.assertEqual(yaml.safe_load(actual),value);self.assertIs(module.bind_inputs,encode)
        self.assertEqual(pool.binding_codec.report()['dump_calls'],1)

    def test_binding_exception_restores_private_function(self):
        pool,module=self.make();pool.fast_binding_enabled=True
        with self.assertRaises(yaml.representer.RepresenterError):pool.bind({'unsupported':object()})
        self.assertIs(module.bind_inputs,encode);self.assertIsNone(pool.binding_input_scope.owner)


if __name__=='__main__':unittest.main()
