"""Keep configuration values, ownership, and original bind failure behavior."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest

import yaml

from .binding_serialization import BindingCodec, BindingInputScope, serialization_pool


def prepare_inputs(text, *, fail=False):
    value = yaml.safe_load(text)
    if fail:
        raise ValueError('Original input failure')
    return value, yaml.safe_dump(value, sort_keys=False)


class BindingSerializationTests(unittest.TestCase):
    def test_scalar_types_and_safe_roundtrip_match(self):
        text = 'a: 0.044\nb: -0.0\nc: 1.0e-06\nd: true\ne: null\nf: "001"\ng: [1, 2.0]\n'
        codec = BindingCodec()
        original, _ = prepare_inputs(text)
        current, emitted = codec.function(prepare_inputs)(text)
        self.assertEqual(original, current)
        for key in original:
            self.assertIs(type(original[key]), type(current[key]))
        self.assertEqual(yaml.safe_load(emitted), original)

    def test_reference_cache_is_copied_before_mutation(self):
        cached = {'robot': {'locks': [.044, .044]}}
        codec = BindingCodec({'reference': cached})
        first = codec.safe_load('reference');first['robot']['locks'][0] = 0.
        self.assertEqual(codec.safe_load('reference'), cached)
        self.assertEqual(codec.report()['cache_hits'], 2)

    def test_global_yaml_functions_are_never_patched(self):
        old_load, old_dump = yaml.safe_load, yaml.safe_dump
        codec = BindingCodec();module = SimpleNamespace(bind_inputs=prepare_inputs)
        scope = BindingInputScope(module, codec)
        with scope.activate():
            self.assertEqual(module.bind_inputs('x: 1')[0], {'x': 1})
            self.assertIs(yaml.safe_load, old_load);self.assertIs(yaml.safe_dump, old_dump)
        self.assertIs(module.bind_inputs, prepare_inputs)

    def test_exception_restores_original_policy(self):
        module = SimpleNamespace(bind_inputs=prepare_inputs)
        scope = BindingInputScope(module, BindingCodec())
        with self.assertRaisesRegex(ValueError, 'Original input failure'):
            with scope.activate(): module.bind_inputs('x: 1', fail=True)
        self.assertIs(module.bind_inputs, prepare_inputs)
        self.assertIsNone(scope.owner)

    def test_nested_or_changed_policy_is_rejected(self):
        module = SimpleNamespace(bind_inputs=prepare_inputs)
        scope = BindingInputScope(module, BindingCodec())
        with scope.activate():
            with self.assertRaises(RuntimeError):
                with scope.activate(): pass
        module.bind_inputs = lambda text: None
        with self.assertRaises(RuntimeError):
            with scope.activate(): pass

    def test_other_thread_cannot_consume_temporary_binding(self):
        module = SimpleNamespace(bind_inputs=prepare_inputs)
        scope = BindingInputScope(module, BindingCodec())
        with scope.activate(), ThreadPoolExecutor(max_workers=1) as worker:
            future = worker.submit(module.bind_inputs, 'x: 1')
            with self.assertRaises(RuntimeError): future.result()
        self.assertIs(module.bind_inputs, prepare_inputs)

    def test_original_bind_still_runs_and_propagates_its_error(self):
        class Base:
            def bind(self, *args, **kwargs):
                self.calls += 1
                raise ValueError('Original collision gate')
        pool = serialization_pool(Base).__new__(serialization_pool(Base))
        module = SimpleNamespace(bind_inputs=prepare_inputs)
        pool.binding_input_scope = BindingInputScope(module, BindingCodec());pool.calls = 0
        for enabled in [False, True]:
            pool.fast_binding_enabled = enabled
            with self.assertRaisesRegex(ValueError, 'Original collision gate'): pool.bind(None)
        self.assertEqual(pool.calls, 2)
        self.assertIs(module.bind_inputs, prepare_inputs)

    def test_unsafe_yaml_tags_still_fail(self):
        text = '!!python/object/apply:os.getcwd []'
        with self.assertRaises(yaml.YAMLError): BindingCodec().safe_load(text)

    def test_changed_scope_function_is_a_fault_and_restores_original(self):
        module = SimpleNamespace(bind_inputs=prepare_inputs)
        scope = BindingInputScope(module, BindingCodec())
        with self.assertRaises(RuntimeError):
            with scope.activate(): module.bind_inputs = lambda *a: None
        self.assertIs(module.bind_inputs, prepare_inputs)


if __name__ == '__main__': unittest.main()
