"""Reject detached/reallocated scene storage and prevent ownership mistakes."""
from types import SimpleNamespace
import unittest
import torch

from .session import AnalyticBatchSession,scene_storage


class SharedSceneTests(unittest.TestCase):
    def make_session(self):
        scene=SimpleNamespace(data={'poses':torch.zeros((8,7))})
        obj=AnalyticBatchSession.__new__(AnalyticBatchSession)
        obj._shared_scene=scene;obj._collision=SimpleNamespace(checker=SimpleNamespace(scene_model=scene))
        obj._scene_storage=scene_storage(scene.data)
        return obj,scene

    def test_scene_values_may_change_but_reallocated_storage_is_rejected(self):
        session,scene=self.make_session();scene.data['poses'].add_(1)
        session.verify_shared_scene(scene)
        scene.data['poses']=scene.data['poses'].clone()
        with self.assertRaises(RuntimeError):session.verify_shared_scene(scene)

    def test_shared_scene_updates_belong_to_parent(self):
        session,scene=self.make_session()
        with self.assertRaises(RuntimeError):session.update_scene(None)
        with self.assertRaises(RuntimeError):session.verify_shared_scene(SimpleNamespace(data=scene.data))


if __name__=='__main__':unittest.main()
