import unittest

from arx_sim.runtime.rendering import configure_camera_antialiasing


class RtxSettings:
    """Model RTX's restriction that would otherwise silently reject TAA."""

    def __init__(self):
        self.settings = {
            "/rtx-transient/post/aa/limitedOps": True,
            "/rtx/post/aa/op": 3,
            "/physics/dt": 1 / 120,
        }

    def set_setting(self, name, value):
        if name == "/rtx/post/aa/op" and value in (0, 1, 2):
            if self.settings["/rtx-transient/post/aa/limitedOps"]:
                return
        self.settings[name] = value


class CameraRenderingTests(unittest.TestCase):
    def test_taa_is_effective_without_changing_physics(self):
        app = RtxSettings()
        configure_camera_antialiasing(app, 1)
        self.assertEqual(app.settings["/rtx/post/aa/op"], 1)
        self.assertEqual(app.settings["/physics/dt"], 1 / 120)

    def test_dlss_remains_available_when_explicitly_selected(self):
        app = RtxSettings()
        configure_camera_antialiasing(app, 3)
        self.assertEqual(app.settings["/rtx/post/aa/op"], 3)
        self.assertTrue(app.settings["/rtx-transient/post/aa/limitedOps"])

    def test_invalid_mode_does_not_modify_settings(self):
        app = RtxSettings()
        original = dict(app.settings)
        with self.assertRaises(ValueError):
            configure_camera_antialiasing(app, 99)
        self.assertEqual(app.settings, original)


if __name__ == "__main__":
    unittest.main()
