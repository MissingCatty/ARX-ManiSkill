from __future__ import annotations

import importlib.util


MODULES = [
    "torch",
    "torchvision",
    "websockets",
    "msgpack",
    "yaml",
    "lightning.pytorch",
    "ocnn",
    "dwconv",
    "torch_scatter",
    "graspnetAPI",
    "open3d",
    "cv2",
    "imageio.v3",
    "PIL",
    "segmentation_models_pytorch",
    "rotary_embedding_torch",
    "mcubes",
    "skimage",
    "sklearn",
    "scipy",
    "trimesh",
    "webdataset",
    "wandb",
    "ofe.cuda.ofe",
]


def main() -> int:
    missing = []
    for module in MODULES:
        name = module.split(".")[0]
        if importlib.util.find_spec(name) is None:
            missing.append(module)
            print(f"MISS {module}")
            continue
        try:
            __import__(module)
            print(f"OK   {module}")
        except Exception as exc:
            missing.append(module)
            print(f"FAIL {module}: {type(exc).__name__}: {exc}")

    try:
        import torch

        print(f"torch_version {torch.__version__}")
        print(f"torch_cuda {torch.version.cuda}")
        print(f"cuda_available {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch_runtime_check_failed {type(exc).__name__}: {exc}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
