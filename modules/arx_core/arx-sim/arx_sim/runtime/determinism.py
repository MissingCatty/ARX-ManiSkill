from __future__ import annotations

import random


def configure_determinism(seed: int) -> None:
    """Seed the ordinary Python/NumPy sources used by this runtime."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
