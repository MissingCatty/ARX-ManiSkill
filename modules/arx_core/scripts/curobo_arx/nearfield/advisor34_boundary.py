"""Same hand-range isolation; separately account simulation and wall timeouts."""
from .advisor33_boundary import ISOLATED_OUTCOMES as PREVIOUS

ISOLATED_OUTCOMES = PREVIOUS | frozenset(('stationary_wait_10sim_s_timeout',
                                        'stationary_wall_safety_120s_timeout'))
