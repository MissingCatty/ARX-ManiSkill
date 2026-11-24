"""Advisor34 simulation-clock stop budget; unchanged physical stop thresholds."""
import math
from .advisor32_rules import StopSample, settle as measured_settle

VERSION = 'arx_advisor34_sim_clock_v1'
WAIT_SIM_SECONDS = 10.
WAIT_WALL_SECONDS = 120.


def settle(samples, *, active, fault, elapsed_wall_s, elapsed_sim_s):
    if not all(math.isfinite(x) and x >= 0 for x in (elapsed_wall_s, elapsed_sim_s)):
        raise ValueError('Invalid simulation/wall wait interval')
    # Reuse the frozen measured-position/speed checks, without its former wall
    # timeout. Ten 5-Hz intervals need eleven distinct recorded samples.
    answer = measured_settle(samples, active=active, fault=fault, elapsed_wall_s=0.)
    if answer['origin'] == 'settled_by_position_window' and answer['window_samples'] < 11:
        answer.update(pass_=False, origin=None)
        answer['pass'] = answer.pop('pass_')
    reason = ('stationary_wall_safety_120s_timeout' if elapsed_wall_s >= WAIT_WALL_SECONDS else
              'stationary_wait_10sim_s_timeout' if elapsed_sim_s >= WAIT_SIM_SECONDS else None)
    # A completed stable window exactly at the sim deadline is admissible.
    if reason == 'stationary_wait_10sim_s_timeout' and answer['pass'] and elapsed_sim_s <= WAIT_SIM_SECONDS+1e-8:
        reason = None
    answer.update(timeout=reason is not None, timeout_reason=reason, clock_version=VERSION,
                  wait_elapsed_sim_s=elapsed_sim_s, wait_elapsed_wall_s=elapsed_wall_s,
                  maximum_wait_sim_s=WAIT_SIM_SECONDS, maximum_wait_wall_s=WAIT_WALL_SECONDS,
                  effective_wait_intervals_5Hz=int(math.floor((elapsed_sim_s+1e-8)*5)))
    if reason:
        answer.update(origin=None)
        answer['pass'] = False
    return answer
