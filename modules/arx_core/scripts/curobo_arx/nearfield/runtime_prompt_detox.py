"""Advisor26 explicitly registered visual-only clearance; physical robot unchanged."""
import argparse
from pathlib import Path
from scripts.curobo_arx.data_factory.common import read
from scripts.curobo_arx.data_factory import visual_geometry
from . import runtime_entry_guard,runtime_learned


def install(campaign):
    reg=read(Path(campaign)/'registration.json');policy=reg['visual_geometry_policy']
    expected={**visual_geometry.POLICY,'minimum_surface_distance_m':.008}
    if policy!=expected or reg['visual_clearance_change']['new_m']!=.008:raise ValueError('Unexpected camera-geometry threshold change')
    # inspect_row's default and explicit planner policy must agree in this new process.
    visual_geometry.POLICY.update(policy)
    runtime_entry_guard.install()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--launch',required=True);a=p.parse_args()
    install(a.campaign);runtime_learned.LearnedGroup(a.campaign,a.launch).run()
