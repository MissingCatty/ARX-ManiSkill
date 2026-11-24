"""Compatibility entry: module workers and shared RPC live in ARX-ManiSkill."""
import numpy as np
from common import paths,read,save,sha
from arx_project.registry import load_adapter
from arx_project.service import main
SAM3=load_adapter('sam3').SAM3
RaySt3R=load_adapter('rayst3r').RaySt3R
GraspGenX=load_adapter('graspgenx').GraspGenX
planner=load_adapter('curobo').create
if __name__=='__main__':
    paths()
    main()
