"""ARX physical TCP and the paired GraspGenX conditioning frame."""
import numpy as np
VERSION='arx_graspgenx_virtual_origin100_contact1mm_v1'
ORIGIN_BACK_M=.100
CONTACT_OFFSET_M=.001
TCP_M=.043
R_EEF_GENERATOR=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
T_EEF_GENERATOR=np.eye(4)
T_EEF_GENERATOR[:3,:3]=R_EEF_GENERATOR
T_EEF_GENERATOR[2,3]=-ORIGIN_BACK_M

def generator_to_eef(pose):
    return np.asarray(pose)@np.linalg.inv(T_EEF_GENERATOR)


def generator_to_contact(pose):
    eef=generator_to_eef(pose)
    contact=eef.copy();contact[:3,3]+=TCP_M*eef[:3,2]
    return contact

