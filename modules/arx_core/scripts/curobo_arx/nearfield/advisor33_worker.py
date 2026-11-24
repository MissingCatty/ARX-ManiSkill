import argparse
from scripts.curobo_arx.seed_augmentation import worker
from scripts.curobo_arx.data_factory.common import read
from .advisor33_teacher import Teacher

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--registration',required=True)
    a=p.parse_args();worker.Teacher=Teacher;worker.serve(a.directory,read(a.registration))
