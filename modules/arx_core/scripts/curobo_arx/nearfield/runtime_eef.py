"""Advisor24 codec on the same ARX executor and physical gates."""
import argparse
from training.nearfield_arx.eef_fit import install_runtime_codec

if __name__ == '__main__':
    install_runtime_codec()
    from .runtime_learned import LearnedGroup
    p = argparse.ArgumentParser();p.add_argument('--campaign', required=True);p.add_argument('--launch', required=True)
    a = p.parse_args();LearnedGroup(a.campaign, a.launch).run()
