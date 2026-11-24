"""Same frozen physical executor, with advisor23's numerical action parser."""
import argparse
from training.nearfield_arx.boundary_fit import install_runtime_codec

if __name__ == '__main__':
    install_runtime_codec()
    from .runtime_learned import LearnedGroup
    parser = argparse.ArgumentParser();parser.add_argument('--campaign', required=True);parser.add_argument('--launch', required=True)
    args = parser.parse_args();LearnedGroup(args.campaign, args.launch).run()
