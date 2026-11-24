"""Four advisor29 slots using the existing bounded service/shutdown coordinator."""
import argparse
from training.nearfield_arx import stage_recovery as policy
from training.nearfield_arx.stage_verification import audit_policy
from . import closure_physical, learned_report


def run(campaign):
    policy.install(); learned_report.decode = policy.decode; learned_report.audit_policy = audit_policy
    return closure_physical.run(campaign)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('campaign'); run(parser.parse_args().campaign)
