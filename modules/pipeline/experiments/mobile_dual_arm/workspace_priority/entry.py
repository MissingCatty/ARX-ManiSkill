"""Optional high-workspace selector for a future, separately frozen V4 run."""
from arx_project.layout import ROOT, read_json
from modules.mobile_manipulation.workspace_priority.search import HighWorkspaceSearch
from modules.pipeline.experiments.mobile_dual_arm.active_search import StationBackend


def make_search(model, settings, request, policy=None):
    policy = dict(policy) if policy is not None else read_json(ROOT/'configs/experiments/high_workspace_placement.json')
    workspace = read_json(ROOT/policy['workspace'])
    return HighWorkspaceSearch(model, settings, StationBackend(request), workspace, policy)


def service_search(model, settings, services, planner_module='curobo_dual_arm'):
    def request(directory, **flags):
        response = services.request(planner_module, directory, **flags)
        if response.get('status') != 'ok':
            raise RuntimeError('Native planner infrastructure error: '+str(response))
        return response['result']
    return make_search(model, settings, request)
