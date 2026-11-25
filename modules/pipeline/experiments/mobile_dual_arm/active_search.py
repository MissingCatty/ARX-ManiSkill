"""Wire map proposals to planner RPC; algorithm modules own their own logic."""
from arx_project.layout import ROOT, read_json
from modules.mobile_manipulation.inverse_reachability import InverseReachability
from modules.mobile_manipulation.hand_table_precheck import HandTablePrecheck
from modules.mobile_manipulation.station_search import StationSearch
from modules.reachability.orientation_index import OrientationIndex


class StationBackend:
    def __init__(self, request):
        self.request = request

    def endpoint(self, directory, arm):
        return self.request(directory, station_probe=True, arm=arm)

    def plan(self, directory, arm=None):
        return self.request(directory, **({} if arm is None else {'arm': arm}))


def make_search(model, settings, request, options):
    index = OrientationIndex(ROOT/options['map_directory'], ROOT/options['index_directory'])
    generator = InverseReachability(index, model,
        read_json(ROOT/'simulation/robots/arx/runtime/action_tcp.json')['T_eef_action'], options)
    policy = read_json(ROOT/'configs/policies/fingertip_table.json')
    allowance = policy['maximum_penetration_m'] if policy['enabled'] else 0.
    return StationSearch(model, settings, StationBackend(request), generator=generator, options=options,
                         precheck_factory=lambda state: HandTablePrecheck(model, state, allowance))


def service_search(model, settings, services, options):
    def request(directory, **flags):
        response = services.request(options['planner_module'], directory, **flags)
        # Services.request already treats RPC infrastructure failures as fatal.
        if response.get('status') != 'ok':
            raise RuntimeError('Station planner RPC failed: '+str(response))
        return response['result']
    return make_search(model, settings, request, options)
