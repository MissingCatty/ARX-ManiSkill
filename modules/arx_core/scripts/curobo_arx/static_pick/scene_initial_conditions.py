"""Declared initial conditions for the legacy clear-table setup."""
import copy


def scenario_for(catalog, name):
    asset = next((a for a in catalog if a['name'] == name), None)
    if asset is None or 'egad' in asset['container_path'].lower():
        raise ValueError('Choose an existing non-EGAD tabletop asset')
    return dict(target='test_object', asset=name, placement=dict(
        position=[4.048, 2.08, .679 + asset['size_m'][2] / 2 + .012]))


def expected_initial(historical, catalog, scenario):
    if scenario != scenario_for(catalog, scenario['asset']):
        raise ValueError('Only the registered legacy clear-table placement is supported')
    initial = copy.deepcopy(historical)
    for i, asset in enumerate(catalog):
        record = initial['objects']['asset:' + asset['name']]
        record['position'] = [10 + i * .4, 10., .4]
        record['quaternion_wxyz'] = [1., 0., 0., 0.]
    initial['objects']['asset:' + scenario['asset']]['position'] = scenario['placement']['position']
    return initial
