"""Restore and audit the recorded bottle_clear_00 state from an explicit bundle.

Uses DynamicBridge.dynamic_setup; does not fabricate missing research state files
or change mechanical, contact, material, drive, or solver configuration.
"""
from arx_project.layout import canonical_path
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
from client import Client
from research_guard import invariant_digest


def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        result=[]
        for key in sorted(set(a)|set(b)):
            if key not in a or key not in b:result.append(path+'/'+key)
            else:result+=differences(a[key],b[key],path+'/'+key)
        return result
    return [] if a==b else [path]


def physical(fingerprint, additional_render_sensor=False):
    result=copy.deepcopy(fingerprint['invariant'])
    result.pop('configured',None)
    for path,prim in result['usd'].items():
        if path.startswith('/World/COOLASSensors/'):
            prim['attributes']={k:v for k,v in prim['attributes'].items() if not k.startswith('xformOp')}
    if additional_render_sensor:
        path='/World/COOLASSensors/right_wrist'
        if path in result['usd']:
            sensor=result['usd'][path]
            assert sensor['type']=='Camera' and not sensor['attributes'] and not sensor['relationships'],sensor
            assert not any('Physics' in schema or 'Physx' in schema for schema in sensor['schemas']),sensor
            result['usd'].pop(path)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mailbox',type=Path,help='Reuse an already running simulator transport')
    parser.add_argument('--scenario',type=Path,help='Explicit existing-object substitution at the legacy clear-table pose')
    parser.add_argument('--additional-render-sensor',action='store_true',
                        help='Allow one physics-free right-wrist render camera; calibrated robot sensor transforms remain frozen')
    args=parser.parse_args()
    out=args.output.resolve();ref=out/'historical_reference';transport=args.mailbox.resolve() if args.mailbox else out/'mailbox'
    os.environ['ARX_CUROBO_VALIDATION_DIR']=str(transport);client=Client(transport)
    def read(path):return json.loads(path.read_text())
    def save(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
    deadline=time.monotonic()+1200
    while not (client.root/'ready.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Bottle simulator startup did not finish')
        time.sleep(2)
    scenario=read(args.scenario) if args.scenario else read(ref/'registration.json')['scenario']
    catalog=read(ref/'catalog.json')
    historical=read(ref/'setup.json')
    expected=historical['initial_conditions']
    if args.scenario:
        from static_pick.scene_initial_conditions import expected_initial
        expected=expected_initial(expected,catalog,scenario)
    print('Restoring historical robot reset and registered target: '+scenario['asset'],flush=True)
    setup=client.request('dynamic_setup',scenario=scenario,timeout=240)
    save('setup.json',setup)
    mismatches=differences(physical(historical['fingerprint'],args.additional_render_sensor),
                          physical(setup['fingerprint'],args.additional_render_sensor))
    configuration_diffs=differences(historical['fingerprint']['invariant']['configured'],setup['fingerprint']['invariant']['configured'])
    declared_diffs=differences(expected,setup['initial_conditions'])
    if setup['physics_steps']!=360 or abs(setup['captured_sim_time']-setup['reset_sim_time']-3.)>.001:
        declared_diffs.append('historical three-second settling')
    no_obstacles=not any(path.startswith(('/World/StaticPickPole','/World/StaticPickRack','/World/DynamicPickObstacle')) for path in setup['fingerprint']['invariant']['usd'])
    review={'historical_physical_properties_equal':not mismatches,'historical_physical_mismatch_paths':mismatches,'configured_difference_paths':configuration_diffs,'declared_initial_conditions_equal':not declared_diffs,'initial_condition_difference_paths':declared_diffs,'no_added_obstacles':no_obstacles,'source':'Recorded successful bottle_clear_00 setup.json, not the later unverified startup lift height','historical_research_guard_610_source_hashes':'unverified; original guard absent on this machine; historical live USD/getter evidence is compared directly'}
    review['additional_right_wrist_render_sensor']=args.additional_render_sensor
    review['object_substitution']=scenario if args.scenario else None
    if args.scenario:
        review['source']='Historical robot reset/physics plus explicitly registered existing-object substitution; not the original BBQ target pose'
    save('historical_configuration_review.json',review)
    assert not mismatches,'Historical physical properties differ; see historical_configuration_review.json'
    assert not declared_diffs,'Declared initial conditions differ; see historical_configuration_review.json'
    assert no_obstacles
    save('active_asset.json',setup['asset'])
    # These sections include calibration, gains, ranges and all mechanical mappings.
    for section in ('controllers','calibration','motion','chassis','lift','robot','arms','objects'):
        assert historical['fingerprint']['invariant']['configured'][section]==setup['fingerprint']['invariant']['configured'][section],section
    view=read(ref/'viewport_initial.json');client.request('grasp_viewport_state',restore=view)
    actual_view=client.request('grasp_viewport_state');save('viewport_initial.json',actual_view)
    sources={}
    for name in ('arx-sim/config/sim.yaml','arx-sim/config/runtime.yaml','arx-sim/config/controllers.yaml','arx-sim/config/tasks/grasp.yaml','arx-sim/config/tasks/grasp.usda','configs/arx.yaml','configs/calibration.yaml'):
        path=ROOT/name;sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in historical['fingerprint']['used_layers']:
        if name.startswith('/workspace/asset/'):
            path=canonical_path(ROOT/'asset'/name.removeprefix('/workspace/asset/'))
        elif name.startswith('/workspace/COOLAS-ARX/asset/'):
            path=canonical_path(ROOT/'asset'/name.removeprefix('/workspace/COOLAS-ARX/asset/'))
        else:path=Path(name)
        if path.is_file():sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    save('guard.json',{'invariant_sha256':invariant_digest(setup['fingerprint']),'file_sha256':sources})
    save('fingerprint_before.json',setup['fingerprint'])
    client.request('static_view_capture',path=str(out/'scene_initial.png'))
    save('setup_verification.json',{'pass':True,**review,'current_protected_source_count':len(sources)})
    print('SETUP VERIFIED: historical physical properties and declared initial state match; no added obstacles.',flush=True)


if __name__=='__main__':main()
