"""Mechanical planning metric and no-regression gate, from immutable receipts."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from 公共 import read,MS


def evaluate(directory,baseline=None):
    sys.path.insert(0,str(MS/'arx_pipeline'))
    from scene_tools import ArmFK
    fk=ArmFK()
    d=read(Path(directory)/'汇总.json');rows=d['rows']
    assert d['total']==len(rows) and d['passes']==sum(r['success'] for r in rows),'Receipt totals disagree'
    reference=read(Path(baseline)/'汇总.json') if baseline else d
    by={r['case']:r for r in reference['rows']}
    assert set(by)=={r['case'] for r in rows}
    assert len(rows)==len(by)
    gains=[];regressions=[]
    for r in rows:
        b=by[r['case']]
        assert r['input_state_sha256']==b['input_state_sha256']
        assert r['input_candidates_sha256']==b['input_candidates_sha256']
        assert not r['G_modified'] and not r['hard_map_rejection']
        assert r['physical_attempts']==0
        if r['success']:
            selected=r['selected'];assert selected['original_full_scene_entry_checked'] and selected['original_dense_checks_passed']
            case=next(c for c in read(Path(directory).parents[1]/'场景清单.json')['cases'] if c['case']==r['case'])
            source=Path(case['directory']);state=read(source/'state.json')
            candidate=read(source/'candidates/candidate_pool.json')['candidates'][selected['candidate_index']]
            grasp=np.asarray(candidate['pose']).copy();grasp[:3,3]-=.043*grasp[:3,2]
            assert any(abs(selected['pregrasp_distance_m']-v)<1e-10 for v in [.02,.04,.06,.08]),'Undeclared H distance'
            pre=grasp.copy();pre[:3,3]-=selected['pregrasp_distance_m']*grasp[:3,2]
            lift=grasp.copy();lift[2,3]+=.03
            invbase=np.linalg.inv(np.asarray(state['links']['body4']))
            for expected,key,qkey in [(grasp,'grasp_world','q_grasp'),(pre,'pregrasp_world','q_pre'),(lift,'nominal_lift_world','q_nominal_lift')]:
                assert np.allclose(expected,np.asarray(selected[key]),atol=1e-10,rtol=0),'Goal changed'
                actual=fk.fk(np.asarray(selected[qkey]));target=invbase@expected
                assert np.linalg.norm(actual[:3,3]-target[:3,3])<=.00201,'IK position tolerance failed'
                assert Rotation.from_matrix(actual[:3,:3]@target[:3,:3].T).magnitude()<=.03001,'IK rotation tolerance failed'
            for route in selected['paths'].values():
                q=np.array(route['q']);assert q.ndim==2 and q.shape[1]==6 and np.isfinite(q).all()
                assert (q>=fk.lower-1e-6).all() and (q<=fk.upper+1e-6).all()
                times=np.asarray(route['times']);assert len(times)==len(q) and times[0]==0 and (np.diff(times)>0).all()
        if r['success'] and not b['success']:gains.append(r['case'])
        if b['success'] and not r['success']:regressions.append(r['case'])
    return {'pass':d['protected_sources_unchanged'] and not regressions,'score':d['passes']/d['total'],
            'passes':d['passes'],'total':d['total'],'gains':gains,'regressions':regressions,
            'median_seconds':float(np.median([r['wall_seconds'] for r in rows])),
            'median_ik_calls':float(np.median([r['counts'].get('ik_calls',0) for r in rows])),
            'scope':'offline complete initial planning; not physical grasp success'}


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--基线',type=Path);a=p.parse_args()
    print(json.dumps(evaluate(a.directory,a.基线),ensure_ascii=False))
