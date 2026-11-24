"""Preserve the height sweep, validating the original EEF pose tolerance."""
import argparse
import inspect
from pathlib import Path
import shutil
import 比较升降高度 as original

SELF=Path(__file__).resolve()


def scan(out):
    source=inspect.getsource(original.scan)
    old='maximum_fk_error=0.;maximum_rotation_error=0.'
    assert source.count(old)==1
    source=source.replace(old,old+'maximum_eef_error=0.')
    old='assert ep.max()<=.00205 and er.max()<=.0301,(ep.max(),er.max())'
    new='''eef_xyz=xyz-.043*(actual_r@profile[:3,:3].T)[:,:,2]
                eef_error=np.linalg.norm(eef_xyz-world[found,:3,3],axis=1)
                maximum_eef_error=max(maximum_eef_error,float(eef_error.max()))
                assert eef_error.max()<=.00205 and er.max()<=.0301,(eef_error.max(),er.max())'''
    assert source.count(old)==1;source=source.replace(old,new)
    old="'maximum_independent_rotation_error_rad':maximum_rotation_error"
    assert source.count(old)==1
    source=source.replace(old,"'maximum_independent_eef_error_m':maximum_eef_error,"+old)
    namespace=dict(original.__dict__)
    exec(compile(source,str(SELF)+':EEF_tolerance','exec'),namespace)
    return namespace['scan'](out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','scan'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu')
    args=parser.parse_args();out=args.output.resolve()
    if args.stage=='prepare':
        original.prepare(out,args.gpu)
        reg=original.read(out/'登记.json');reg['source_sha256'][str(SELF)]=original.sha(SELF)
        reg.update(independent_error_reference='Original EEF <=2 mm and orientation <=0.03 rad; TCP error recorded separately.',
                   preflight_correction='Previous attempt stopped during independent EEF/TCP reference check, no robot motions.')
        original.save(out/'登记.json',reg);shutil.copy2(SELF,out/'代码快照'/SELF.name)
    else:scan(out)
