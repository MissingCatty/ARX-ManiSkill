"""Gate the existing 400-slot controller on successful endpoint-selection exit."""
import argparse
from pathlib import Path
import subprocess
import time
import traceback

import 运行选定高度抓取对照 as base
from 公共 import read,save,sha,now

SELF=Path(__file__).resolve()


def main(root,selection):
    assert not list(root.glob('*/claims/*.json'))
    assert not (root/'registration.json').exists()
    waiting=read(root/'等待高度选择.json')
    assert waiting['physical_budget']==400
    assert all(sha(path)==digest for path,digest in waiting['source_sha256'].items())
    receipt={'created_at':now(),'previous_unit':'arx-height-grasp-pair-20261120-01.service',
        'selection_unit':'arx-height-candidates-20261120-01.service','actual_test_slots_claimed':0,
        'physical_budget':400,'source_sha256':{str(SELF):sha(SELF)},
        'change':'Stop if the offline process exits or fails without a complete successful selection record. No algorithm, scene or physical budget change.'}
    save(root/'调度承接登记.json',receipt)
    deadline=time.monotonic()+3*3600
    while True:
        if (selection/'故障.json').exists():raise RuntimeError('Offline height verification failed; no physical slots claimed')
        status=subprocess.check_output(['systemctl','--user','show','arx-height-candidates-20261120-01',
            '-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],text=True)
        fields=dict(line.split('=',1) for line in status.splitlines())
        if fields['ActiveState']=='failed':raise RuntimeError('Offline selection service failed: '+status)
        if fields['MainPID']=='0':
            assert fields['ExecMainStatus']=='0',status
            assert (selection/'高度选择.json').exists(),'Offline process exited without selection evidence'
            assert read(selection/'高度选择.json')['passed']
            break
        if time.monotonic()>deadline:raise TimeoutError('Bounded height-selection wait exceeded')
        time.sleep(3)
    base.prepare(root,selection)
    for path in [root/'registration.json',*[root/arm/'registration.json' for arm,_,_ in base.ARMS]]:
        reg=read(path)
        reg['source_hashes'].update({str(SELF):sha(SELF),str(root/'调度承接登记.json'):sha(root/'调度承接登记.json')})
        save(path,reg)
    base.run(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--selection',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve()
    try:main(root,args.selection.resolve())
    except BaseException:
        save(root/'调度故障.json',{'time':now(),'error':traceback.format_exc(),
            'claimed_slots':len(list(root.glob('*/claims/*.json'))),'automatic_retries':False})
        raise
