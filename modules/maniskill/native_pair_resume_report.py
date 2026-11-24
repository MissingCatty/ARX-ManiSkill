"""Current cohort report with interruption time separate from active round time."""
import argparse
from datetime import datetime
from pathlib import Path
import time

import native_pair_report as report
from graspcl_pipeline import read,save,sha


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.output.resolve();cache={}
save(root / 'report_registration.json',{'source_sha256':sha(Path(__file__)),'read_only_observer':True,
    'upstream_report_sha256':sha(Path(report.__file__)),'selection_or_outcome_updates':False})
while True:
    done=report.report(root,cache)
    clock=read(root / 'suite_timing.json') if (root / 'suite_timing.json').exists() else {}
    if 'segment_started_at' in clock:
        elapsed=(datetime.now().astimezone()-datetime.fromisoformat(clock['segment_started_at'])).total_seconds()
        active=clock.get('total_active_suite_wall_seconds',clock['prior_segments_total_seconds']+elapsed)
        gap=clock['maintenance_gap_seconds']
        target=root / '耗时统计.md'
        text=target.read_text().replace('两轮合计总耗时（含启动和服务关闭）','两轮合计总耗时（含启动、中断维护和服务关闭）')
        text=text.replace('整轮用时直接测墙钟时间，不把逐例用时相加。','本轮用时累计各运行段的墙钟时间，不把逐例用时相加，中断维护另列。')
        text+='\n## 中断与续跑计时\n\n'
        text+=f'- 中断维护时间：{report.duration(gap)}。\n'
        text+=f'- 累计实际运行时间（含各次启动、关闭，不含中断维护）：{report.duration(active)}。\n'
        text+=f'- 含中断维护的累计墙钟时间：{report.duration(active+gap)}。\n'
        temporary=target.with_suffix('.resume.tmp');temporary.write_text(text);temporary.replace(target)
    if done:break
    time.sleep(15)
