"""Build an immutable, success-only browsing view of admitted ARX exports.

Original episodes, failures, hashes and training exports are never rewritten.
"""
import argparse
import json
from pathlib import Path
import sqlite3
from scripts.curobo_arx.data_factory.common import ROOT,read,sha,write
from scripts.curobo_arx.lilo_vla.egad_evaluate import evaluate


def checked_episode(root, ident):
    root=Path(root);meta=read(root/'episode.json');physical=read(root/'physical_receipt.json')
    quality=read(root/'quality_receipt.json')
    if meta['attempt_id']!=ident or not physical['pass'] or not quality['pass']:
        raise ValueError('Episode is not an admitted successful demonstration: '+ident)
    fresh=evaluate(root)
    if not fresh['pass']:raise ValueError('Original physical evaluator rejected success view episode: '+ident)
    return {'episode_id':ident,'object_id':meta['object_id'],'source_episode':str(root.resolve()),
        'grasp_success':True,'physical_recheck_pass':True,'hold_min_rise_m':fresh['hold_min_rise_m'],
        'hold_seconds':fresh['recorded_hold_duration_s'],'bilateral_closure':fresh['bilateral_closure'],
        'distance_mm':meta['sampled_entry']['distance_mm'],'entry_sample':meta['sampled_entry']['sample'],
        'physical_receipt_sha256':sha(root/'physical_receipt.json'),
        'quality_receipt_sha256':sha(root/'quality_receipt.json')}


def build(exports,out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve existing views; choose a new output directory')
    rows=[];sources=[];seen=set()
    for directory in map(lambda p:Path(p).resolve(),exports):
        manifest=read(directory/'manifest.json');index=read(directory/'entry_index_manifest.json')
        if index['index_sha256']!=sha(directory/'entry_index.jsonl') or index['lilo_manifest_sha256']!=sha(directory/'manifest.json'):
            raise ValueError('Export index hash chain mismatch')
        entries=[json.loads(line) for line in (directory/'entry_index.jsonl').read_text().splitlines()]
        if len(entries)!=manifest['episodes'] or len(entries)!=index['entries']:raise ValueError('Export membership count mismatch')
        campaign=Path(manifest['source_campaign']);db=sqlite3.connect('file:'+str(campaign/'ledger.sqlite3')+'?mode=ro',uri=True)
        try:committed={ident:path for ident,path in db.execute("SELECT id,episode FROM attempts WHERE state='committed'")}
        finally:db.close()
        for entry in entries:
            ident=entry['episode_id'];root=Path(entry['episode_path'])
            if ident in seen or committed.get(ident)!=str(root):raise ValueError('Duplicate or noncommitted episode in view: '+ident)
            row=checked_episode(root,ident)
            for key in ('physical_receipt_sha256','quality_receipt_sha256'):
                if row[key]!=entry[key]:raise ValueError('Admission receipt changed: '+ident)
            seen.add(ident);rows.append(row)
        sources.append({'campaign':str(campaign),'export':str(directory),'episodes':len(entries),
            'manifest_sha256':sha(directory/'manifest.json'),'entry_index_sha256':sha(directory/'entry_index.jsonl')})
    out.mkdir(parents=True)
    for row in rows:
        target=out/'successes'/row['object_id']/row['episode_id'];target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to(row['source_episode'],target_is_directory=True)
        row['view_path']=str(target)
    for source in sources:
        target=out/'exports'/Path(source['campaign']).name;target.parent.mkdir(exist_ok=True)
        target.symlink_to(source['export'],target_is_directory=True)
    with (out/'manifest.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    (out/'README.md').write_text('''# 仅含合格抓取示教的浏览入口

`successes/<物体>/<轨迹ID>/` 中只含已入库成功，重新用原物理评估器核对通过；没有失败或中断轨迹。
原始数据用软链接引用，不修改旧目录、轨迹、哈希、停止标记或评估器。复制为独立数据包时需复制软链接指向的实际文件。

`manifest.jsonl` 列出每条示教的最终物理结果和来源；`exports/` 指向原训练/验证索引与入口索引。
原 `groups/` 是全部尝试的原始档案，包含失败和中断，不能直接当成功训练集使用。
旧 `grasp_report.json` 的 `success` 仅反映动作流程完成，最终抓取成功以 `physical_receipt.json`、`quality_receipt.json` 和已入库索引为准。

本视图只改善数据入口，不扩大成功判据，不把失败伪装成示教，也不代表已具备VLA泛化能力。
''')
    result={'episodes':len(rows),'objects':len({r['object_id'] for r in rows}),'sources':sources,
        'manifest_sha256':sha(out/'manifest.jsonl'),'original_physical_evaluator_sha256':sha(ROOT/'scripts/curobo_arx/lilo_vla/egad_evaluate.py'),
        'original_data_changed':False,'scope':'Success-only browsing view; training still uses the original verified exports'}
    write(out/'READY.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',dest='exports',action='append',required=True)
    parser.add_argument('--out',required=True);args=parser.parse_args()
    print(build(args.exports,args.out))
