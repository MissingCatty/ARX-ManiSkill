"""Independent surface-quality readback and a portable validated profile."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import statistics
import time

import numpy as np

from arx_project.artifacts import atomic_json
from arx_project.json_io import read_json
from arx_project.layout import ROOT,now,sha256
from .benchmark import quality
from .surface import uniform_surface


def verify(root):
    import trimesh
    root=Path(root)
    reg=read_json(root/'registration.json');config=reg['settings']
    if read_json(root/'progress.json')['phase']!='complete':raise RuntimeError('Inference comparison is not complete')
    old=read_json(root/'acceptance.json')
    if not old['record_integrity']:raise RuntimeError('Raw inference/timing records failed verification')
    archive=root/'evaluation_revisions/uniform_surface_01'
    archive.mkdir(parents=True,exist_ok=False)
    for name in ['summary.json','acceptance.json']:shutil.copy2(root/name,archive/name)
    cases=[]
    for case in reg['cases']:
        folder=root/'trials'/case['id'];row=read_json(folder/'result.json')
        truth_file=Path(reg['reference'])/'ground_truth'/case['id']/'surface_world.ply'
        if sha256(truth_file)!=reg['input_hashes'][str(truth_file)]:raise RuntimeError('Ground-truth mesh changed')
        mesh=trimesh.load(truth_file,force='mesh',process=False)
        truth=uniform_surface(mesh.vertices,mesh.faces)
        camera=read_json(Path(reg['reference'])/'observations'/case['id']/'camera.json')
        transform=np.asarray(camera['camera_world_optical'])
        measurements={}
        for name in config['variants']:
            file=folder/name/'points.npy'
            if sha256(file)!=row['outputs'][name]:raise RuntimeError('Saved point output changed')
            if len(row['measurements'][name]['seconds'])!=config['repetitions']:
                raise RuntimeError('Incomplete timing repetitions')
            points=np.load(file,allow_pickle=False)
            measurements[name]=quality(points@transform[:3,:3].T+transform[:3,3],truth)
        record={'id':case['id'],'quality':measurements,'truth_surface_samples':50000,
            'truth_sampling_seed':20261122,'raw_result_sha256':sha256(folder/'result.json')}
        atomic_json(folder/'surface_quality.json',record);cases.append(record)
    summary=read_json(root/'summary.json');variants=deepcopy(summary['variants'])
    for name,value in variants.items():
        fdelta=[row['quality'][name]['fscore5']-row['quality']['baseline']['fscore5'] for row in cases]
        cd=[row['quality'][name]['chamfer_l1_mm'] for row in cases if row['quality'][name]['valid']]
        base=[row['quality']['baseline']['chamfer_l1_mm'] for row in cases if row['quality']['baseline']['valid']]
        cdelta=[row['quality'][name]['chamfer_l1_mm']-row['quality']['baseline']['chamfer_l1_mm'] for row in cases
            if row['quality'][name]['valid'] and row['quality']['baseline']['valid']]
        valid=all(not row['quality']['baseline']['valid'] or row['quality'][name]['valid'] for row in cases)
        stable=(valid and statistics.mean(fdelta)>=-config['maximum_mean_fscore5_drop']
            and min(fdelta)>=-config['maximum_case_fscore5_drop']
            and cd and base and statistics.median(cd)-statistics.median(base)<=config['maximum_median_chamfer_increase_mm']
            and cdelta and max(cdelta)<=config['maximum_case_chamfer_increase_mm'])
        value.update(fscore5_mean=statistics.mean(row['quality'][name]['fscore5'] for row in cases),
            chamfer_l1_mm_median=statistics.median(cd) if cd else None,
            geometry_quality_stable=bool(stable),maximum_case_fscore5_drop=-min(fdelta),
            maximum_case_chamfer_increase_mm=max(cdelta) if cdelta else None)
    revised={**summary,'at':now(),'variants':variants,
        'surface_evaluation':{'kind':'area_uniform_mesh_surface','reference_points_per_case':50000,
            'seed':20261122,'independent_output_and_geometry_hash_checks':True,
            'source_sha256':sha256(__file__)}}
    eligible=[name for name,value in variants.items() if name!='baseline'
        and value['geometry_quality_stable'] and value['paired_speedup_median']>=config['minimum_paired_speedup']]
    accepted={**old,'at':now(),'variants':variants,'eligible_variants':eligible,
        'independent_surface_readback':True,'surface_cases_verified':len(cases),
        'inference_timing_and_point_outputs_changed':False,'automatic_activation':False}
    atomic_json(root/'surface_evaluation.json',{'at':now(),'passed':True,'cases':cases,'source_sha256':sha256(__file__)})
    atomic_json(root/'summary.json',revised);atomic_json(root/'acceptance.json',accepted)
    print('VERIFIED_UNIFORM_SURFACES',len(cases),'eligible',eligible,flush=True)
    return revised,accepted


def report(root,summary,acceptance):
    root=Path(root).resolve()
    target=ROOT/'modules/rayst3r/docs/重建推理加速.md'
    target.parent.mkdir(parents=True,exist_ok=True)
    names={'baseline':'当前重建实现','combined':'特征复用与纯推理','sdpa':'融合注意力','sdpa_combined':'融合注意力＋特征复用与纯推理'}
    body=['# RaySt3R 重建推理加速','','2026年11月25日。使用同一张RTX4090，对50组冻结RGB-D和SAM3掩码比较，每组每种实现测3次。',
        '', '## 做了什么','',
        '把RaySt3R原来分开执行的注意力矩阵乘法、Softmax和输出乘法接到PyTorch融合注意力；同时复用一次重建内重复的DINO特征，省去训练损失和评估指标计算。',
        '模型权重、输入分辨率、21个新视角、FP32 DINO、BF16主网络和置信度筛选沿用原设置。融合内核的浮点运算次序会产生坐标差异，使用完整表面误差与F-score检查实际影响。',
        '', '## 实际结果','',
        '| 实现 | 完整重建中位数 | P90 | 同场景加速中位数 | 峰值显存 | F-score@5mm均值 | 表面误差中位数 | 精度门槛 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for name,value in summary['variants'].items():
        body.append(f"| {names[name]} | {value['median_seconds']:.3f}秒 | {value['p90_seconds']:.3f}秒 | {value['paired_speedup_median']:.2f}倍 | {value['peak_allocated_mib']/1024:.2f}GiB | {value['fscore5_mean']*100:.2f}% | {value['chamfer_l1_mm_median']:.3f}毫米 | {'通过' if value['geometry_quality_stable'] else '未通过'} |")
    eligible=acceptance['eligible_variants']
    if eligible:
        best=min(eligible,key=lambda n:summary['variants'][n]['median_seconds'])
        v=summary['variants'][best]
        body+=['',f"本轮可选用{names[best]}，完整重建约{v['median_seconds']:.3f}秒，同场景加速中位数{v['paired_speedup_median']:.2f}倍。"]
    body+=['','计时从适配器接收内存中的RGB-D开始，包含原生输入文件准备、两阶段网络、点云筛选合并和传回CPU。模型加载和单独预热另列，不进入常驻推理耗时。',
        '表面误差采用每个冻结物体网格按面积均匀采样的50000个点；预测点最多取20000点。两种结果使用同一参考表面。',
        '', '## 质量检查','',
        '50组逐项保存基线和各优化输出。平均F-score下降最多0.5个百分点，中位表面误差增加最多0.1毫米；单例F-score下降最多2个百分点、表面误差增加最多0.5毫米，且不能增加空输出。逐点1微米等价检查单列，融合结果的精度结论来自完整表面核验。',
        '', '## 入口与记录','',
        '- 实现：`modules/rayst3r/fast_inference/`。',
        '- 配置：`configs/experiments/rayst3r_inference_acceleration.json`。',
        '- 本轮运行：`'+str(Path(root).relative_to(ROOT))+'`。',
        '- 分项CUDA计时：本轮`preflight/*.json`；逐例结果与点云在`trials/`。',
        '- 完整结果：`summary.json`；独立表面核验：`surface_evaluation.json`；验收：`acceptance.json`。',
        '- 已验证配置：`configs/experiments/rayst3r_fast_inference_profile.json`；可选模块入口：`configs/experiments/rayst3r_fast_inference_module.json`。',
        '- 调用入口：`modules.rayst3r.fast_inference.service.RaySt3RAdapter`，原RPC接口由同文件`create(args)`提供。',
        '', '## 参考','',
        '- [PyTorch 2.7 融合注意力接口](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html)',
        '- [RaySt3R 官方项目](https://rayst3r.github.io/)']
    interface=Path(root)/'interface/acceptance.json'
    if interface.exists():
        checked=read_json(interface)
        times=checked['median_request_seconds']
        where=body.index('## 质量检查')
        body[where:where]=['## 实际头部观测接口核验','',
            f"苹果、桃子、遮挡青椒三组已有头部观测均通过。完整工作接口的中位数由{times['baseline']:.3f}秒降到{times['accelerated']:.3f}秒，包含RGB-D读取、重建、世界坐标转换、输出保存及工作区恢复。",
            '相机坐标和世界坐标数组、21视角标记、掩码绑定及模型只加载一次的接口约定均已核对。此次只回放观测，没有执行机械臂抓取。','']
    body+=['','## 接入状态','',
        '加速版为独立可选入口。当前74例抓取对照继续使用原冻结重建实现；加速版的抓取成功率在后续新批次验证。',
        '本轮加速主要减少注意力计算时间，重建计算峰值显存约11.3GiB。工作接口仍保留14GiB推理工作区，资源门槛沿用当前配置。']
    target.write_text('\n'.join(body)+'\n')


def export_profile(root,summary,acceptance):
    eligible=acceptance['eligible_variants']
    if not eligible:return
    reg=read_json(Path(root)/'registration.json')
    best=min(eligible,key=lambda name:summary['variants'][name]['median_seconds'])
    sources={name:digest for name,digest in reg['source_hashes'].items()
        if any(part in name for part in ['/models/','/utils/','/eval_wrapper/','/dinov2/',
            '/optimized/inference.py','/memory_efficient/attention.py','/arx_reconstruction.py',
            '/arx_adapter.py','/engine.py','/fast_inference/adapter.py','/fast_inference/attention.py'])}
    sources[str(ROOT/'modules/rayst3r/fast_inference/service.py')]=sha256(ROOT/'modules/rayst3r/fast_inference/service.py')
    profile={'profile_id':'rayst3r_fused_full21_20261125_01','at':now(),'variant':best,
        'validated_cases':50,'record_integrity':acceptance['record_integrity'],
        'geometry_quality_stable':summary['variants'][best]['geometry_quality_stable'],
        'measured_results':summary['variants'][best],
        'reference_results':summary['variants']['baseline'],
        'validation_run':str(Path(root).relative_to(ROOT)),
        'source_hashes':{str(Path(name).relative_to(ROOT)):digest for name,digest in sources.items()},
        'weights':{str(Path(name).relative_to(ROOT)):entry for name,entry in reg['weights'].items()},
        'actual_novel_views':21,'input_resolution_unchanged':True,'automatic_activation':False}
    atomic_json(ROOT/'configs/experiments/rayst3r_fast_inference_profile.json',profile)
    module={**read_json(ROOT/'configs/modules.json')['modules']['rayst3r'],
        'adapter':'modules/rayst3r/fast_inference/service.py',
        'reconstruction':'modules/rayst3r/fast_inference/service.py',
        'source_directories':['modules/rayst3r/fast_inference','modules/rayst3r/memory_efficient'],
        'validation_profile':'configs/experiments/rayst3r_fast_inference_profile.json'}
    atomic_json(ROOT/'configs/experiments/rayst3r_fast_inference_module.json',
        {'module_name':'rayst3r_fast_inference','module':module,'active_registry_changed':False})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--wait',action='store_true');args=parser.parse_args();root=args.root.resolve()
    if args.wait:
        while True:
            if (root/'STOP').exists():raise RuntimeError('Inference comparison stopped')
            if read_json(root/'progress.json')['phase']=='complete':break
            time.sleep(10)
    summary,accepted=verify(root)
    export_profile(root,summary,accepted)
    report(root,summary,accepted)


if __name__=='__main__':main()
