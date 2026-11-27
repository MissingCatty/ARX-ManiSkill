"""Separate CUDA Graph compute from Python dispatch and host transfers."""
import time

import numpy as np
import torch


def summarize(samples):
    return {'median_ms':float(np.median(samples)),'p90_ms':float(np.percentile(samples,90)),
        'mean_ms':float(np.mean(samples)),'samples':len(samples)}


def measure(functions,graph_repeats=50,eager_repeats=10):
    graphs={};outputs={};eager={key:[] for key in functions};events={key:[] for key in functions}
    for name,fn in functions.items():
        for _ in range(3):outputs[name]=fn()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):outputs[name]=fn()
        graphs[name]=g
        for _ in range(3):g.replay()
    torch.cuda.synchronize();keys=list(functions)
    for repeat in range(eager_repeats):
        for name in keys[repeat%len(keys):]+keys[:repeat%len(keys)]:
            begin=time.perf_counter();value=functions[name]();torch.cuda.synchronize()
            eager[name].append((time.perf_counter()-begin)*1000)
    for repeat in range(graph_repeats):
        for name in keys[repeat%len(keys):]+keys[:repeat%len(keys)]:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();graphs[name].replay();b.record();events[name].append((a,b))
    torch.cuda.synchronize()
    result={name:{'graph_device':summarize([a.elapsed_time(b) for a,b in events[name]]),
        'eager_host_wall':summarize(eager[name])} for name in functions}
    # Outputs/graph owners must remain alive until all captured work is finished.
    return result


def host_roundtrip(solver,checker,targets,widths,input_valid,repeats=10):
    cpu_targets=torch.as_tensor(targets,dtype=torch.float32)
    cpu_widths=torch.as_tensor(widths,dtype=torch.float32)
    static_targets=cpu_targets.to('cuda');static_widths=cpu_widths.to('cuda')
    def call():
        answer=solver.solve(static_targets)
        collision=checker.check(answer['q'],answer['valid']&input_valid[:,None],static_widths)
        return answer,collision
    for _ in range(3):answer,collision=call()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):answer,collision=call()
    g.replay();torch.cuda.synchronize();times=[]
    for _ in range(repeats):
        begin=time.perf_counter();static_targets.copy_(cpu_targets);static_widths.copy_(cpu_widths)
        g.replay()
        q=answer['q'].cpu().numpy();valid=collision['valid_world_and_self'].cpu().numpy()
        torch.cuda.synchronize();times.append((time.perf_counter()-begin)*1000)
    return {'graph_with_host_upload_and_q_mask_download':summarize(times),
        'world_binding_excluded':True,'physical_motion_included':False}
