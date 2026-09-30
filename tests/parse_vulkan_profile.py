#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, statistics
from collections import defaultdict
from pathlib import Path

OP_RE = re.compile(r'^moge_op_profile: phase="([^"]*)" op=([^ ]+) name="([^"]*)" ms=([0-9.eE+-]+) bytes=(\d+) ne=\[(.*?)\]$')
MEM_RE = re.compile(r'^moge_device_memory: tag=([^ ]+) device="([^"]*)" free_bytes=(\d+) total_bytes=(\d+) used_bytes=(\d+)$')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('log', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    a=ap.parse_args()
    ops=[]; mem=[]
    for line in a.log.read_text(errors='replace').splitlines():
        m=OP_RE.match(line)
        if m:
            ops.append({'phase':m.group(1),'op':m.group(2),'name':m.group(3),'ms':float(m.group(4)),'bytes':int(m.group(5)),'ne':[int(x) for x in m.group(6).split(',')]})
            continue
        m=MEM_RE.match(line)
        if m:
            mem.append({'tag':m.group(1),'device':m.group(2),'free_bytes':int(m.group(3)),'total_bytes':int(m.group(4)),'used_bytes':int(m.group(5))})
    by_op=defaultdict(lambda:{'count':0,'total_ms':0.0,'max_ms':0.0,'bytes':0})
    by_phase=defaultdict(lambda:{'count':0,'total_ms':0.0})
    by_phase_op=defaultdict(lambda:{'count':0,'total_ms':0.0,'max_ms':0.0,'bytes':0})
    for x in ops:
        o=by_op[x['op']]; o['count']+=1; o['total_ms']+=x['ms']; o['max_ms']=max(o['max_ms'],x['ms']); o['bytes']+=x['bytes']
        p=by_phase[x['phase']]; p['count']+=1; p['total_ms']+=x['ms']
        po=by_phase_op[(x['phase'],x['op'])]; po['count']+=1; po['total_ms']+=x['ms']; po['max_ms']=max(po['max_ms'],x['ms']); po['bytes']+=x['bytes']
    top_nodes=sorted(ops,key=lambda x:x['ms'],reverse=True)[:50]
    top_phase_ops=[{'phase':phase,'op':op,**stats} for (phase,op),stats in sorted(by_phase_op.items(),key=lambda kv:kv[1]['total_ms'],reverse=True)[:100]]
    report={
        'schema':1,
        'timing_note':'MOGE_OP_PROFILE synchronizes every graph node; timings identify hotspots but are not production end-to-end latency.',
        'nodes_profiled':len(ops),
        'profiled_node_total_ms':sum(x['ms'] for x in ops),
        'by_op':dict(sorted(by_op.items(),key=lambda kv:kv[1]['total_ms'],reverse=True)),
        'by_phase':dict(sorted(by_phase.items(),key=lambda kv:kv[1]['total_ms'],reverse=True)),
        'by_phase_op':{f'{phase}::{op}':stats for (phase,op),stats in sorted(by_phase_op.items(),key=lambda kv:kv[1]['total_ms'],reverse=True)},
        'top_phase_ops':top_phase_ops,
        'top_nodes':top_nodes,
        'device_memory_samples':mem,
    }
    if mem:
        weights=next((x for x in mem if x['tag']=='weights_loaded'),None)
        peak=max(mem,key=lambda x:x['used_bytes'])
        baseline=min(x['used_bytes'] for x in mem)
        report['device_memory']={
            'device':mem[-1]['device'],
            'total_bytes':max(x['total_bytes'] for x in mem),
            'max_observed_used_bytes':peak['used_bytes'],
            'peak_tag':peak['tag'],
            'min_observed_free_bytes':min(x['free_bytes'] for x in mem),
            'used_range_bytes':peak['used_bytes']-baseline,
            'baseline_used_bytes':baseline,
        }
        if weights:
            report['device_memory']['weights_loaded_used_bytes']=weights['used_bytes']
            report['device_memory']['peak_over_weights_bytes']=max(0,peak['used_bytes']-weights['used_bytes'])
        report['top_device_memory_samples']=sorted(mem,key=lambda x:x['used_bytes'],reverse=True)[:20]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__': main()
