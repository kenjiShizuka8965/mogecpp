#!/usr/bin/env python3
"""Compare a trusted moge-cli output against a candidate using stdlib only.

Designed for cross-backend calibration: it reports rich numerical drift but is
report-only until thresholds are deliberately established from matched runs.
"""
from __future__ import annotations
import argparse, array, json, math, statistics, struct, sys, zlib
from pathlib import Path


def pfm(path: Path):
    with path.open('rb') as f:
        if f.readline().strip() != b'Pf': raise ValueError(f'{path}: not grayscale PFM')
        wh=f.readline().split(); w,h=map(int,wh)
        scale=float(f.readline().strip()); raw=f.read()
    if len(raw)!=w*h*4: raise ValueError(f'{path}: malformed PFM payload')
    a=array.array('f'); a.frombytes(raw)
    if (scale<0)!=(sys.byteorder=='little'): a.byteswap()
    out=array.array('f')
    for y in range(h-1,-1,-1): out.extend(a[y*w:(y+1)*w])
    return w,h,out


def ply_xyz(path: Path):
    with path.open('rb') as f:
        lines=[]
        while True:
            x=f.readline()
            if not x: raise ValueError(f'{path}: truncated PLY')
            lines.append(x)
            if x.rstrip()==b'end_header': break
        hdr=b''.join(lines).decode('ascii')
        if 'format binary_little_endian 1.0' not in hdr: raise ValueError('PLY must be binary little endian')
        n=None; props=[]; vertex=False
        for line in hdr.splitlines():
            s=line.split()
            if s[:2]==['element','vertex']: n=int(s[2]); vertex=True
            elif s and s[0]=='element' and s[1]!='vertex': vertex=False
            elif vertex and s[:2]==['property','float']: props.append(s[2])
        if n is None or props!=['x','y','z']: raise ValueError('PLY must contain only float x/y/z vertices')
        raw=f.read()
    if len(raw)!=n*12: raise ValueError('PLY payload size mismatch')
    a=array.array('f'); a.frombytes(raw)
    if sys.byteorder!='little': a.byteswap()
    return n,a


def png(path: Path):
    data=path.read_bytes()
    if data[:8]!=b'\x89PNG\r\n\x1a\n': raise ValueError(f'{path}: not PNG')
    pos=8; idat=bytearray(); w=h=ct=bd=None; interlace=None
    while pos < len(data):
        ln=struct.unpack('>I',data[pos:pos+4])[0]; typ=data[pos+4:pos+8]; body=data[pos+8:pos+8+ln]; pos += 12+ln
        if typ==b'IHDR': w,h,bd,ct,_,_,interlace=struct.unpack('>IIBBBBB',body)
        elif typ==b'IDAT': idat.extend(body)
        elif typ==b'IEND': break
    if bd!=8 or ct not in (0,2) or interlace!=0: raise ValueError(f'{path}: only non-interlaced 8-bit gray/RGB PNG supported')
    channels=1 if ct==0 else 3; stride=w*channels; raw=zlib.decompress(bytes(idat)); expected=(stride+1)*h
    if len(raw)!=expected: raise ValueError(f'{path}: PNG decompressed size mismatch')
    out=bytearray(w*h*channels); prev=bytearray(stride); off=0
    def paeth(a,b,c):
        p=a+b-c; pa=abs(p-a); pb=abs(p-b); pc=abs(p-c)
        return a if pa<=pb and pa<=pc else b if pb<=pc else c
    for y in range(h):
        ft=raw[off]; off+=1; cur=bytearray(raw[off:off+stride]); off+=stride
        for x in range(stride):
            a=cur[x-channels] if x>=channels else 0; b=prev[x]; c=prev[x-channels] if x>=channels else 0
            if ft==1: cur[x]=(cur[x]+a)&255
            elif ft==2: cur[x]=(cur[x]+b)&255
            elif ft==3: cur[x]=(cur[x]+((a+b)//2))&255
            elif ft==4: cur[x]=(cur[x]+paeth(a,b,c))&255
            elif ft!=0: raise ValueError(f'{path}: unsupported PNG filter {ft}')
        out[y*stride:(y+1)*stride]=cur; prev=cur
    return w,h,channels,out


def q(sorted_vals,p):
    if not sorted_vals: return None
    x=(len(sorted_vals)-1)*p; lo=int(x); hi=math.ceil(x); t=x-lo
    return sorted_vals[lo]*(1-t)+sorted_vals[hi]*t


def errors(ref, got, rel_floor=1e-6):
    ae=[]; re=[]; ss=rs=0.0; nonfinite=0
    for a,b in zip(ref,got):
        if not(math.isfinite(a) and math.isfinite(b)):
            if math.isfinite(a)!=math.isfinite(b) or math.isnan(a)!=math.isnan(b): nonfinite+=1
            continue
        d=abs(float(b)-float(a)); ae.append(d); re.append(d/max(abs(float(a)),rel_floor)); ss+=d*d; rs+=float(a)*float(a)
    ae.sort(); re.sort(); n=len(ae)
    return {'count':n,'nonfinite_mismatch':nonfinite,'mae':statistics.fmean(ae) if ae else None,'median_abs':q(ae,.5),
            'p95_abs':q(ae,.95),'p99_abs':q(ae,.99),'max_abs':ae[-1] if ae else None,
            'rmse':math.sqrt(ss/n) if n else None,'nrmse_l2':math.sqrt(ss/rs) if rs else None,
            'mean_rel':statistics.fmean(re) if re else None,'p95_rel':q(re,.95),'p99_rel':q(re,.99),'max_rel':re[-1] if re else None}


def scale_aligned(ref, got):
    pairs=[(float(a),float(b)) for a,b in zip(ref,got) if math.isfinite(a) and math.isfinite(b)]
    denom=sum(b*b for _,b in pairs); s=sum(a*b for a,b in pairs)/denom if denom else 1.0
    e=errors((a for a,_ in pairs),(s*b for _,b in pairs))
    return s,e


def mask_metrics(r,g):
    rb=[x>=128 for x in r]; gb=[x>=128 for x in g]
    inter=sum(x and y for x,y in zip(rb,gb)); union=sum(x or y for x,y in zip(rb,gb))
    return {'iou':inter/union if union else 1.0,'agreement':sum(x==y for x,y in zip(rb,gb))/len(rb),
            'reference_valid_fraction':sum(rb)/len(rb),'candidate_valid_fraction':sum(gb)/len(gb)}


def normal_metrics(r,g):
    vals=[]
    for i in range(0,len(r),3):
        a=[r[i+j]/127.5-1 for j in range(3)]; b=[g[i+j]/127.5-1 for j in range(3)]
        na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
        if na<1e-8 or nb<1e-8: continue
        dot=max(-1.0,min(1.0,sum(a[j]*b[j] for j in range(3))/(na*nb)))
        vals.append(math.degrees(math.acos(dot)))
    vals.sort()
    return {'count':len(vals),'angular_mean_deg':statistics.fmean(vals) if vals else None,
            'angular_median_deg':q(vals,.5),'angular_p95_deg':q(vals,.95),'angular_p99_deg':q(vals,.99),'angular_max_deg':vals[-1] if vals else None}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('reference',type=Path); ap.add_argument('candidate',type=Path); ap.add_argument('--json',type=Path)
    ap.add_argument('--enforce', action='store_true')
    ap.add_argument('--depth-rmse-max', type=float); ap.add_argument('--depth-p95-rel-max', type=float)
    ap.add_argument('--points-rmse-max', type=float); ap.add_argument('--normal-p95-deg-max', type=float)
    ap.add_argument('--mask-iou-min', type=float); ap.add_argument('--intrinsics-max-abs', type=float); ap.add_argument('--metric-scale-rel-max', type=float)
    a=ap.parse_args(); r=a.reference; c=a.candidate
    rw,rh,rd=pfm(r/'depth.pfm'); cw,ch,cd=pfm(c/'depth.pfm')
    if (rw,rh)!=(cw,ch): raise SystemExit(f'depth shape mismatch {rw}x{rh} vs {cw}x{ch}')
    rn,rp=ply_xyz(r/'points.ply'); cn,cp=ply_xyz(c/'points.ply')
    if rn!=cn: raise SystemExit(f'point count mismatch {rn} vs {cn}')
    rc=json.loads((r/'camera.json').read_text()); cc=json.loads((c/'camera.json').read_text())
    s,sa=scale_aligned(rd,cd)
    report={'schema':1,'status':'report-only','shape':{'width':rw,'height':rh,'points':rn},
            'reference_backend':rc.get('backend'),'candidate_backend':cc.get('backend'),
            'depth':errors(rd,cd,1e-4),'depth_scale_alignment':s,'depth_scale_aligned_rmse':sa['rmse'],
            'depth_scale_aligned':sa,'points_xyz':errors(rp,cp,1e-4)}
    ri=rc['intrinsics']; ci=cc['intrinsics']; di=[abs(float(x)-float(y)) for x,y in zip(ri,ci)]
    rs=float(rc['metric_scale']); cs=float(cc['metric_scale'])
    report['camera']={'intrinsics_max_abs':max(di),'intrinsics_mae':statistics.fmean(di),
                      'metric_scale_reference':rs,'metric_scale_candidate':cs,'metric_scale_abs':abs(cs-rs),
                      'metric_scale_rel':abs(cs-rs)/max(abs(rs),1e-12)}
    # Optional visualization-derived metrics; absence does not invalidate primary comparison.
    if (r/'mask.png').is_file() and (c/'mask.png').is_file():
        mw,mh,mch,mr=png(r/'mask.png'); nw,nh,nch,mg=png(c/'mask.png')
        if (mw,mh,mch)==(nw,nh,nch) and mch==1:
            rb=[x>=128 for x in mr]; gb=[x>=128 for x in mg]; inter=sum(x and y for x,y in zip(rb,gb)); union=sum(x or y for x,y in zip(rb,gb))
            report['mask']=mask_metrics(mr,mg)
    if (r/'normal.png').is_file() and (c/'normal.png').is_file():
        nw,nh,nch,nr=png(r/'normal.png'); gw,gh,gch,ng=png(c/'normal.png')
        if (nw,nh,nch)==(gw,gh,gch) and nch==3: report['normal']=normal_metrics(nr,ng)

    failures=[]
    if report['depth']['nonfinite_mismatch']: failures.append('depth non-finite mismatch')
    if report['points_xyz']['nonfinite_mismatch']: failures.append('point non-finite mismatch')
    if a.enforce:
        checks=[
            ('depth.rmse', report['depth']['rmse'], a.depth_rmse_max, 'max'),
            ('depth.p95_rel', report['depth']['p95_rel'], a.depth_p95_rel_max, 'max'),
            ('points_xyz.rmse', report['points_xyz']['rmse'], a.points_rmse_max, 'max'),
            ('camera.intrinsics_max_abs', report['camera']['intrinsics_max_abs'], a.intrinsics_max_abs, 'max'),
            ('camera.metric_scale_rel', report['camera']['metric_scale_rel'], a.metric_scale_rel_max, 'max'),
        ]
        if 'normal' in report: checks.append(('normal.angular_p95_deg', report['normal']['angular_p95_deg'], a.normal_p95_deg_max, 'max'))
        if 'mask' in report: checks.append(('mask.iou', report['mask']['iou'], a.mask_iou_min, 'min'))
        for name,val,limit,kind in checks:
            if limit is None or val is None: continue
            bad=(val>limit) if kind=='max' else (val<limit)
            if bad: failures.append(f'{name}={val:.9g} violates {kind} {limit:.9g}')
    report['failures']=failures
    report['status']='fail' if failures else ('pass' if a.enforce else 'report-only')

    text=json.dumps(report,indent=2,sort_keys=True)+'\n'
    if a.json: a.json.parent.mkdir(parents=True,exist_ok=True); a.json.write_text(text)
    print(text,end='')
    return 1 if failures else 0
if __name__=='__main__': raise SystemExit(main())
