#!/usr/bin/env python3
"""Dependency-free MOGG v1 metadata/tensor inspector."""
import argparse, json, struct
from pathlib import Path

TYPES={1:'f32',2:'f16',3:'bf16',4:'i32',10:'q4_k',11:'q6_k',12:'q8_0'}
FLAGS={1:'sensitive',2:'linearized_1x1',4:'phase_convt',8:'sparse',16:'sparse_packed',32:'pixelshuffle_phase',64:'hadamard_rotated'}

def u16s(f):
    n=struct.unpack('<H',f.read(2))[0]; return f.read(n).decode()
def main(p):
    with open(p,'rb') as f:
        if f.read(8)!=b'MOGG\0\0\0\x01': raise SystemExit('not MOGG v1')
        ver,nm,nt,align,mo,ds=struct.unpack('<IIIIQQ',f.read(32)); f.seek(mo)
        meta={}
        for _ in range(nm):
            k=u16s(f); tag,n=struct.unpack('<BI',f.read(5)); b=f.read(n); q=memoryview(b); pos=0
            if tag==1: v=struct.unpack_from('<q',q)[0]
            elif tag==2: v=struct.unpack_from('<d',q)[0]
            elif tag==3:
                m=struct.unpack_from('<I',q)[0]; v=bytes(q[4:4+m]).decode()
            elif tag==4:
                m=struct.unpack_from('<I',q)[0]; v=list(struct.unpack_from('<'+'q'*m,q,4))
            elif tag==5:
                m=struct.unpack_from('<I',q)[0]; pos=4; v=[]
                for _ in range(m):
                    z=struct.unpack_from('<H',q,pos)[0]; pos+=2; v.append(bytes(q[pos:pos+z]).decode()); pos+=z
            elif tag==6: v=bool(q[0])
            else: v=f'<tag {tag}>'
            meta[k]=v
        tensors=[]
        for _ in range(nt):
            name=u16s(f); typ,nd,flags=struct.unpack('<HBB',f.read(4)); ne=struct.unpack('<4q',f.read(32)); off,size=struct.unpack('<QQ',f.read(16))
            tensors.append({'name':name,'type':TYPES.get(typ,typ),'shape':list(reversed(ne[:nd])),
                            'flags':[v for k,v in FLAGS.items() if flags&k], 'offset':off,'bytes':size})
    print(json.dumps({'version':ver,'alignment':align,'data_start':ds,'metadata':meta,'tensors':tensors},indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('file'); main(a.parse_args().file)
