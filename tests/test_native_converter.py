import subprocess
from pathlib import Path
import shutil
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_native_pt_reader_handles_torch_checkpoint(tmp_path):
    cxx = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if not cxx:
        return
    ckpt = tmp_path / 'model.pt'
    torch.save({
        'model_config': {'encoder': {'backbone': 'synthetic', 'dim_out': 64, 'intermediate_layers': [0, 1]}, 'num_tokens_range': [12, 36]},
        'model': {
            'encoder.backbone.cls_token': torch.zeros(1, 1, 64),
            'weight': torch.arange(64, dtype=torch.float32).reshape(8, 8),
        },
    }, ckpt)
    src = tmp_path / 'probe.cpp'
    src.write_text(r'''
#include "pt_checkpoint.hpp"
#include <iostream>
int main(int argc,char**argv){
  moge_pt::Checkpoint c(argv[1]);
  auto *d=std::get_if<moge_pt::Value::Dict>(&c.config.v);
  if(!d || !d->count("encoder") || c.tensors.size()!=2) return 2;
  auto it=c.tensors.find("weight"); if(it==c.tensors.end()) return 3;
  auto raw=c.storage(it->second.storage); if(raw.size()!=64*4) return 4;
  std::cout << "pt_checkpoint=ok\n"; return 0;
}
''')
    exe = tmp_path / 'probe'
    subprocess.run([cxx, '-std=c++17', '-O0', '-I', str(ROOT/'tools'), str(src), '-o', str(exe)], check=True)
    out = subprocess.check_output([str(exe), str(ckpt)], text=True)
    assert 'pt_checkpoint=ok' in out


def test_converter_public_cli_is_simple():
    src=(ROOT/'tools/moge-convert.cpp').read_text()
    assert 'moge-convert MODEL.pt [--quant q8|f16|f32] [--output FILE.moge]' in src
    assert 'std::string quant="q8"' in src
    assert '"v"+std::to_string(ver)+"_"+a.quant+".moge"' in src
    public = src[src.index('const char * usage_text()'):]
    assert '--config' not in public
