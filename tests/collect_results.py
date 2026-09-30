#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, io, json, os, platform, re, sys, tarfile, zipfile
from datetime import datetime, timezone
from pathlib import Path

TEXT_SUFFIXES = {'.txt', '.log', '.json', '.md', '.cmake'}
SKIP_SUFFIXES = {'.f32', '.u8', '.npz', '.mogg', '.pt', '.pth', '.safetensors'}
SENSITIVE_LINE = re.compile(r'(Serial Number \(system\)|Hardware UUID|Provisioning UDID|Activation Lock Status):', re.I)
UUID = re.compile(r'\b[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b')


def sanitize_text(text: str) -> str:
    home = str(Path.home())
    user = os.environ.get('USER') or os.environ.get('USERNAME') or ''
    lines = [line for line in text.splitlines() if not SENSITIVE_LINE.search(line)]
    text = '\n'.join(lines) + ('\n' if text.endswith('\n') else '')
    if home and home != '/':
        text = text.replace(home, '$HOME')
    if user:
        text = re.sub(rf'(?<=/Users/){re.escape(user)}(?=/|\b)', '$USER', text)
        text = re.sub(rf'(?<=/home/){re.escape(user)}(?=/|\b)', '$USER', text)
    return UUID.sub('<redacted-uuid>', text)


def payload(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name == 'CMakeCache.txt':
        try:
            return sanitize_text(raw.decode('utf-8', errors='replace')).encode('utf-8')
        except Exception:
            pass
    return raw


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def keep(path: Path) -> bool:
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    # Keep small diagnostic material only. Native parity arrays/checkpoints stay local.
    return path.stat().st_size <= 8 * 1024 * 1024


def main() -> None:
    p = argparse.ArgumentParser(description='Bundle sanitized MoGe validation logs/results for return to the developer.')
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--output', default='moge-test-results.zip', help='Output .zip, .tar.gz, or .tgz archive')
    args = p.parse_args()
    root = Path(args.root).resolve(); output = Path(args.output).resolve()
    candidates = [root/'test-results', root/'README.md', root/'tests'/'README.md']
    for name in ('build-metal', 'build-vulkan'):
        candidates += [root/name/'CMakeCache.txt']
    files: list[Path] = []
    for c in candidates:
        if c.is_dir(): files.extend(x for x in c.rglob('*') if x.is_file() and keep(x))
        elif c.is_file() and keep(c): files.append(c)
    files = sorted(set(files))
    contents = {f: payload(f) for f in files}
    manifest = {
        'schema': 2,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'platform': platform.platform(),
        'sanitized': True,
        'files': {str(f.relative_to(root)): {'bytes': len(contents[f]), 'sha256': sha256_bytes(contents[f])} for f in files},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
    lower = output.name.lower()
    if lower.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(output, 'w:gz') as t:
            for f in files:
                data = contents[f]
                info = tarfile.TarInfo(str(f.relative_to(root)))
                info.size = len(data)
                info.mtime = 0
                t.addfile(info, io.BytesIO(data))
            info = tarfile.TarInfo('manifest.json')
            info.size = len(manifest_bytes)
            info.mtime = 0
            t.addfile(info, io.BytesIO(manifest_bytes))
    elif lower.endswith('.zip'):
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in files:
                z.writestr(str(f.relative_to(root)), contents[f])
            z.writestr('manifest.json', manifest_bytes)
    else:
        p.error('--output must end in .zip, .tar.gz, or .tgz')
    print(output)

if __name__ == '__main__': main()
