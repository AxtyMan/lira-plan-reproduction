"""Record immutable inputs, executable sources, venv, and the numbered plan formulae."""
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    sources=json.loads((ROOT/'reports/sources.json').read_text(encoding='utf-8-sig'))
    hashes={}
    for key,entry in {**sources['resources'],**sources['provided_papers']}.items():
        path=ROOT/entry['path'] if 'path' in entry else ROOT.parent/entry['file']
        if sha(path)!=entry['sha256']:
            raise ValueError(f'Input SHA-256 mismatch for {path}')
        hashes[key]=entry['sha256']
    plan=next((ROOT.parent/'参考文献').glob('*研究基础*.docx'))
    hashes[plan.name]=sha(plan)
    with ZipFile(plan) as z:
        x=ET.fromstring(z.read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paragraphs=[''.join(n.text or '' for n in p.iter() if n.tag.endswith('}t')) for p in x.findall('.//w:p',ns)]
    equations=[p for p in paragraphs if any(p.endswith(f'（{i}）') for i in range(1,16))]
    assert len(equations)==15
    (ROOT/'docs/计划公式原文.txt').write_text('\n'.join(equations)+'\n',encoding='utf-8')
    files=[f for folder in ('src','prompts','configs','tools','tests','reports/paper-comparison','reports') for f in ((ROOT/folder).rglob('*') if folder not in ('reports','reports/paper-comparison') else (ROOT/folder).glob('*.py')) if f.is_file() and '__pycache__' not in f.parts]
    code={f.relative_to(ROOT).as_posix():sha(f) for f in files+[ROOT/'run.py',ROOT/'requirements-lock.txt']}
    assert sys.prefix!=sys.base_prefix and Path(sys.prefix).resolve()==(ROOT/'.venv').resolve()
    record=dict(verified_input_hashes=hashes,code_config_hashes=code,python=sys.version,venv=sys.prefix,
        base_python=sys.base_prefix,platform=str(sys.getwindowsversion()) if sys.platform=='win32' else platform.platform(),torch=version('torch'),numpy=version('numpy'))
    (ROOT/'reports/input-verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Verified',len(hashes),'inputs and',len(equations),'numbered equations')

if __name__=='__main__':main()
