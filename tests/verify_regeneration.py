"""Regenerate CSVs in a TEMP directory, verify byte-identical to packaged originals."""
from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        result=subprocess.run([sys.executable,str(ROOT/'scripts'/'generate_datasets.py'),
                               '--output-dir',tmp],text=True,capture_output=True,cwd=ROOT)
        if result.returncode:
            print(result.stdout,result.stderr);raise SystemExit(1)
        originals=sorted((ROOT/'datasets').glob('*.csv'))
        assert len(originals)==8, f'Expected 8 datasets, saw {len(originals)}'
        for original in originals:
            generated=Path(tmp)/original.name
            assert generated.exists(),f'Missing regenerated {original.name}'
            assert digest(generated)==digest(original),f'Not reproducible: {original.name}'
            print('PASS identical SHA-256:',original.name,digest(original)[:16])
    print('All 8 CSVs regenerate byte-for-byte with the shipped seed and generator.')


if __name__=='__main__':main()
