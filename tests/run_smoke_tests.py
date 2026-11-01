"""Run representative CLI paths. Invoke from the lab root: python tests/run_smoke_tests.py"""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    commands=[
        ['scripts/inspect_data.py','--dataset','phishing','--rows','2'],
        ['scripts/verify_lab.py'],
        ['scripts/run_classification.py','--name','_smoke_classification'],
        ['scripts/run_regression.py','--name','_smoke_regression'],
        ['scripts/run_clustering.py','--name','_smoke_clustering'],
        ['scripts/run_anomaly.py','--name','_smoke_anomaly'],
        ['scripts/compare_models.py','_smoke_classification','_smoke_classification','--name','_smoke_comparison'],
        ['scripts/leakage_demo.py'],
        ['scripts/neural_network_intro.py','--epochs','200','--name','_smoke_neural'],
    ]
    for cmd in commands:
        print('RUN:',sys.executable,*cmd,flush=True)
        proc=subprocess.run([sys.executable,*cmd],cwd=ROOT,stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,text=True)
        if proc.returncode:
            print(proc.stdout[-2500:]); raise SystemExit(f'FAILED: {cmd} (exit {proc.returncode})')
        print('PASS',cmd[0],flush=True)
    print('All representative command-line smoke tests passed.')


if __name__=='__main__':main()
