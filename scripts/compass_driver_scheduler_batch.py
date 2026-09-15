"""Freeze the scheduler correction, same trajectory law/seeds, and native-CUDA timing separately."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'predeclared.json').exists():raise FileExistsError('already frozen')
    rel=f'out/{out.name}';jobs=[]
    def job(commands,name):
        log=f'{rel}/{name}.txt';chain=' && '.join(commands)
        jobs.append(f"mkdir -p {rel} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && ( {chain} ) > {log} 2>&1; st=$?; tail -3 {log}; exit $st")
    for seed in range(10,16):
        job([f'python scripts/probe_compass_driver.py --mode instrumented --seed {seed} --out {rel}/instrumented_s{seed}'],f'contract_s{seed}')
    job([f'python scripts/compass_driver_profile.py {flag} --out {rel}/profile{suffix}.json' for flag,suffix in (('',''),('--native','_native'))],'profile')
    job([f'python scripts/compass_driver_cuda_check.py --out {rel}/cuda_checks.json'],'cuda_checks')
    job([f'python scripts/compass_driver_room.py --mode instrumented --seed 10 --out {rel}/room_instrumented'],'room_instrumented')
    # The same native backend on both arms. This is a timing/side-effect observation, not a tuned law.
    for mode in ('raw','instrumented'):
        job([f'python scripts/compass_driver_room.py --mode {mode} --seed 10 --native --out {rel}/room_native_{mode}'],f'room_native_{mode}')
    command='python scripts/cluster_run.py --target house --name compass-scheduler --minutes 30 --arm-block fam '+ ' '.join(shlex.quote(j) for j in jobs)+f' --fetch {rel}/ 2>&1 | tee {rel}/client_stdout.txt'
    (out/'batch.sh').write_text('#!/bin/bash\nset -o pipefail\n'+command+'\n',encoding='utf-8',newline='\n')
    paths=[p for p in (ROOT/'flyverse').rglob('*') if p.suffix in ('.py','.cu','.metal','.csv')]
    paths+=list((ROOT/'scripts').glob('*compass_driver*.py'))
    paths+=[ROOT/'docs/audits/compass_standin.md',ROOT/'scripts/probe_compass_driver.py']
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths}
    record=dict(stamped_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status='frozen before scheduler rerun',
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                jobs=jobs,source_sha256_lf=hashes,batch_sha256=hashlib.sha256((out/'batch.sh').read_bytes()).hexdigest(),
                rules='compass_standin.md scheduler correction: exact prior trajectories/room; <=10% overhead; no law or gain change',
                same_contract_seeds=list(range(10,16)),room_seconds=60,room_batch=6)
    (out/'predeclared.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('froze',len(jobs),'scheduler jobs')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',default='out/compass_standin_r2')
    a=ap.parse_args();build(a.out)
