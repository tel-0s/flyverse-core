"""One sequential house job: capture correctness, traces, timing, fixed-seed trajectories and room controls."""
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
    rel=f'out/{out.name}'
    commands=[f'python scripts/compass_driver_cuda_check.py --out {rel}/cuda_checks.json',
              'FLYVERSE_CUDA_TESTS=1 python -m pytest tests/test_cuda.py -q -p no:cacheprovider',
              f'python scripts/compass_driver_profile.py --native --out {rel}/profile_native.json',
              f'python scripts/compass_driver_profile.py --native --module-eager --out {rel}/profile_eager.json',
              f'python scripts/compass_driver_trace.py --out {rel}/trace']
    commands += [f'python scripts/probe_compass_driver.py --mode instrumented --seed {seed} --out {rel}/instrumented_s{seed}' for seed in range(10,16)]
    commands += [f'python scripts/compass_driver_room.py --mode instrumented --seed 10 --scheduler {scheduler} --out {rel}/{name}'
                 for scheduler,name in (('eager','room_eager_a'),('eager','room_eager_b'),('captured','room_instrumented'))]
    commands += [f'python scripts/compass_driver_room.py --mode {mode} --seed 10 --native --out {rel}/room_native_{mode}' for mode in ('raw','instrumented')]
    log=f'{rel}/run.txt'
    job=f"mkdir -p {rel} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && ( "+' && '.join(commands)+f' ) > {log} 2>&1 && tail -8 {log}'
    command='python scripts/cluster_run.py --target house --name compass-capture-retry --minutes 30 --arm-block fam '+shlex.quote(job)+f' --fetch {rel}/ 2>&1 | tee {rel}/client_stdout.txt'
    (out/'batch.sh').write_text('#!/bin/bash\nset -o pipefail\n'+command+'\n',encoding='utf-8',newline='\n')
    paths=[p for p in (ROOT/'flyverse').rglob('*') if p.suffix in ('.py','.cu','.metal','.csv')]
    paths+=list((ROOT/'scripts').glob('*compass_driver*.py'))+[ROOT/'docs/audits/compass_standin.md',ROOT/'tests/test_cuda.py']
    record=dict(stamped_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                status='frozen before capture validation; one sequential job to avoid contention from this experiment',commands=commands,
                rules='compass_standin.md capture declaration; no law/gain change; report all equality failures and room repeat control',
                source_sha256_lf={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths},
                batch_sha256=hashlib.sha256((out/'batch.sh').read_bytes()).hexdigest())
    (out/'predeclared.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('froze one sequential job with',len(commands),'commands')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',default='out/compass_standin_r3')
    a=ap.parse_args();build(a.out)
