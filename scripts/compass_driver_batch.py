"""Freeze and generate the house-only engineering validation for docs/audits/compass_standin.md."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shlex
import time

ROOT=Path(__file__).resolve().parents[1]


def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'predeclared.json').exists():raise FileExistsError('already frozen')
    rel=f'out/{out.name}';jobs=[]
    def job(commands, name):
        chain=' && '.join(commands)
        log=f'{rel}/{name}.txt'
        jobs.append(f"mkdir -p {rel} && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && ( {chain} ) > {log} 2>&1; st=$?; tail -3 {log}; exit $st")
    for seed in range(10,16):
        job([f'python scripts/probe_compass_driver.py --mode {mode} --seed {seed} --out {rel}/{mode}_s{seed}' for mode in ('raw','instrumented')],f'contract_s{seed}')
    for seed in range(3):
        job([f'python scripts/benchmark.py --sections all --draw-seed {seed} --seeds {seed} --preset {mode} '+
             ('--instrument compass ' if mode=='instrumented' else '')+f'--json {rel}/suite_{mode}_s{seed}.json' for mode in ('raw','instrumented')],f'suite_s{seed}')
    for mode in ('raw','instrumented'):
        job([f'python scripts/compass_driver_room.py --mode {mode} --seed 10 --out {rel}/room_{mode}'],f'room_{mode}')
    job([f'python scripts/compass_driver_profile.py --out {rel}/profile.json'],'profile')
    job([f'python scripts/room_demo.py --headless --seconds 1 --preset instrumented --instrument compass --brain-map --cuda-graphs --screenshot {rel}/ui.png'],'ui')
    command='python scripts/cluster_run.py --target house --name compass-standin --minutes 30 --arm-block fam '+ ' '.join(shlex.quote(j) for j in jobs)+f' --fetch {rel}/ 2>&1 | tee {rel}/client_stdout.txt'
    batch='#!/bin/bash\nset -o pipefail\n'+command+'\n'
    (out/'batch.sh').write_text(batch,encoding='utf-8',newline='\n')
    paths=[p for p in (ROOT/'flyverse').rglob('*') if p.suffix in ('.py','.cu','.metal','.csv')]
    paths += [ROOT/'scripts'/n for n in ('probe_compass_driver.py','compass_driver_room.py','compass_driver_profile.py',
                                      'compass_driver_batch.py','instrumented_benchmark.py','benchmark.py','room_demo.py')]
    paths += [ROOT/'docs/audits/compass_standin.md']
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths}
    declaration=dict(stamped_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                     commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                     status='frozen before submission',jobs=jobs,source_sha256_lf=hashes,
                     batch_sha256=hashlib.sha256((out/'batch.sh').read_bytes()).hexdigest(),
                     rules='docs/audits/compass_standin.md, House declaration; engineering validation, no adoption',
                     contract_seeds=list(range(10,16)),suite_seeds=[0,1,2],room_seconds=60,room_batch=6,
                     parameters=dict(peak_hz=50.,width_deg=35.,velocity_gain=1.,initial_phase_deg=0.))
    (out/'predeclared.json').write_text(json.dumps(declaration,indent=2)+'\n',encoding='utf-8')
    print('froze',len(jobs),'jobs in',out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',default='out/compass_standin');a=ap.parse_args();build(a.out)
