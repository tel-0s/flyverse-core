"""Plot the cx8 follow traces and their eligibility, without fitting or selecting a protocol.

    python scripts/cx8_plot.py --runs out/cx8r --out out/cx8r/analysis/follow.png

All six seeds are shown. Gray traces fail the predeclared >=50% confinement gate; a moving centre on those
traces cannot establish following. The lower raster shows confined frames per seed, not a rate or probability.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'scripts'))
from probe_compass_room import bump_frames


def plot(runs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(12,6), sharex=True, height_ratios=[2,1], layout='constrained')
    for col, arm in enumerate(('HG','HGV','HGV-')):
        eligible = 0
        grid = []
        for seed in range(6):
            path = Path(runs)/f'{arm}_s{seed}.json'
            row = json.loads(path.read_text(encoding='utf-8'))[0]
            with np.load(path.parent/Path(row['ledger_npz']).name) as z:
                b = bump_frames(z['epg'], z['wedge_of'])
                turn = (z['t'] >= 3.5) & (z['t'] < 6.5)
                t = z['t'][turn]-3.5
                y = np.unwrap(b['centre'][turn]*2*np.pi/16)*16/(2*np.pi)
                y -= y[0]
                keep = b['confined'][turn].mean() >= .5
                eligible += int(keep)
                axes[0,col].plot(t,y,color=f'C{seed}' if keep else '0.65', alpha=.8, lw=1,
                                 label=f'seed {seed}' + ('' if keep else ' (excluded)'))
                grid.append(b['confined'][turn])
        if arm != 'HG':
            axes[0,col].plot(t, (4 if arm=='HGV' else -4)*t,'k--',lw=1.2,label='ideal sign control')
        axes[0,col].set_title(f'{arm}: {eligible}/6 eligible')
        axes[0,col].set_ylabel('Centre displacement (wedges)')
        axes[0,col].grid(alpha=.2)
        axes[0,col].legend(fontsize=6, loc='best')
        axes[1,col].imshow(grid, origin='lower',aspect='auto',interpolation='nearest',
                           extent=(0,3,-.5,5.5),cmap='Greys',vmin=0,vmax=1)
        axes[1,col].set_yticks(range(6))
        axes[1,col].set_ylabel('Seed (black = confined)')
        axes[1,col].set_xlabel('Time since prescribed turn onset (s)')
    fig.suptitle('Round 7: prescribed turn; all seeds shown, gray centres fail eligibility')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=180)
    plt.close(fig)
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', default='out/cx8r')
    p.add_argument('--out', default='out/cx8r/analysis/follow.png')
    a=p.parse_args();plot(a.runs,a.out)
