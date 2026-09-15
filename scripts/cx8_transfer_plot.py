"""Plot every cx8t run and its operating state; input is the checked analysis/runs.csv.

    python scripts/cx8_transfer_plot.py --runs out/cx8t/analysis/runs.csv --out out/cx8t/analysis/transfer.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def plot(runs,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df=pd.read_csv(runs,usecols=['arm','seed','checks','GLNO_LR_hz','PEN_LR_hz',
                               'PEN_L_hz_turn','PEN_R_hz_turn'])
    if len(df)!=36 or df.checks.fillna('').ne('').any():raise ValueError('need all 36 checked runs')
    df['PEN_turn_mean']=(df.PEN_L_hz_turn+df.PEN_R_hz_turn)/2
    order=['H0','HL','HR','C0','CL','CR']
    fig,ax=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for axis,key,label in zip(ax,['GLNO_LR_hz','PEN_LR_hz','PEN_turn_mean'],
                              ['GLNO L-R (Hz)','PEN L-R (Hz)','PEN (L+R)/2 during challenge (Hz)']):
        axis.axvspan(2.5,5.5,color='#eeeeee',zorder=0)
        axis.axhline(0,color='0.65',lw=.8,zorder=1)
        for j,arm in enumerate(order):
            group=df[df.arm==arm].sort_values('seed')
            if len(group)!=6:raise ValueError(arm)
            y=group[key].to_numpy()
            axis.scatter(j+np.linspace(-.14,.14,6),y,c=group.seed,cmap='tab10',vmin=0,vmax=9,s=26,zorder=3)
            axis.plot([j-.24,j+.24],[y.mean(),y.mean()],color='black',lw=1.4,zorder=4)
        axis.set_xticks(range(6),order)
        axis.set_ylabel(label)
        axis.grid(axis='y',alpha=.2)
    fig.suptitle('Direct GLNO challenge: all six seeds; black lines = means; gray = GLNO-PEN held')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=180);plt.close(fig)
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',default='out/cx8t/analysis/runs.csv')
    p.add_argument('--out',default='out/cx8t/analysis/transfer.png')
    a=p.parse_args();plot(a.runs,a.out)
