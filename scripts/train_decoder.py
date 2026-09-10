"""Train a decoder from descending-neuron activity to walking commands with evolution strategies.

Policy: action = tanh(W @ obs + b), obs = descending-neuron rates (1,314) -> 2 outputs. Trained with
OpenAI-style ES (antithetic Gaussian perturbations, rank-shaped fitness) using the batched env: each
of the B flies runs one perturbed policy for an episode, so a generation costs one episode of wall time
(~8 s of brain time ~ 30 s wall for B=32 on the RTX 4090).

    python scripts/train_decoder.py --batch 64 --generations 40 --episode-s 5 [--spawn-radius 0.1]
    python scripts/train_decoder.py --eval out/decoder.npz          # evaluate + compare with oracle/random

Writes out/decoder.npz (W, b) and out/train_log.csv. PufferLib does not build on this Windows box; the
env is vectorised in the PufferLib style, so PuffeRL/PPO is a drop-in on Linux (docs/NOTES.md).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import env  # noqa: E402


def rollout(e: env.FlyRoomEnv, policies, same_start=False):
    """policies: (B, n_obs+1, 2) per-fly linear policies. Returns per-fly returns and end-of-episode info."""
    obs = e.reset(same_start=same_start)
    B = e.B
    ret = np.zeros(B)
    tasted = np.zeros(B)
    while True:
        x = np.concatenate([obs, np.ones((B, 1), np.float32)], axis=1)               # bias
        act = np.tanh(np.einsum("bi,bio->bo", x, policies))
        obs, r, done, info = e.step(act)
        ret += r
        tasted += info["tasting"]
        if done.all():
            break
    return ret, {"dist_cm": info["dist_cm"], "tasted_frames": tasted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--episode-s", type=float, default=6.0)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--eval", type=str, default="")
    ap.add_argument("--init", type=str, default="", help="resume from a saved decoder (out/decoder.npz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spawn-radius", type=float, default=0.0, help="curriculum: spawn within this many m of a fruit")
    args = ap.parse_args()
    e = env.FlyRoomEnv(batch=args.batch, params=env.EnvParams(episode_s=args.episode_s, seed=args.seed, spawn_radius=args.spawn_radius))
    rng = np.random.default_rng(args.seed)
    n_in = e.n_obs + 1
    os.makedirs("out", exist_ok=True)

    if args.eval:
        d = np.load(args.eval)
        theta = np.concatenate([d["W"], d["b"][None]], axis=0)
        for name in ["decoder", "random", "oracle", "still"]:
            rets = []
            for rep in range(3):
                if name == "decoder":
                    ret, info = rollout(e, np.repeat(theta[None], e.B, axis=0))
                elif name == "random":
                    ret, info = rollout(e, rng.normal(0, 0.05, (e.B, n_in, 2)))
                elif name == "still":
                    ret, info = rollout(e, np.zeros((e.B, n_in, 2)))
                else:
                    obs = e.reset(); ret = np.zeros(e.B); tasted = np.zeros(e.B)
                    while True:
                        obs, r, done, info = e.step(e.oracle_action()); ret += r; tasted += info["tasting"]
                        if done.all():
                            break
                    info = {"dist_cm": info["dist_cm"], "tasted_frames": tasted}
                rets.append(ret)
            rets = np.concatenate(rets)
            print(f"{name:8s} return {rets.mean():7.2f} +- {rets.std():.2f}   final dist {info['dist_cm'].mean():.1f} cm   tasted frames {info['tasted_frames'].mean():.0f}")
        return

    theta = np.zeros((n_in, 2))
    theta[-1, 0] = 0.5          # start with a mild forward drive
    if args.init:
        d = np.load(args.init)
        theta = np.concatenate([d["W"], d["b"][None]], axis=0)
        print("resumed from", args.init)
    log = open("out/train_log.csv", "a" if args.init else "w", newline="")
    wr = csv.writer(log); wr.writerow(["gen", "mean_return", "best_return", "mean_final_dist_cm", "tasted_frames", "wall_s"])
    half = args.batch // 2
    for gen in range(args.generations):
        t0 = time.time()
        eps = rng.normal(0, 1, (half, n_in, 2))
        eps = np.concatenate([eps, -eps], axis=0)                       # antithetic
        pol = theta[None] + args.sigma * eps
        ret, info = rollout(e, pol, same_start=True)                     # common random numbers
        # rank-shaped fitness
        ranks = np.empty(len(ret)); ranks[np.argsort(ret)] = np.arange(len(ret))
        fit = ranks / (len(ret) - 1) - 0.5
        grad = (fit[:, None, None] * eps).sum(0) / (len(ret) * args.sigma)
        theta = theta + args.lr * grad
        theta *= 0.999                                                   # mild weight decay
        wall = time.time() - t0
        wr.writerow([gen, ret.mean(), ret.max(), info["dist_cm"].mean(), info["tasted_frames"].mean(), wall]); log.flush()
        print(f"gen {gen:3d}  return mean {ret.mean():7.2f} best {ret.max():7.2f}  final dist {info['dist_cm'].mean():5.1f} cm  tasted {info['tasted_frames'].mean():4.0f} frames  |theta| {np.abs(theta).mean():.4f}  {wall:.0f}s")
        np.savez("out/decoder.npz", W=theta[:-1], b=theta[-1])
    log.close()


if __name__ == "__main__":
    main()
