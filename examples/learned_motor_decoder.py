"""Train a named motor boundary without changing connectome weights.

CPU smoke: python examples/learned_motor_decoder.py --mode smoke
Supervision: --mode supervised --recording out/teacher.npz
RL (needs graph cache): --mode rl --device cuda --batch 16 --epochs 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from flyverse.modules import MotorDecoder

FEATURES = ("fwd_dn", "back_dn", "turn_L", "turn_R")


def make_decoder():
    return MotorDecoder(torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Tanh(),
                                            torch.nn.Linear(16, 2), torch.nn.Tanh()), features=FEATURES)


def fit(decoder, x, targets, epochs):
    """Recorder motor fields retain physical Hz; normalized action targets are [-1, 1]."""
    decoder.reset(len(x), x.device)
    optimizer = torch.optim.Adam(decoder.net.parameters(), lr=.01)
    first = None
    for _ in range(epochs):
        prediction = decoder.forward(x)["command"]
        loss = torch.nn.functional.mse_loss(prediction, targets)
        first = float(loss.detach()) if first is None else first
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {"initial_mse": first, "final_mse": float(loss.detach()), "examples": len(x)}


def reinforcement(decoder, args):
    from flyverse.env import EnvParams, FlyRoomEnv
    # The same object is trained and attached: every provenance call hashes current weights.
    env = FlyRoomEnv(args.batch, EnvParams(modules_attached=lambda e: [decoder],
                                          motor_decoder=decoder.name, episode_s=.5, seed=args.seed), device=args.device)
    optimizer = torch.optim.Adam(decoder.net.parameters(), lr=.001)
    rewards = []
    for _ in range(args.epochs):
        env.reset()
        log_prob, earned = [], []
        for _ in range(env.max_frames):
            mean = decoder.action(env.fb.motor())
            distribution = torch.distributions.Normal(mean, .2)
            action = distribution.sample()
            # Policy-gradient training uses the body's reward; the brain stays an untrained environment.
            _, reward, _, _ = env.step(action)
            log_prob.append(distribution.log_prob(action).sum(-1))
            earned.append(torch.as_tensor(reward, device=env.fb.device))
        returns, tail = [], torch.zeros(args.batch, device=env.fb.device)
        for reward in reversed(earned):
            tail = reward + .99 * tail
            returns.append(tail)
        returns = torch.stack(list(reversed(returns)))
        advantages = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-6)
        loss = -(torch.stack(log_prob) * advantages).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        rewards.append(float(torch.stack(earned).sum(0).mean()))
    return {"episode_rewards": rewards, "provenance": env.provenance()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("smoke", "supervised", "rl"), default="smoke")
    ap.add_argument("--recording", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("out/learned_motor_decoder.pt"))
    args = ap.parse_args(argv)
    if args.epochs < 1:
        ap.error("epochs must be positive")
    torch.manual_seed(args.seed)
    decoder = make_decoder()
    if args.mode == "rl":
        report = reinforcement(decoder, args)
    elif args.mode == "supervised":
        from flyverse.interp.common import Recording
        if args.recording is None:
            ap.error("supervised mode requires --recording")
        rec = Recording.load(args.recording)
        # Collect with Recorder.capture(fb, motor=fb.motor()), then store teacher actions
        # as recording.motor['teacher_forward'] and ['teacher_yaw'] before Recording.save().
        x = np.stack([rec.motor[k] for k in FEATURES], axis=-1).reshape(-1, 4) / decoder.rate_scale
        y = np.stack([rec.motor[k] for k in ("teacher_forward", "teacher_yaw")], axis=-1).reshape(-1, 2)
        report = fit(decoder, torch.tensor(x, device=args.device), torch.tensor(y, device=args.device), args.epochs)
        report["teacher_recording"] = str(args.recording)
    else:
        x = torch.rand(128, 4, device=args.device)
        # Deliberately artificial targets: a software smoke check, not a behavioural result.
        y = torch.stack([x[:, 0] - x[:, 1], x[:, 2] - x[:, 3]], dim=1)
        report = fit(decoder, x, y, args.epochs)
        report["teacher"] = "synthetic software check; no claim about fly behaviour"
    report.update(mode=args.mode, seed=args.seed, module=decoder.describe())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"module": decoder.describe(), "state": decoder.state_dict(), "features": FEATURES}, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("module", "provenance")}, indent=2))
    print(f"checkpoint: {args.out}")


if __name__ == "__main__":
    main()
