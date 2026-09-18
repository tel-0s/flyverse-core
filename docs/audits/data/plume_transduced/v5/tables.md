Means +/- across-run sample SD (six independent runs per arm).

| Arm | Fed >=1 s | Feeding s | Final energy | Minimum fruit-surface distance m | Goal offset deg | Airborne s |
|---|---:|---:|---:|---:|---:|---:|
| full | 6/6 | 14.81 +/- 0.31 | 0.841 +/- 0.040 | -0.016 +/- 0.028 | 24.5 +/- 6.6 | 0.44 +/- 0.46 |
| transduced | 1/6 | 2.4 +/- 5.9 | 0.13 +/- 0.32 | 0.10 +/- 0.12 | 61.3 +/- 6.3 | 0.78 +/- 0.51 |
| goal_only | 3/6 | 7.4 +/- 8.2 | 0.43 +/- 0.47 | 0.07 +/- 0.12 | 21 +/- 19 | 0.60 +/- 0.46 |

| Arm | ORN L Hz | ORN R Hz | ORN L-R Hz | Temporal SD of ORN L-R Hz | DNa02 target Hz | DNa02 measured L-R Hz | Mean absolute tracking error Hz |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 5.81 +/- 0.93 | 5.79 +/- 0.92 | 0.023 +/- 0.016 | 0.275 +/- 0.024 | -0.34 +/- 0.55 | -0.36 +/- 0.55 | 0.80 +/- 0.36 |
| transduced | 3.0 +/- 2.3 | 3.0 +/- 2.3 | 0.017 +/- 0.027 | 0.184 +/- 0.070 | 0.34 +/- 0.35 | 0.40 +/- 0.39 | 3.4 +/- 1.5 |
| goal_only | 3.9 +/- 2.7 | 3.9 +/- 2.7 | 0.006 +/- 0.035 | 0.216 +/- 0.087 | 0.0 +/- 1.6 | -0.03 +/- 0.36 | 1.11 +/- 0.71 |

All trace-derived quantities use the same 600 post-frame samples, 0.1-60 s, without a behavior mask.
ORN rates are post-frame; target/plume state used frame-start rates; DNa02 motor rates are the post-frame readout.
Temporal SD includes changing stimuli and behavior; it is not a stationary noise estimate or an SNR.
Tracking error is the sampled target-minus-measured magnitude, not a causal lag or servo-gain estimate.
Minimum fruit distance is distance to the fruit surface, including the fly's height.
first_feed_s is first contact at any frame; blank means censored at 60 s. Contact-time statistics condition on uncensored runs.
The >=1 s indicator uses cumulative feeding duration. Goal offset is commanded-turn magnitude, not true-bearing error.
