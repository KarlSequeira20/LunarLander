# Teaching a rocket to land with reward shaping

Trained a PPO agent (Stable-Baselines3, Gymnasium LunarLander-v3), then
wrote custom reward wrappers, deliberately caused reward hacking, and fixed it.

## Results (50 test landings each, wrapper off)

| Model | Mean score | Solved (200+) | Crashes | Timeouts |
|---|---|---|---|---|
| Baseline PPO, 1M steps | 142.5 | 18 / 50 | 7 | 2 |
| Time fee, 0.1 per frame | 195.8 | 37 / 50 | 4 | 2 |
| +1 per frame bonus (hack) | -41.7 | 0 / 50 | 15 | 32 |
| Improvement-based fix | 178.1 | 29 / 50 | 5 | 2 |

One training run per version, so treat the ordering as evidence, not proof.

## What I found

A "+1 per frame while upright and slow" bonus looked reasonable, but the
rocket learned to hover until the 1,000 frame limit instead of landing.
Paying for improvement (potential-based shaping) plus a small time fee
removed the exploit.

## Run it

    pip install "gymnasium[box2d]" stable-baselines3 moviepy
    python train.py
    python evaluate_detail.py ppo_lunar_baseline 50
    python eval_all.py

Demo site: run `python3 -m http.server 8000` and open http://localhost:8000
