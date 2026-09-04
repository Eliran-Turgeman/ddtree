# Step 5 online DFlash2 tree evaluation

- Source commit: `660645efbe0cfe7913a218c577290c360cbed5d5`
- GPU: NVIDIA A100-SXM4-40GB
- PyTorch/CUDA/Transformers: 2.7.0 / 12.8 / 5.16.1
- Prompts: 128

## Online acceptance

| Method | B | Matched draft tokens | 95% CI | Full block | Tokens/s | Exact sequential |
| :----- | -: | -------------------: | -----: | ---------: | -------: | ---------------: |
| DFlash2-GreedyPath | 7 | 4.1984 | [4.0888, 4.3067] | 37.11% | 96.30 | 22/128 |
| Unary-FullMass | 8 | 4.3289 | [4.2281, 4.4292] | 33.74% | 94.34 | 32/128 |
| Unary-FullMass | 16 | 4.7318 | [4.6305, 4.8334] | 41.18% | 101.57 | 27/128 |
| Unary-FullMass | 32 | 4.9757 | [4.8812, 5.0674] | 45.15% | 105.36 | 25/128 |
| Unary-FullMass | 64 | 5.1900 | [5.1059, 5.2724] | 49.10% | 106.52 | 24/128 |
| Pairwise-MassPreserving | 8 | 4.6158 | [4.5117, 4.7166] | 37.69% | 99.16 | 34/128 |
| Pairwise-MassPreserving | 16 | 5.1967 | [5.0972, 5.2930] | 49.55% | 109.58 | 30/128 |
| Pairwise-MassPreserving | 32 | 5.5306 | [5.4474, 5.6111] | 56.69% | 114.70 | 29/128 |
| Pairwise-MassPreserving | 64 | 5.7498 | [5.6681, 5.8290] | 61.85% | 115.54 | 32/128 |

## Pairwise minus Unary

| B | Matched-token gain | 95% CI | Prompts improve/tie/hurt |
| -: | -----------------: | -----: | :----------------------- |
| 8 | +0.2869 | [+0.2486, +0.3252] | 117/1/10 |
| 16 | +0.4648 | [+0.4124, +0.5184] | 122/1/5 |
| 32 | +0.5549 | [+0.5095, +0.6050] | 125/1/2 |
| 64 | +0.5598 | [+0.5129, +0.6069] | 126/0/2 |

## Interpretation

The online methods encounter different state trajectories. Paired statistics therefore compare prompt-level averages, not decoding round N across methods. Offline values are off-policy frozen-state predictions and are included only to check whether the relative ordering survives.

**Recommendation: STRONG GO.**
