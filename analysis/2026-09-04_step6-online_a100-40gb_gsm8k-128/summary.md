# Step 5 online DFlash2 tree evaluation

- Source commit: `660645efbe0cfe7913a218c577290c360cbed5d5`
- GPU: NVIDIA A100-SXM4-40GB
- PyTorch/CUDA/Transformers: 2.7.0 / 12.8 / 5.16.1
- Prompts: 128

## Online acceptance

| Method | B | Matched draft tokens | 95% CI | Full block | Tokens/s | Exact sequential |
| :----- | -: | -------------------: | -----: | ---------: | -------: | ---------------: |
| DFlash2-GreedyPath | 7 | 3.9145 | [3.8276, 4.0022] | 29.12% | 90.39 | 69/128 |
| Unary-FullMass | 8 | 4.0833 | [3.9994, 4.1666] | 25.67% | 89.79 | 67/128 |
| Unary-FullMass | 16 | 4.4729 | [4.3892, 4.5562] | 33.31% | 96.71 | 70/128 |
| Unary-FullMass | 32 | 4.7395 | [4.6595, 4.8184] | 37.91% | 100.66 | 67/128 |
| Unary-FullMass | 64 | 4.9456 | [4.8653, 5.0234] | 41.70% | 101.74 | 74/128 |
| Pairwise-MassPreserving | 8 | 4.3386 | [4.2580, 4.4164] | 29.44% | 93.94 | 70/128 |
| Pairwise-MassPreserving | 16 | 4.9057 | [4.8211, 4.9893] | 41.95% | 103.91 | 68/128 |
| Pairwise-MassPreserving | 32 | 5.2481 | [5.1658, 5.3302] | 48.54% | 109.28 | 63/128 |
| Pairwise-MassPreserving | 64 | 5.4909 | [5.4150, 5.5651] | 53.81% | 110.79 | 66/128 |

## Pairwise minus Unary

| B | Matched-token gain | 95% CI | Prompts improve/tie/hurt |
| -: | -----------------: | -----: | :----------------------- |
| 8 | +0.2554 | [+0.2146, +0.2960] | 110/7/11 |
| 16 | +0.4328 | [+0.3834, +0.4830] | 121/4/3 |
| 32 | +0.5087 | [+0.4598, +0.5577] | 122/3/3 |
| 64 | +0.5453 | [+0.4998, +0.5920] | 123/4/1 |

## Interpretation

The online methods encounter different state trajectories. Paired statistics therefore compare prompt-level averages, not decoding round N across methods. Offline values are off-policy frozen-state predictions and are included only to check whether the relative ordering survives.

**Recommendation: STRONG GO.**
