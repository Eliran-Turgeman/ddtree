# Step 5 online DFlash2 tree evaluation

- Source commit: `660645efbe0cfe7913a218c577290c360cbed5d5`
- GPU: NVIDIA A100-SXM4-40GB
- PyTorch/CUDA/Transformers: 2.7.0 / 12.8 / 5.16.1
- Prompts: 160

## Online acceptance

| Method | B | Matched draft tokens | 95% CI | Full block | Tokens/s | Exact sequential |
| :----- | -: | -------------------: | -----: | ---------: | -------: | ---------------: |
| DFlash2-GreedyPath | 7 | 2.6258 | [2.4337, 2.8265] | 9.81% | 67.62 | 46/160 |
| Unary-FullMass | 8 | 2.8625 | [2.6879, 3.0428] | 8.49% | 68.86 | 49/160 |
| Unary-FullMass | 16 | 3.1758 | [2.9962, 3.3600] | 11.51% | 74.48 | 39/160 |
| Unary-FullMass | 32 | 3.4329 | [3.2528, 3.6161] | 13.47% | 78.65 | 45/160 |
| Unary-FullMass | 64 | 3.6424 | [3.4599, 3.8253] | 15.56% | 80.29 | 48/160 |
| Pairwise-MassPreserving | 8 | 3.0313 | [2.8458, 3.2202] | 9.58% | 71.51 | 47/160 |
| Pairwise-MassPreserving | 16 | 3.4490 | [3.2571, 3.6430] | 13.99% | 79.02 | 47/160 |
| Pairwise-MassPreserving | 32 | 3.7465 | [3.5549, 3.9444] | 17.20% | 83.86 | 48/160 |
| Pairwise-MassPreserving | 64 | 4.0210 | [3.8274, 4.2160] | 20.52% | 86.36 | 49/160 |

## Pairwise minus Unary

| B | Matched-token gain | 95% CI | Prompts improve/tie/hurt |
| -: | -----------------: | -----: | :----------------------- |
| 8 | +0.1688 | [+0.1270, +0.2128] | 131/10/19 |
| 16 | +0.2732 | [+0.2248, +0.3212] | 132/10/18 |
| 32 | +0.3137 | [+0.2596, +0.3704] | 133/14/13 |
| 64 | +0.3786 | [+0.3229, +0.4343] | 137/14/9 |

## Interpretation

The online methods encounter different state trajectories. Paired statistics therefore compare prompt-level averages, not decoding round N across methods. Offline values are off-policy frozen-state predictions and are included only to check whether the relative ordering survives.

**Recommendation: MODERATE GO.**
