# Step 5 online DFlash2 tree evaluation

- Source commit: `a9165f1c379fa07aaa6136a28bfa113754e77578`
- GPU: NVIDIA A100-SXM4-40GB
- PyTorch/CUDA/Transformers: 2.7.0 / 12.8 / 5.16.1
- Prompts: 160

## Online acceptance

| Method | B | Matched draft tokens | 95% CI | Full block | Tokens/s | Exact sequential |
| :----- | -: | -------------------: | -----: | ---------: | -------: | ---------------: |
| DFlash2-GreedyPath | 7 | 2.5918 | [2.4068, 2.7862] | 9.35% | 67.07 | 21/160 |
| Unary-FullMass | 8 | 2.8664 | [2.6955, 3.0398] | 8.08% | 68.96 | 23/160 |
| Unary-FullMass | 16 | 3.1874 | [3.0091, 3.3701] | 12.00% | 74.71 | 21/160 |
| Unary-FullMass | 32 | 3.4237 | [3.2474, 3.6043] | 12.51% | 78.42 | 20/160 |
| Unary-FullMass | 64 | 3.6445 | [3.4663, 3.8244] | 14.93% | 80.33 | 21/160 |
| Pairwise-MassPreserving | 8 | 3.0086 | [2.8297, 3.1896] | 8.87% | 71.09 | 23/160 |
| Pairwise-MassPreserving | 16 | 3.4071 | [3.2167, 3.6003] | 12.78% | 78.34 | 22/160 |
| Pairwise-MassPreserving | 32 | 3.7659 | [3.5705, 3.9637] | 18.84% | 84.02 | 23/160 |
| Pairwise-MassPreserving | 64 | 4.0210 | [3.8274, 4.2160] | 20.52% | 86.29 | 23/160 |

## Pairwise minus Unary

| B | Matched-token gain | 95% CI | Prompts improve/tie/hurt |
| -: | -----------------: | -----: | :----------------------- |
| 8 | +0.1422 | [+0.1012, +0.1829] | 121/9/30 |
| 16 | +0.2197 | [+0.1676, +0.2714] | 121/10/29 |
| 32 | +0.3422 | [+0.2832, +0.4026] | 132/12/16 |
| 64 | +0.3766 | [+0.3178, +0.4341] | 138/11/11 |

## Interpretation

The online methods encounter different state trajectories. Paired statistics therefore compare prompt-level averages, not decoding round N across methods. Offline values are off-policy frozen-state predictions and are included only to check whether the relative ordering survives.

**Recommendation: MODERATE GO.**
