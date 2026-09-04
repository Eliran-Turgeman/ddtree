# Step 5 online DFlash2 tree evaluation

- Source commit: `660645efbe0cfe7913a218c577290c360cbed5d5`
- GPU: NVIDIA A100-SXM4-40GB
- PyTorch/CUDA/Transformers: 2.7.0 / 12.8 / 5.16.1
- Prompts: 32

## Online acceptance

| Method | B | Matched draft tokens | 95% CI | Full block | Tokens/s | Exact sequential |
| :----- | -: | -------------------: | -----: | ---------: | -------: | ---------------: |
| DFlash2-GreedyPath | 7 | 3.9117 | [3.7263, 4.1040] | 29.69% | 90.78 | 20/32 |
| Unary-FullMass | 7 | 3.9623 | [3.7829, 4.1417] | 23.43% | 88.42 | 21/32 |
| Unary-FullMass | 8 | 4.0590 | [3.8814, 4.2393] | 25.49% | 88.75 | 20/32 |
| Unary-FullMass | 16 | 4.4820 | [4.2930, 4.6738] | 32.91% | 96.94 | 18/32 |
| Unary-FullMass | 32 | 4.7768 | [4.6010, 4.9527] | 37.31% | 101.45 | 17/32 |
| Unary-FullMass | 64 | 4.9506 | [4.7769, 5.1216] | 41.63% | 102.10 | 17/32 |
| Pairwise-MassPreserving | 7 | 4.1711 | [4.0062, 4.3405] | 27.22% | 92.15 | 20/32 |
| Pairwise-MassPreserving | 8 | 4.3445 | [4.1845, 4.5053] | 29.33% | 94.22 | 22/32 |
| Pairwise-MassPreserving | 16 | 4.8562 | [4.6696, 5.0355] | 41.10% | 102.69 | 19/32 |
| Pairwise-MassPreserving | 32 | 5.1694 | [4.9917, 5.3470] | 46.71% | 108.26 | 19/32 |
| Pairwise-MassPreserving | 64 | 5.4642 | [5.3029, 5.6191] | 52.95% | 110.61 | 18/32 |

## Pairwise minus Unary

| B | Matched-token gain | 95% CI | Prompts improve/tie/hurt |
| -: | -----------------: | -----: | :----------------------- |
| 7 | +0.2088 | [+0.1250, +0.2832] | 26/1/5 |
| 8 | +0.2855 | [+0.2205, +0.3477] | 28/3/1 |
| 16 | +0.3742 | [+0.2724, +0.4758] | 30/1/1 |
| 32 | +0.3926 | [+0.3086, +0.4759] | 29/2/1 |
| 64 | +0.5136 | [+0.4255, +0.5994] | 31/1/0 |

## Interpretation

The online methods encounter different state trajectories. Paired statistics therefore compare prompt-level averages, not decoding round N across methods. Offline values are off-policy frozen-state predictions and are included only to check whether the relative ordering survives.

**Recommendation: STRONG GO.**
