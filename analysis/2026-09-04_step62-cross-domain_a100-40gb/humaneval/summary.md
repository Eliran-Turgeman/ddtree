# Step 5 online DFlash2 tree evaluation

- Source commit: `660645efbe0cfe7913a218c577290c360cbed5d5`
- GPU: NVIDIA A100-SXM4-40GB
- PyTorch/CUDA/Transformers: 2.7.0 / 12.8 / 5.16.1
- Prompts: 164

## Online acceptance

| Method | B | Matched draft tokens | 95% CI | Full block | Tokens/s | Exact sequential |
| :----- | -: | -------------------: | -----: | ---------: | -------: | ---------------: |
| DFlash2-GreedyPath | 7 | 3.2289 | [3.1553, 3.3006] | 20.31% | 78.78 | 42/164 |
| Unary-FullMass | 8 | 3.5027 | [3.4279, 3.5753] | 19.96% | 80.31 | 33/164 |
| Unary-FullMass | 16 | 3.8914 | [3.8183, 3.9620] | 25.41% | 87.32 | 45/164 |
| Unary-FullMass | 32 | 4.1959 | [4.1182, 4.2704] | 29.83% | 92.12 | 42/164 |
| Unary-FullMass | 64 | 4.4456 | [4.3692, 4.5206] | 32.65% | 94.01 | 48/164 |
| Pairwise-MassPreserving | 8 | 3.7054 | [3.6332, 3.7741] | 21.18% | 83.63 | 44/164 |
| Pairwise-MassPreserving | 16 | 4.2195 | [4.1414, 4.2939] | 28.98% | 92.78 | 42/164 |
| Pairwise-MassPreserving | 32 | 4.6150 | [4.5356, 4.6927] | 35.93% | 99.07 | 48/164 |
| Pairwise-MassPreserving | 64 | 4.8918 | [4.8170, 4.9651] | 39.57% | 101.27 | 49/164 |

## Pairwise minus Unary

| B | Matched-token gain | 95% CI | Prompts improve/tie/hurt |
| -: | -----------------: | -----: | :----------------------- |
| 8 | +0.2027 | [+0.1688, +0.2363] | 133/4/27 |
| 16 | +0.3280 | [+0.2922, +0.3652] | 151/3/10 |
| 32 | +0.4190 | [+0.3818, +0.4580] | 160/0/4 |
| 64 | +0.4462 | [+0.4092, +0.4809] | 161/1/2 |

## Interpretation

The online methods encounter different state trajectories. Paired statistics therefore compare prompt-level averages, not decoding round N across methods. Offline values are off-policy frozen-state predictions and are included only to check whether the relative ordering survives.

**Recommendation: STRONG GO.**
