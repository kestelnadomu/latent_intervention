# latent_lab: latest run

- run: `run-20261030-183501.json`, created 2026-10-30T18:35:01
- code: commit `18f09e3`
- encoder: `Encoder('embeddinggemma', 768)`; seeds [0, 1, 2, 3, 4]; main state version S1 = ['X', 'T', 'D', 'U']
- units: {'pool': 2800, 'psi': 2000, 'judge': 2000, 'eval': 1000}
- selected renderer configurations: {'editor': {'epochs': 500, 'proximity': 3.0}, 'flow': {'epochs': 500, 'lr': 0.0003, 'hidden': 68, 'fidelity': 0.1}, 'ot_editor': {'epochs': 500, 'lr': 0.01, 'proximity': 3.0}, 'ot_flow': {'epochs': 500, 'lr': 0.0003, 'hidden': 512}}
- training time: 217.2 min

## Reader ψ and judge ψ_eval on the eval units

| encoder, reader | acc_X | acc_T | acc_D | acc_U | acc_M | joint_acc | joint_nll |
|---|---|---|---|---|---|---|---|
| embeddinggemma-128, psi (mean) | 0.995 | 0.769 | 0.992 | 0.842 | 0.603 | 0.650 | 1.000 |
| embeddinggemma-128, judge | 0.992 | 0.747 | 0.982 | 0.835 | 0.588 | 0.615 | 1.060 |
| embeddinggemma-256, psi (mean) | 0.995 | 0.818 | 0.999 | 0.953 | 0.688 | 0.778 | 0.613 |
| embeddinggemma-256, judge | 0.994 | 0.791 | 0.997 | 0.943 | 0.643 | 0.743 | 0.715 |
| embeddinggemma-512, psi (mean) | 0.995 | 0.834 | 0.999 | 0.990 | 0.766 | 0.822 | 0.478 |
| embeddinggemma-512, judge | 0.994 | 0.800 | 0.999 | 0.981 | 0.730 | 0.780 | 0.585 |
| embeddinggemma-768, psi (mean) | 0.996 | 0.837 | 0.999 | 0.990 | 0.793 | 0.827 | 0.454 |
| embeddinggemma-768, judge | 0.994 | 0.810 | 0.998 | 0.987 | 0.740 | 0.792 | 0.556 |

## Renderers, deployment (trained and tested on ψ(z), target ω_S(s)), mean over seeds

| rho | joint_acc | acc_X | acc_D | acc_U | acc_T | acc_M | gap_closed | distance | down_acc | down_agree | parity_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| editor | 0.388 | 0.988 | 0.661 | 0.690 | 0.823 | 0.712 | 0.108 | 0.720 | 0.612 | 0.678 | 0.498 |
| flow | 0.374 | 0.986 | 0.662 | 0.688 | 0.796 | 0.733 | 0.177 | 0.664 | 0.617 | 0.628 | 0.680 |
| group_ot | 0.268 | 0.997 | 0.550 | 0.572 | 0.759 | 0.716 | 0.160 | 0.679 | 0.614 | 0.626 | 0.662 |
| ot_editor | 0.387 | 0.985 | 0.662 | 0.690 | 0.825 | 0.648 | 0.130 | 0.702 | 0.612 | 0.662 | 0.436 |
| ot_flow | 0.369 | 0.995 | 0.662 | 0.684 | 0.781 | 0.731 | 0.112 | 0.714 | 0.617 | 0.596 | 0.734 |
| state_effect | 0.385 | 0.998 | 0.662 | 0.690 | 0.802 | 0.749 | 0.162 | 0.676 | 0.616 | 0.589 | 0.781 |

## References (true source states)

| rho | joint_acc | acc_X | acc_D | acc_U | acc_T | acc_M | gap_closed | distance | down_acc | down_agree | parity_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| constant | 0.130 | 0.485 | 0.561 | 0.505 | 0.811 | 0.743 | 0.097 | 0.727 | 0.555 | 0.763 | 0.513 |
| identity | 0.001 | 0.003 | 0.565 | 0.511 | 0.810 | 0.740 | 0.000 | 0.803 | 0.494 | 1.000 | 0.300 |
| real φ(x') | 0.819 | 0.989 | 1.000 | 0.993 | 0.834 | 0.743 | 1.000 | 0.000 | 0.648 | 0.569 | 0.758 |

## Estimation gap (target ω_S(s)): train × test states

| rho | down_acc / true s / true s | down_acc / true s / ψ(z) | down_acc / ψ(z) / true s | down_acc / ψ(z) / ψ(z) | joint_acc / true s / true s | joint_acc / true s / ψ(z) | joint_acc / ψ(z) / true s | joint_acc / ψ(z) / ψ(z) |
|---|---|---|---|---|---|---|---|---|
| editor | 0.644 | 0.616 | 0.639 | 0.612 | 0.466 | 0.394 | 0.454 | 0.388 |
| flow | 0.620 | 0.613 | 0.621 | 0.617 | 0.380 | 0.372 | 0.382 | 0.374 |
| group_ot | 0.613 | 0.611 | 0.616 | 0.614 | 0.268 | 0.267 | 0.270 | 0.268 |
| ot_editor | 0.635 | 0.614 | 0.632 | 0.612 | 0.460 | 0.392 | 0.452 | 0.387 |
| ot_flow | 0.625 | 0.621 | 0.620 | 0.617 | 0.384 | 0.375 | 0.377 | 0.369 |
| state_effect | 0.621 | 0.614 | 0.623 | 0.616 | 0.391 | 0.382 | 0.393 | 0.385 |

## Renderers without the operator (given the true s', true source states, trained on ψ(z))

| rho | joint_acc | acc_X | acc_D | acc_U | acc_T | acc_M | gap_closed | distance | down_acc | down_agree | parity_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| editor | 0.960 | 0.991 | 0.999 | 1.000 | 0.970 | 0.712 | 0.128 | 0.706 | 0.652 | 0.671 | 0.538 |
| flow | 0.762 | 0.987 | 0.994 | 0.955 | 0.812 | 0.736 | 0.199 | 0.647 | 0.631 | 0.625 | 0.679 |
| group_ot | 0.270 | 0.998 | 0.550 | 0.573 | 0.759 | 0.716 | 0.160 | 0.678 | 0.616 | 0.626 | 0.659 |
| ot_editor | 0.951 | 0.989 | 0.999 | 1.000 | 0.963 | 0.652 | 0.154 | 0.685 | 0.646 | 0.644 | 0.492 |
| ot_flow | 0.763 | 0.997 | 0.997 | 0.962 | 0.796 | 0.731 | 0.134 | 0.698 | 0.639 | 0.596 | 0.714 |
| state_effect | 0.804 | 0.999 | 0.997 | 0.982 | 0.821 | 0.747 | 0.188 | 0.656 | 0.649 | 0.582 | 0.774 |

## Downstream references

| reference | value |
|---|---|
| pi(z) vs factual Y | 0.661 |
| pi(phi(x')) vs Y' | 0.648 |
| ECE of pi(z) vs factual Y | 0.048 |
| ECE of pi(phi(x')) vs Y' | 0.058 |
| Bayes vs Y' \| all parents (T, M known) | 0.783 |
| Bayes vs Y' \| state only (T known, M unknown) | 0.680 |
| Bayes vs Y' \| text content (X, D, U, M, proxies) | 0.773 |
| Bayes vs Y' \| text content without M | 0.663 |
| Bayes vs Y' \| no talent or motivation information (X, D, U) | 0.652 |
