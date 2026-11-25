# Operator benchmark

Assumed vs. true symbolic operator on states (exact transition matrices; the log score uses 50,000 simulated units). TV: expected total-variation distance of the counterfactual distributions over factual states and non-identity do(X=k). Missing: true mass outside the assumed support. Shift: error of E[V'] along the intervention (positive = the assumed operator moves V further in the direction of the change in X than the truth). Y': Bayes-optimal outcome prediction (TV; E[Y'] error along the intervention). NLL: log score of realized counterfactuals, assumed vs. true kernel.

## Conditions

| condition | space | target | TV | missing | shift D | shift U | shift M | Y' TV | Y' shift | NLL assumed | NLL true |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1/true | S1 | total | 0.0000 | 0.0000 | 0.000 | 0.000 | nan | 0.0000 | 0.000 | 0.929 | 0.929 |
| S1/true | S1 | ps_D | 0.4400 | 0.0000 | 0.440 | -0.000 | nan | 0.0671 | 0.082 | 1.275 | 0.461 |
| S1/true | S1 | ps_DU | 0.6810 | 0.0723 | 0.440 | 0.519 | nan | 0.1452 | 0.180 | 3.472 | 0.000 |
| S6/true | S6 | total | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.000 | 0.929 | 0.929 |
| S2/unaware | S2 | total | 0.2204 | 0.0000 | 0.012 | -0.002 | nan | 0.0832 | 0.005 | 1.201 | 0.977 |
| S2/refit | S2 | total | 0.3571 | 0.0143 | 0.032 | 0.057 | nan | 0.1088 | 0.033 | 2.968 | 0.977 |
| S3/unaware | S3 | total | 0.0000 | 0.0000 | nan | nan | nan | nan | nan | 0.000 | -0.000 |
| S3/unaware | S3 | ps_DU | 0.0000 | 0.0000 | nan | nan | nan | nan | nan | 0.000 | 0.000 |
| S4/unaware | S4 | total | 0.0000 | 0.0000 | nan | 0.000 | nan | nan | nan | 0.461 | 0.461 |
| S4/unaware | S4 | ps_D | 0.0000 | 0.0000 | nan | 0.000 | nan | nan | nan | 0.461 | 0.461 |
| S5/unaware | S5 | total | 0.1395 | 0.0000 | nan | -0.002 | nan | nan | nan | 0.572 | 0.489 |
| S5/unaware | S5 | ps_D | 0.1395 | 0.0000 | nan | -0.002 | nan | nan | nan | 0.572 | 0.489 |
| S5/refit | S5 | total | 0.1912 | 0.0000 | nan | 0.057 | nan | nan | nan | 0.921 | 0.489 |
| S5/refit | S5 | ps_D | 0.1912 | 0.0000 | nan | 0.057 | nan | nan | nan | 0.921 | 0.489 |
| O1/unaware | S1 | total | 0.2219 | 0.0000 | 0.012 | -0.002 | nan | 0.0474 | 0.004 | 1.201 | 0.929 |
| O1/refit | S1 | total | 0.3668 | 0.0143 | 0.032 | 0.057 | nan | 0.0656 | 0.020 | 2.968 | 0.929 |
| O2a/naive | S1 | total | 0.2629 | 0.0052 | 0.022 | 0.007 | nan | 0.0384 | 0.004 | 1.543 | 0.929 |
| O2a/intercept | S1 | total | 0.2679 | 0.0052 | 0.019 | 0.009 | nan | 0.0393 | 0.004 | 1.562 | 0.929 |
| O2d/coef_X/x0.5 | S1 | total | 0.2776 | 0.0052 | -0.180 | -0.211 | nan | 0.0601 | -0.073 | 1.378 | 0.929 |
| O2d/coef_X/x0.75 | S1 | total | 0.1403 | 0.0052 | -0.087 | -0.104 | nan | 0.0298 | -0.036 | 1.149 | 0.929 |
| O2d/coef_X/x1.25 | S1 | total | 0.1375 | 0.0099 | 0.083 | 0.097 | nan | 0.0291 | 0.034 | 1.261 | 0.929 |
| O2d/coef_X/x1.5 | S1 | total | 0.2624 | 0.0699 | 0.158 | 0.184 | nan | 0.0557 | 0.065 | 2.934 | 0.929 |
| O2d/coef_T/x0.5 | S1 | total | 0.1190 | 0.0000 | 0.010 | 0.003 | nan | 0.0255 | 0.004 | 1.002 | 0.929 |
| O2d/coef_T/x0.75 | S1 | total | 0.0611 | 0.0000 | 0.007 | 0.003 | nan | 0.0131 | 0.002 | 0.948 | 0.929 |
| O2d/coef_T/x1.25 | S1 | total | 0.0616 | 0.0000 | -0.009 | -0.005 | nan | 0.0133 | -0.003 | 0.947 | 0.929 |
| O2d/coef_T/x1.5 | S1 | total | 0.1202 | 0.0000 | -0.018 | -0.011 | nan | 0.0260 | -0.006 | 1.006 | 0.929 |
| O2d/noise_mu/x0.5 | S1 | total | 0.0283 | 0.0000 | 0.000 | 0.002 | nan | 0.0061 | 0.000 | 0.931 | 0.929 |
| O2d/noise_mu/x0.75 | S1 | total | 0.0142 | 0.0000 | 0.000 | 0.001 | nan | 0.0030 | 0.000 | 0.929 | 0.929 |
| O2d/noise_mu/x1.25 | S1 | total | 0.0142 | 0.0000 | -0.000 | -0.001 | nan | 0.0030 | -0.000 | 0.930 | 0.929 |
| O2d/noise_mu/x1.5 | S1 | total | 0.0283 | 0.0000 | -0.001 | -0.002 | nan | 0.0061 | -0.001 | 0.932 | 0.929 |
| O2d/noise_sigma/x0.5 | S1 | total | 0.2162 | 0.0000 | 0.041 | 0.043 | nan | 0.0310 | 0.015 | 1.253 | 0.929 |
| O2d/noise_sigma/x0.75 | S1 | total | 0.0920 | 0.0000 | 0.022 | 0.023 | nan | 0.0134 | 0.008 | 0.972 | 0.929 |
| O2d/noise_sigma/x1.25 | S1 | total | 0.0667 | 0.0000 | -0.021 | -0.024 | nan | 0.0104 | -0.008 | 0.948 | 0.929 |
| O2d/noise_sigma/x1.5 | S1 | total | 0.1171 | 0.0000 | -0.039 | -0.045 | nan | 0.0187 | -0.016 | 0.984 | 0.929 |
| O3/drop_X_D | S1 | total | 0.4400 | 0.4400 | -0.440 | 0.000 | nan | 0.0671 | -0.082 | 12.358 | 0.929 |
| O3/drop_X_D | S1 | ps_D | 0.0000 | 0.0000 | 0.000 | 0.000 | nan | 0.0000 | 0.000 | 0.461 | 0.461 |
| O3/drop_X_U | S1 | total | 0.5138 | 0.5138 | -0.000 | -0.519 | nan | 0.0790 | -0.096 | 14.455 | 0.929 |
| O4/block_X_D | S1 | ps_D | 0.0000 | 0.0000 | 0.000 | 0.000 | nan | 0.0000 | 0.000 | 0.461 | 0.461 |
| O4/block_X_D | S1 | total | 0.4400 | 0.4400 | -0.440 | 0.000 | nan | 0.0671 | -0.082 | 12.358 | 0.929 |
| O5/c0.0 | S6 | total | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0.000 | 0.929 | 0.929 |
| O5/c0.25 | S6 | total | 0.1012 | 0.0000 | 0.000 | 0.000 | 0.101 | 0.0329 | 0.036 | 1.040 | 0.929 |
| O5/c0.5 | S6 | total | 0.1933 | 0.0000 | -0.000 | 0.000 | 0.193 | 0.0626 | 0.069 | 1.163 | 0.929 |
| O5/c1.0 | S6 | total | 0.3655 | 0.0000 | -0.000 | 0.000 | 0.366 | 0.1172 | 0.130 | 1.525 | 0.929 |
| O5/c1.5 | S6 | total | 0.4983 | 0.0833 | -0.000 | 0.000 | 0.511 | 0.1633 | 0.181 | 4.010 | 0.929 |

## O2a: estimator bias (large sample)

| estimator | TV | rel. err coef X | coef T | noise mu | noise sigma |
|---|---|---|---|---|---|
| naive | 0.2629 | 0.105 | 0.114 | 1.292 | 0.553 |
| intercept | 0.2679 | 0.113 | 0.119 | 1.378 | 0.562 |

## O2b: sampling error (mean ± SD over resamples)

| estimator | n_fit | TV | rel. err coef X |
|---|---|---|---|
| intercept | 100 | 0.3033 ± 0.0558 | 0.178 ± 0.089 |
| intercept | 300 | 0.2736 ± 0.0248 | 0.123 ± 0.056 |
| intercept | 1000 | 0.2676 ± 0.0150 | 0.104 ± 0.043 |
| intercept | 3000 | 0.2715 ± 0.0112 | 0.116 ± 0.013 |
| intercept | 10000 | 0.2661 ± 0.0059 | 0.117 ± 0.010 |
| naive | 100 | 0.3041 ± 0.0585 | 0.163 ± 0.078 |
| naive | 300 | 0.2599 ± 0.0230 | 0.108 ± 0.039 |
| naive | 1000 | 0.2664 ± 0.0171 | 0.100 ± 0.035 |
| naive | 3000 | 0.2644 ± 0.0096 | 0.106 ± 0.013 |
| naive | 10000 | 0.2619 ± 0.0052 | 0.109 ± 0.009 |

## O2d: sensitivity (TV; one parameter group scaled)

| group | ×0.5 | ×0.75 | ×1.25 | ×1.5 |
|---|---|---|---|---|
| coef_T | 0.1190 | 0.0611 | 0.0616 | 0.1202 |
| coef_X | 0.2776 | 0.1403 | 0.1375 | 0.2624 |
| noise_mu | 0.0283 | 0.0142 | 0.0142 | 0.0283 |
| noise_sigma | 0.2162 | 0.0920 | 0.0667 | 0.1171 |

## O5 refit: estimated false coefficient c (true 0)

| estimator | n_fit | c_hat | TV |
|---|---|---|---|
| intercept | 100 | -0.001 ± 0.064 | 0.4752 ± 0.0443 |
| intercept | 300 | 0.007 ± 0.035 | 0.4557 ± 0.0193 |
| intercept | 1000 | 0.008 ± 0.015 | 0.4566 ± 0.0103 |
| intercept | 3000 | 0.002 ± 0.010 | 0.4500 ± 0.0067 |
| intercept | 10000 | -0.001 ± 0.006 | 0.4501 ± 0.0038 |
| naive | 100 | 0.432 ± 0.048 | 0.6132 ± 0.0447 |
| naive | 300 | 0.437 ± 0.024 | 0.6088 ± 0.0194 |
| naive | 1000 | 0.432 ± 0.013 | 0.6033 ± 0.0132 |
| naive | 3000 | 0.430 ± 0.006 | 0.5992 ± 0.0066 |
| naive | 10000 | 0.428 ± 0.004 | 0.5976 ± 0.0034 |

## Screening (TV > 0.001)

Enter the end-to-end runs: O1/refit, O1/unaware, O2a/intercept, O2a/naive, O2d/coef_T/x0.5, O2d/coef_T/x0.75, O2d/coef_T/x1.25, O2d/coef_T/x1.5, O2d/coef_X/x0.5, O2d/coef_X/x0.75, O2d/coef_X/x1.25, O2d/coef_X/x1.5, O2d/noise_mu/x0.5, O2d/noise_mu/x0.75, O2d/noise_mu/x1.25, O2d/noise_mu/x1.5, O2d/noise_sigma/x0.5, O2d/noise_sigma/x0.75, O2d/noise_sigma/x1.25, O2d/noise_sigma/x1.5, O3/drop_X_D, O3/drop_X_U, O4/block_X_D, O5/c0.25, O5/c0.5, O5/c1.0, O5/c1.5, S2/refit, S2/unaware, S5/refit, S5/unaware. The S family enters regardless, because omission errors are invisible at state level.
