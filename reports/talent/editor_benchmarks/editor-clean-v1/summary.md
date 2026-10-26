# Clean semantic editors

Existing records only; missing fits/evaluations are not launched.

| Method | Dimension | Tier | Seed | Selected epoch | Validation objective | Test evaluation |
|---|---:|---:|---:|---:|---:|---|
| semantic_editor | 768 | 1 | 42 | 115 | 0.0485764 | available |
| semantic_editor | 768 | 1 | 43 | 112 | 0.0484552 | available |
| semantic_editor | 768 | 1 | 44 | 198 | 0.0475649 | available |
| semantic_editor | 768 | 1 | 45 | 186 | 0.0479681 | available |
| semantic_editor | 768 | 1 | 46 | 203 | 0.0482582 | available |
| ot_editor | 768 | 1 | 42 | 242 | 0.124298 | available |
| ot_editor | 768 | 1 | 43 | 292 | 0.125392 | available |
| ot_editor | 768 | 1 | 44 | 291 | 0.123049 | available |
| ot_editor | 768 | 1 | 45 | 280 | 0.120995 | available |
| ot_editor | 768 | 1 | 46 | 263 | 0.121068 | available |
| semantic_editor | 256 | 1 | 42 | 258 | 0.0465422 | available |
| semantic_editor | 256 | 1 | 43 | 167 | 0.0468514 | available |
| semantic_editor | 256 | 1 | 44 | 290 | 0.0460611 | available |
| semantic_editor | 256 | 1 | 45 | 208 | 0.048036 | available |
| semantic_editor | 256 | 1 | 46 | 278 | 0.046335 | available |
| ot_editor | 256 | 1 | 42 | 416 | 0.121431 | available |
| ot_editor | 256 | 1 | 43 | 292 | 0.130134 | available |
| ot_editor | 256 | 1 | 44 | 386 | 0.121486 | available |
| ot_editor | 256 | 1 | 45 | 345 | 0.127897 | available |
| ot_editor | 256 | 1 | 46 | 372 | 0.129748 | available |

## Explicit test evaluations

g-consistency, realized-state accuracy and oracle recovery are different criteria. Means ± sample standard deviations below are across training seeds, not test-unit errors. Previously viewed test results remain exploratory.

### ot_editor, 256D, tier 1

5 training seed(s): multi-seed.

all:

- accuracy_D: 0.8368 ± 0.00216794
- accuracy_T: 0.826 ± 0.00374166
- accuracy_U: 0.81 ± 0.0023452
- accuracy_X: 0.9976 ± 0.000894446
- energy_score: 0.620506 ± 0.00149678
- mean_cosine: 0.925141 ± 0.000669947
- mean_l2: 0.355483 ± 0.000678839
- mean_mse: 0.00058768 ± 6.99969e-06
- mean_shift_l2: 0.195087 ± 0.00118506
- realized_joint_accuracy: 0.574 ± 0.00273861
- realized_joint_nll: 1.22718 ± 0.00439065
- sample_spread: 0.00323986 ± 0.000152511
- semantic_kl: 0.0840278 ± 0.00195971
- standardized_mean_mse: 0.523169 ± 0.00645688

identity:

- accuracy_D: 0.996094 ± 0
- accuracy_T: 0.799219 ± 0.00810017
- accuracy_U: 0.944531 ± 0.00174693
- accuracy_X: 1 ± 0
- energy_score: 0.179237 ± 0.013775
- mean_cosine: 0.994275 ± 0.000823414
- mean_l2: 0.105517 ± 0.00795879
- mean_mse: 4.4817e-05 ± 6.41129e-06
- mean_shift_l2: 0.107031 ± 0.00793825
- realized_joint_accuracy: 0.765625 ± 0.00730792
- realized_joint_nll: 0.582887 ± 0.00314298
- sample_spread: 0.000897954 ± 5.8508e-05
- semantic_kl: 0.0538203 ± 0.00298865
- standardized_mean_mse: 0.0359058 ± 0.00496484

nonidentity:

- accuracy_D: 0.781989 ± 0.00291394
- accuracy_T: 0.835215 ± 0.0051708
- accuracy_U: 0.76371 ± 0.00275455
- accuracy_X: 0.996774 ± 0.00120219
- energy_score: 0.77234 ± 0.00425619
- mean_cosine: 0.901353 ± 0.00116741
- mean_l2: 0.441493 ± 0.00320956
- mean_mse: 0.000774472 ± 1.14986e-05
- mean_shift_l2: 0.225385 ± 0.00126331
- realized_joint_accuracy: 0.508065 ± 0.00251455
- realized_joint_nll: 1.44887 ± 0.00633859
- sample_spread: 0.00404568 ± 0.00019596
- semantic_kl: 0.0944218 ± 0.00346594
- standardized_mean_mse: 0.690829 ± 0.0102681

### ot_editor, 768D, tier 1

5 training seed(s): multi-seed.

all:

- accuracy_D: 0.8344 ± 0.00230217
- accuracy_T: 0.8334 ± 0.00270186
- accuracy_U: 0.8372 ± 0.00164317
- accuracy_X: 1 ± 0
- energy_score: 0.633547 ± 0.00482814
- mean_cosine: 0.904328 ± 0.00144068
- mean_l2: 0.40114 ± 0.00299674
- mean_mse: 0.000249861 ± 4.56081e-06
- mean_shift_l2: 0.227313 ± 0.00270664
- realized_joint_accuracy: 0.6042 ± 0.00311449
- realized_joint_nll: 1.10369 ± 0.00457561
- sample_spread: 0.00197714 ± 1.92936e-05
- semantic_kl: 0.0772379 ± 0.00284236
- standardized_mean_mse: 0.539989 ± 0.00977099

identity:

- accuracy_D: 0.999219 ± 0.00174693
- accuracy_T: 0.830469 ± 0.0105541
- accuracy_U: 0.994531 ± 0.00213954
- accuracy_X: 1 ± 0
- energy_score: 0.181019 ± 0.00482573
- mean_cosine: 0.993286 ± 0.000365579
- mean_l2: 0.114652 ± 0.00321864
- mean_mse: 1.75093e-05 ± 9.43319e-07
- mean_shift_l2: 0.115464 ± 0.00322227
- realized_joint_accuracy: 0.825781 ± 0.013131
- realized_joint_nll: 0.428129 ± 0.0157893
- sample_spread: 0.000342468 ± 8.74399e-06
- semantic_kl: 0.0491656 ± 0.00123768
- standardized_mean_mse: 0.0347671 ± 0.00177516

nonidentity:

- accuracy_D: 0.777688 ± 0.00278718
- accuracy_T: 0.834409 ± 0.00147237
- accuracy_U: 0.783065 ± 0.00203844
- accuracy_X: 1 ± 0
- energy_score: 0.789256 ± 0.00766284
- mean_cosine: 0.873718 ± 0.00202606
- mean_l2: 0.499716 ± 0.00478901
- mean_mse: 0.00032981 ± 6.36712e-06
- mean_shift_l2: 0.265798 ± 0.00428866
- realized_joint_accuracy: 0.527957 ± 0.0034791
- realized_joint_nll: 1.33615 ± 0.00538849
- sample_spread: 0.00253961 ± 2.71134e-05
- semantic_kl: 0.0868972 ± 0.0034231
- standardized_mean_mse: 0.713829 ± 0.0135927

### semantic_editor, 256D, tier 1

5 training seed(s): multi-seed.

all:

- accuracy_D: 0.8378 ± 0.000447208
- accuracy_T: 0.8286 ± 0.00151656
- accuracy_U: 0.815 ± 0.00158112
- accuracy_X: 1 ± 0
- energy_score: 0.496569 ± 0.00243417
- mean_cosine: 0.947308 ± 0.000359098
- mean_l2: 0.279177 ± 0.00115835
- mean_mse: 0.000408066 ± 3.12022e-06
- mean_shift_l2: 0.127586 ± 0.00124674
- realized_joint_accuracy: 0.5796 ± 0.00114016
- realized_joint_nll: 1.17115 ± 0.00269009
- sample_spread: 0.00205411 ± 1.726e-05
- semantic_kl: 0.0313249 ± 0.000371792
- standardized_mean_mse: 0.366592 ± 0.00283807

identity:

- accuracy_D: 0.996094 ± 0
- accuracy_T: 0.817969 ± 0.00213954
- accuracy_U: 0.940625 ± 0.00174693
- accuracy_X: 1 ± 0
- energy_score: 0.0171559 ± 0.000701488
- mean_cosine: 0.99987 ± 1.3458e-06
- mean_l2: 0.0101234 ± 0.000402216
- mean_mse: 1.01737e-06 ± 1.00437e-08
- mean_shift_l2: 0.010709 ± 0.000404337
- realized_joint_accuracy: 0.778906 ± 0.00213954
- realized_joint_nll: 0.573222 ± 0.00161105
- sample_spread: 0.000237228 ± 6.23255e-06
- semantic_kl: 0.00875496 ± 0.000189534
- standardized_mean_mse: 0.00083792 ± 9.3354e-06

nonidentity:

- accuracy_D: 0.783333 ± 0.000601093
- accuracy_T: 0.832258 ± 0.0019936
- accuracy_U: 0.771774 ± 0.00175247
- accuracy_X: 1 ± 0
- energy_score: 0.661528 ± 0.00304407
- mean_cosine: 0.929222 ± 0.000482745
- mean_l2: 0.371754 ± 0.00142452
- mean_mse: 0.000548125 ± 4.1947e-06
- mean_shift_l2: 0.167802 ± 0.00180895
- realized_joint_accuracy: 0.511021 ± 0.0019936
- realized_joint_nll: 1.37689 ± 0.0033206
- sample_spread: 0.00267927 ± 2.46736e-05
- semantic_kl: 0.0390909 ± 0.000515204
- standardized_mean_mse: 0.492442 ± 0.00381575

### semantic_editor, 768D, tier 1

5 training seed(s): multi-seed.

all:

- accuracy_D: 0.8438 ± 0.00109545
- accuracy_T: 0.8408 ± 0.00192354
- accuracy_U: 0.8352 ± 0.00130385
- accuracy_X: 1 ± 0
- energy_score: 0.511622 ± 0.00530609
- mean_cosine: 0.931276 ± 0.00130798
- mean_l2: 0.318821 ± 0.00336966
- mean_mse: 0.000177418 ± 3.78161e-06
- mean_shift_l2: 0.142857 ± 0.00205439
- realized_joint_accuracy: 0.6156 ± 0.00194935
- realized_joint_nll: 1.05696 ± 0.00136429
- sample_spread: 0.00127981 ± 2.41056e-05
- semantic_kl: 0.0286395 ± 0.000609069
- standardized_mean_mse: 0.387143 ± 0.00829529

identity:

- accuracy_D: 0.998437 ± 0.00213954
- accuracy_T: 0.842969 ± 0.00174693
- accuracy_U: 0.988281 ± 0
- accuracy_X: 1 ± 0
- energy_score: 0.0164006 ± 0.000391861
- mean_cosine: 0.999873 ± 3.98251e-06
- mean_l2: 0.0103844 ± 0.000198005
- mean_mse: 3.31342e-07 ± 1.04597e-08
- mean_shift_l2: 0.0107298 ± 0.000172273
- realized_joint_accuracy: 0.834375 ± 0.00213954
- realized_joint_nll: 0.40884 ± 0.000765868
- sample_spread: 0.000105899 ± 7.38054e-06
- semantic_kl: 0.0101167 ± 0.000319435
- standardized_mean_mse: 0.000668405 ± 2.30786e-05

nonidentity:

- accuracy_D: 0.790591 ± 0.00112454
- accuracy_T: 0.840054 ± 0.00251455
- accuracy_U: 0.782527 ± 0.0017525
- accuracy_X: 1 ± 0
- energy_score: 0.68202 ± 0.00716831
- mean_cosine: 0.907672 ± 0.00175845
- mean_l2: 0.42495 ± 0.00454115
- mean_mse: 0.000238351 ± 5.08369e-06
- mean_shift_l2: 0.18832 ± 0.00277039
- realized_joint_accuracy: 0.540323 ± 0.00190082
- realized_joint_nll: 1.27997 ± 0.00194168
- sample_spread: 0.00168373 ± 3.12499e-05
- semantic_kl: 0.0350129 ± 0.000854688
- standardized_mean_mse: 0.520123 ± 0.0111529

