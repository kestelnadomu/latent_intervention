# Five-method misspecification benchmark

Run `hz-misspec-v1`. 196 hZ fits prepared; independent evaluation reader selected separately.

True factual labels are available for hZ. This is not an end-to-end 1,000-label experiment.

Results use full predictive distributions; no rare/unseen transition mass is removed. Energy is available only for matching total-effect embedding pairs. Lower energy and KL are better.

| Scenario | Target | Method | Seeds | Conditions | Energy | Own-g KL | Independent X accuracy |
|---|---|---|---|---|---|---|---|

No official-test results have been produced.

See `results.json` for seed SD, separate 95% paired-unit intervals, and paired method differences. Per-condition records contain all head recoveries, preservation, calibration-reference diagnostics, support strata, identity/nonidentity groups, and numerical/inference cost. Intervals quantify unit sampling, not the uncertainty of training a new seed. Single-seed/pilot results are not five-seed findings. Disjoint P/L labels alone do not establish wording robustness.
