# Technical decisions

1. Model predicts binary candidate compatibility, not global arc ID. Arc IDs are local to each trip; using them as global class labels would be invalid.
2. Road coordinates projected with a local spherical equirectangular approximation centred on each trip network. Coordinates are in metres; at these city-scale extents local distances are appropriate. Features never include absolute coordinates or arc IDs.
3. Reference-route alignment uses chronological arc occurrences and cumulative route progress, not GPS row index. Derived labels are weak/reference-derived, not independently verified pointwise truth. Primary accuracy is explicitly named reference-derived agreement.
4. Observed 14 reference discontinuities and irregular timestamps require conservative alignment confidence and decoder resets. Raw data is retained unchanged.
5. Fit all learned transforms inside each trip-grouped CV training fold. Deterministic bounded per-trip sampling controls runtime and balances trip contributions; log sample counts.
6. Candidate generation sees only GPS and road graph. Reference route is loaded separately for training-label construction and post-prediction evaluation. Final evaluator mode accepts only evaluator partition and never substitutes student-test data.
7. Bayesian optimisation uses a real Gaussian-process surrogate and expected improvement; no unavailable package is impersonated. Full trial history retained.
8. Network and package availability, external deployment, evaluator trips and GitHub publication must be verified separately, with unresolved items explicitly marked.

9. Duplicate directed edge pairs occur in many trips. The shortest-path adjacency uses each endpoint pair once, preventing CSR summation from inflating lengths. Original candidate/reference arc IDs remain unchanged; indistinguishable duplicate IDs are documented as an exact-ID evaluation limitation.
10. Five initial experiments completed. HistogramGradientBoosting selected by mean grouped CV Macro F1 0.9654865 versus RandomForest 0.9641179. Ten genuine GP Bayesian trials improved CV to 0.9660143, but student Top-1 agreement declined slightly from 0.9546048 to 0.9539521. Retain and report this tradeoff.
11. Tuned classifier plus HMM reached 0.9705584 reference-derived agreement versus 0.9539521 classifier-only, on 13,790/19,160 confident student-test observations. Nine trips improved, one degraded. Do not claim universal improvement or independent ground truth.
12. Fixed decoder scale 35m, reset gap 30s, permissive motion ceiling 80m/s plus 150m tolerance. These are documented engineering choices, not falsely presented as Bayesian-tuned decoder parameters. Transition rows are normalised probabilities. Ordered route similarity is separately measured with difflib matching blocks and is not called exact edit distance.
