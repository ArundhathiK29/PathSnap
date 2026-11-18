# Final verification matrix

Verdict: **NOT READY** for final hackathon submission. Code, trained models, results and report are published in the public repository https://github.com/ArundhathiK29/PathSnap . Required genuine screenshots are still missing; live/physical deployment and reserved evaluation remain unverified.

| Requirement | Source | Evidence / test executed | Status | Corrective action |
|---|---|---|---|---|
| Dataset extraction and formats | Dataset Exploration pp1-2 | 360/360 extracted files CRC-matched to source ZIP; inventory.csv | PASS |  |
| EDA and topology | Dataset Exploration pp2-4 | inventory.csv, eda_summary.png, eda_example.png | PASS |  |
| Reference-derived labels | Dataset Exploration p3 | core.align_reference; DATA_DICTIONARY; confidence coverage | PASS | Resolve official final scorer when supplied; weak-label limitations retained |
| Features and candidates | Main Guidelines p2 | core.Geometry; 15 features; observed candidate misses | PASS |  |
| Five distinct models | Models pp1-4 | model_metrics.csv, five saved model files, cv_folds.json | PASS |  |
| Trip-grouped CV | Models p4 | Five model fold manifests, all 80 train trips, no overlap | PASS |  |
| Bayesian optimisation | Deliverables p5 | 10 logged trials; GP surrogate and expected improvement | PASS |  |
| HMM/Viterbi implementation | Deliverables pp5-6 | core.py; exhaustive path and normalised-transition tests | PASS |  |
| Classifier versus HMM comparison | Deliverables pp5-6 | decoder_metrics.csv; all ten student-test predictions | PASS |  |
| Sequence-level diagnostics | Models p5 | continuity, arc-set metrics and ordered matching ratio | PASS |  |
| Folium HTML generation | Deliverables p6 | results/maps/eda_example.html and prediction_example.html; genuine Folium core | PASS |  |
| Folium screenshot in report | Deliverables p6 | HTML generated, browser capture prohibited/unavailable | FAIL | Capture the actual map in an authorised browser and include screenshot |
| FastAPI implementation and schemas | Models pp5-6 | api.py; real validated handler tests for all ten accessible trips | PASS |  |
| Live HTTP integration | Models pp5-6 | Socket restrictions block loopback and TestClient event loop | UNVERIFIED | Run API/UI HTTP tests in an unrestricted local terminal |
| Evaluator UI code | Main Guidelines p2 | ui/index.html with real fetch, loading/error states, SVG geometry | PASS | Browser rendering and interactions still need verification |
| UI screenshots and rendering | Deliverables pp6-7 | No screenshot fabricated; browser control stopped by URL-verification restriction | FAIL | Capture working interface during actual demo |
| API log screenshots | Deliverables pp6-7 | Actual handler log exists; no live-network log screenshot | FAIL | Capture actual receiving-laptop API logs |
| Physical two-laptop deployment | Main Guidelines p2 | README exact-origin and network setup instructions | UNVERIFIED | Run on two permitted physical laptops |
| Reserved evaluator trips | Models pp5-6 | Dataset unavailable; evaluator mode lists zero trips without fallback | UNVERIFIED | Add reserved trips and test all ten |
| Reference isolation | Assessment p3 | Inference function has no reference access; references read after prediction; response privacy checks | PASS |  |
| Reproducibility | Assessment pp1-3 | Saved metrics reproduced; independently refitted model matches predictions | PASS |  |
| Dependency clean installation | User final audit | Runtime verified; pip network unavailable; vendor mapping sources licensed | UNVERIFIED | Install requirements in fresh environment and rerun tests |
| Public GitHub repository | Deliverables pp1-2 | https://github.com/ArundhathiK29/PathSnap at 38b65877e864400975d73d308b1766c1a64c3088; 94 file blob hashes verified | PASS |  |
| Consolidated PDF | Deliverables pp2-7 | report.py generates actual-results report; mandatory screenshots absent | FAIL | Add genuine screenshots and final deployment evidence |
| Independent work/provenance | Assessment pp2-3 | DECISIONS, PROVENANCE, source manifests and experiment histories | PASS | Team must review, explain, reproduce and disclose AI assistance |
| External plagiarism verification | User final audit | No checker run; no percentage invented | UNVERIFIED | Optional external checker if available |
| Security review | User final audit | SECURITY_AUDIT; path validation; no credentials; no protections changed | PASS |  |