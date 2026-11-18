# PathSnap requirements and evidence checklist

Official PDFs extracted from D:/PathSnap_student_materials.zip; read in full through uploaded document text. Dataset ZIP inspected and extracted local dataset inspected. Reference video inaccessible; no contents inferred.

| ID | Requirement | Official source | Planned evidence |
|---|---|---|---|
| R01 | Inventory, missing values, timestamps, geography, topology | Dataset Exploration pp.1-4 | results/inventory.csv, eda.json, figures |
| R02 | Defensible observation/reference association; no row correspondence assumption | Dataset Exploration pp.2-3 | DECISIONS.md, alignment diagnostics |
| R03 | Independent candidate generation and feature engineering | Main Guidelines p.2; Models pp.1-2 | core.py and feature dictionary |
| R04 | Five distinct models and measured comparison | Models pp.1-4; Deliverables pp.3-4 | experiment logs, metrics.csv |
| R05 | Trip-grouped CV on 80 train trips; ten student-test trips | Dataset Exploration p.3; Models pp.2,4 | CV fold manifests, metrics |
| R06 | Bayesian search with complete logs, tuned comparison | Models p.4; Deliverables p.5 | Gaussian-process expected-improvement search |
| R07 | HMM/Viterbi, candidate mapping, gaps and disconnected transitions | Models pp.4-5; Deliverables pp.5-6 | decoder, tests, per-trip metrics |
| R08 | Folium maps with GPS, roads, reference and prediction | Deliverables p.6 | interactive HTML and report figures |
| R09 | FastAPI on Laptop 1 and UI on Laptop 2; logs | Main Guidelines p.2; Models pp.5-6 | API tests, deployment guide |
| R10 | Ten reserved choices; labels server-only, no hardcoded output | Models pp.5-6; Assessment p.3 | evaluator mode, response tests |
| R11 | Public GitHub with complete code/artifacts/README | Deliverables pp.1-2 | repository URL and visibility verification |
| R12 | Consolidated PDF with real experimental tables/plots/screenshots | Deliverables pp.2-7 | report.pdf |
| R13 | Independent implementation, actual results, explainable decisions | Assessment pp.1-3 | PROVENANCE, DECISIONS, logs |
| R14 | First evaluation before lunch | Deliverables p.1 | human event checkpoint; exact criteria unspecified |

Additional user requirements: final audit matrix; honest deployment limitations; AI assistance disclosure; no invented plagiarism verification.
