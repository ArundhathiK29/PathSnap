# PathSnap

Candidate-road classification and directed-network HMM/Viterbi route reconstruction for PES University UE24CS352A.

## Current evidence

See `FINAL_AUDIT.md`, `PROJECT_STATE.json`, `results/` and `report.pdf`. Scores are **reference-derived weak-label agreement**, not independently verified pointwise accuracy. `.route` is chronological, not one label per GPS row. Do not conceal this distinction during the viva.

## Setup (Windows PowerShell)

Use Python 3.11. Extract the supplied dataset ZIP separately so its `train` and `student_test` folders share a parent. Create a virtual environment and install `requirements.txt` when network access is available. The included `vendor/` contains attributed upstream Folium core, Branca and xyzservices sources used when package installation was unavailable; it is not team-authored code. System dependencies must still be installed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PATHSNAP_DATA = 'D:\Student folder\Student folder'
python workflow.py prepare --data "$env:PATHSNAP_DATA"
python workflow.py train --data "$env:PATHSNAP_DATA"
python workflow.py optimise --data "$env:PATHSNAP_DATA" --trials 10
python workflow.py decode --data "$env:PATHSNAP_DATA"
python -m pytest tests -q
python verify.py --data "$env:PATHSNAP_DATA"
python report.py
```

`train` resumes completed model records. To intentionally retrain all models, use a fresh output folder/project copy rather than mixing experiment histories. Default training sample: up to 240 confident observations per training trip; every candidate for selected observations is retained. All ten student-test trips are evaluated at their full GPS length, with confident-label coverage explicitly reported. Seed: 2911. CV: three folds grouped by trip.

## Development demo on one laptop

Open two terminals in this folder:

```powershell
$env:PATHSNAP_DATA = 'D:\Student folder\Student folder'
$env:PATHSNAP_MODE = 'development'
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

```powershell
python -m http.server 8090 --directory ui --bind 127.0.0.1
```

Visit http://127.0.0.1:8090 . Backend health: http://127.0.0.1:8000/health ; API docs: http://127.0.0.1:8000/docs . Development mode lists the ten student-test trips and labels them clearly.

## Required two-laptop demonstration

Laptop 1 hosts the dataset, model and API. Laptop 2 hosts the UI. Both must be on a permitted reachable network. Obtain their actual local addresses using `ipconfig`.

On Laptop 1 set `PATHSNAP_UI_ORIGINS` to Laptop 2's exact UI origin, e.g. `http://192.168.1.20:8090`, then run `python -m uvicorn api:app --host 0.0.0.0 --port 8000`. On Laptop 2 run `python -m http.server 8090 --directory ui --bind 0.0.0.0`. Visit its own UI origin and enter Laptop 1's reachable backend address. Observe real POST /predict logs on Laptop 1. Do not expose this unauthenticated demonstration service to the public Internet. Do not disable firewalls; permit only the required private-network application traffic if approved.

Physical two-laptop networking is not established merely by the local tests. See the audit for its actual verification status.

## Reserved evaluator mode

Place the ten supplied reserved trip directories under `PATHSNAP_DATA/evaluator_test/`. Each contains `.track`, `.nodes`, `.arcs`, and optionally server-side `.route` references. Set `PATHSNAP_MODE=evaluator` before starting the backend. Empty evaluator partition returns zero trips, **never** substituted student-test choices. No reserved trip IDs or predictions are fabricated.

Request: `POST /predict` with `{"trip_id":"00000000"}`. The configured partition controls availability. Reference data is read only after prediction; UI receives aggregate evaluation metrics, not reference sequences. Current scorer uses documented chronological reference alignment; reconcile with any organiser-supplied final scoring protocol before claiming official accuracy.

## Layout

`core.py`: metre geometry, candidates, features, weak labels, graph/HMM/Viterbi, route stitching.
`workflow.py`: EDA, five models, CV, GP expected-improvement search, decoder comparison.
`api.py`: FastAPI and server-side evaluation isolation.
`ui/`: dependency-free SVG evaluator UI (works without map tiles).
`vendor/`: licensed upstream mapping dependencies and exact source manifest.
`tests/`, `verify.py`: algorithm and actual-model/API verification.
`results/`: actual metrics, folds, trials, predictions, plots and Folium maps.
`artifacts/`: trusted locally generated models and final configuration.
`report.py`, `report.pdf`: report generator and consolidated report.

## Model and algorithm notes

Five classifiers: scaled logistic regression, scaled Gaussian Naive Bayes, decision tree, random forest and histogram gradient boosting. They predict binary road-candidate compatibility using the same 15 non-reference features. Candidates are exactly reranked after spatial retrieval from sampled segments and midpoint fallback. The selected model's scores are normalised across candidates for HMM emissions. These normalised compatibility scores are not claimed to be perfectly calibrated posterior probabilities.

The HMM uses candidate arc states, row-normalised network-distance transition probabilities, uniform initial prior, log-domain dynamic programming and backtracking. Disconnected transitions have zero probability; the chain restarts if all transitions fail or a timestamp gap exceeds 30s. Directed shortest paths bridge plausible intermediate road segments for display. Gaps over 30s remain explicit breaks.

## Submission and integrity

Publish the complete source/artifacts/results as a public GitHub repository and submit `report.pdf` as the second official deliverable. This folder/bundle is repository-ready but is not proof of publication. Do not publish raw datasets without confirming redistribution permission. Never commit credentials. All source code was developed with AI assistance; the team must review, reproduce and disclose that assistance as required. External plagiarism-checker verification has not been performed.
