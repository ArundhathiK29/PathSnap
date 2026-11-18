# Provenance

Project implementation created with ChatGPT assistance in this session for Arruuu. No other team's solution, weights, results or configurations were used. The team must review, understand, reproduce and disclose assistance according to event rules.

Official requirements: five PES University PathSnap PDFs, extracted to ../official. Dataset: provided student ZIP, 80 training and 10 student-test trips. Reserved evaluator data not supplied.

Standard methods used: point-to-segment projection; directed weighted graph shortest paths (SciPy); trip-grouped cross-validation; scikit-learn classifiers; Gaussian-process Bayesian optimisation with expected improvement; log-domain Viterbi dynamic programming. Third-party library code is imported, not claimed as original. Exact installed versions recorded by workflow.

Reference video could not be accessed. External plagiarism-checker verification has not been performed.

Mapping dependency fallback: vendored unmodified upstream Folium core from python-visualization/folium commit 5c460674cbe9c2c20f2b81fd3233c26e0a42676e (v0.20.0), Branca from python-visualization/branca commit 0590552ac97729b9fd06dc82e20be433c55fe8fc, and xyzservices from geopandas/xyzservices commit 6036e8174b16cf7294b99340cd3437196f7dd011. Retrieved with the connected GitHub read interface after pip could not access package indexes. Licences included in vendor/. These are third-party library sources, not project originality claims. Branca's templates/color_scale.js is additionally included at the same pinned commit.
