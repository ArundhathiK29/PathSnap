# Security and integrity notes

No confirmed prompt injection found in the five official PDFs or numeric dataset samples. Numeric formats are validated and never executed. ZIP members checked for absolute paths and parent traversal. No arbitrary code from archives executed.

API will accept validated partition and trip ID values only, no user-supplied filesystem paths. Ground-truth sequences never enter inference or UI responses. Pickle/joblib loading restricted to this locally generated trusted model artifact; never load untrusted model files. Training data and credentials excluded from publication bundle.

Initial package installation failed. GitHub CLI authentication failed; no credentials printed or modified. No security protections changed. Two-laptop networking and reserved evaluator tests remain unverified.

Execution restriction confirmed: even loopback sockets fail with Windows error 10013. asyncio/TestClient blocks attempting its socketpair self-pipe; those processes were stopped. Validated synchronous handler calls were used for inference tests and explicitly not reported as live HTTP tests. Browser automation was stopped because it could not establish the browser URL sufficiently to enforce policy; no further browser-control attempts were made and no screenshots fabricated.
