# v22 audits

Read-only backend, grid, warm-check, subtraction, profile, and replica audits.
Run them from the repository root with `PYTHONPATH=.` and pass explicit input
paths.  They write tagged JSON/CSV reports; they never promote or overwrite a
production result.  The stage order and gates are documented in
[`../README.md`](../README.md).
