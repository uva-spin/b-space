# v22 perturbative backend

The v22 package is the fixed-target perturbative source layer. It contains the
CSS2 NLO OPE, DY hard factor, convolution reference, Sudakov/profile helpers,
NLO DY luminosity/W assembly, fixed-target backend, scheme-consistent
W-expansion wrapper, tests, and audits.

Start with:

```bash
PYTHONPATH=. python -m pytest -q v22/tests/test_conventions.py
PYTHONPATH=. python v22/tests/run_convolution_smoke.py
PYTHONPATH=. python v22/tests/run_css2_ope_nlo_smoke.py
PYTHONPATH=. python v22/tests/run_css2_ope_nlo_general_smoke.py
PYTHONPATH=. python v22/tests/run_dy_hard_nlo_smoke.py
PYTHONPATH=. python v22/tests/run_dy_w_nlo_reference_smoke.py
PYTHONPATH=. python v22/tools/print_convention_summary.py
```

The v22 backend is the perturbative layer used by the DNN trainer. It does not
contain fitted checkpoints or generated replica caches. See
[`docs/MATCHING.md`](../docs/MATCHING.md) for the W/OPE/Y distinction and
[`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for the complete fit
workflow.
