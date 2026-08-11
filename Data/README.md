# Data package

This directory contains the small fixed-target tables used by the public
workflow. The principal datasets are `E288_200`, `E288_300`, `E288_400`,
`E605`, and `E772`.

The corrected row-99 variants are under:

```text
Data/v23a_fixed_target_lowQ_row99_variants/
```

The production-style table is:

```text
Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99_normpriors15_p2p5_E772_E288400/
```

Its manifest and summary files document the explicit normalization priors and
point-to-point uncertainty columns. The table is a staged analysis input, not
a claim that every accelerator/collider dataset is production-ready.

Run the environment/data inspection helper from the repository root with:

```bash
bash Data/data_check.py
```
