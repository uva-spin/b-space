# Publishable trainer sources

This directory contains the two trainer sources needed by the v22/v23
orchestration scripts:

```text
train_bt_dnn_v21_smoothedA_tail.py
train_bt_dnn_v21_replica_stable.py
prepare_v21_replica_norm_inits.py
```

They are intentionally published without the large historical candidate
directory, checkpoints, logs, and replica outputs. Pass the backend explicitly
when running from another location:

```bash
python v21_tail_release_amp0p019_candidate/train_bt_dnn_v21_smoothedA_tail.py \
  --backend-script v22/backends/bt_internal_css_backend_v22_full.py \
  --data-dir Data/v23a_fixed_target_lowQ_row99_variants/corrected_E288_300_99_normpriors15_p2p5_E772_E288400 \
  --help
```

The trainer learns the constrained nonperturbative factor. It does not replace
the perturbative backend, OPE matching, hard factor, or W/Y construction.
