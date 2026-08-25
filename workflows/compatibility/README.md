# Compatibility and staging helpers

`install_into_repo.sh` is a small source-staging helper retained for users who
need to copy this public release into an existing working checkout:

```bash
./workflows/compatibility/install_into_repo.sh /path/to/working/checkout
```

It mirrors the current checkout with `rsync`; inspect the destination and use
a separate branch before accepting any overwrite.  It is not part of a fit
and does not copy generated `outputs/`, caches, checkpoints, or external
engines unless they are present in the source checkout.

The compatibility backend
`bt_internal_css_backend_v19_smoothprofile.py` remains at the repository root
on purpose.  `v22/backends/bt_internal_css_backend_v22_scheme_y.py` imports it
by that stable path, and several source-only provenance scripts audit the same
path.  Moving it would change the backend identity and invalidate manifests.
