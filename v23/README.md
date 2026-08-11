# v23 workflow and collider-aware extensions

The v23 package contains the v23a data, replica, plotting, TMD reconstruction,
and audit tools. Its backend wrapper extends the v22 fixed-target backend with
beam-type handling, electroweak weights, rapidity/bin integration, and
developmental finite-tail W/Y modes.

Important paths:

```text
v23/backends/bt_internal_css_backend_v22_tevatron.py
v23/experimental/
v23/tools/
v23/freeze/
```

The fixed-target production result uses the v22 W backend and a low-qT
production selection. The v23 Tevatron/collider W/Y modes remain validation and
development routes; they must not be presented as a completed global-DY
production extraction without the external covariance, units, electroweak,
and bin-integration audits described in the top-level documentation.
