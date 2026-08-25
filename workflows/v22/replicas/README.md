# v22 replicas and freeze

Replica launch, continuation, and freeze drivers.  Start with a short pilot,
verify every seed's plateau/status and audit files, then append disjoint seed
blocks.  `freeze_v22_lambda3_50rep_bspace.sh` copies source and artifacts into
a new tagged freeze and refuses an existing destination.  See the full
protocol in [`../README.md`](../README.md).
