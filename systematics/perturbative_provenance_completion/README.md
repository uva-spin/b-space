# Perturbative and provenance completion workspace

This is an isolated workspace for finishing and documenting the perturbative
standard used by the fixed-target b-space TMD extraction.  It is deliberately
separate from the active production package, the frozen upstream outputs, and
the PRD manuscript.

The immediate problem is provenance, not a new fit:

1. determine whether the production W cache was generated with the strict or
   multiplicative NLO organization;
2. make that choice explicit in a regenerated diagnostic cache;
3. inventory the actual Sudakov, hard, and OPE perturbative content;
4. compare that inventory with the precise N3LL/N3LL-prime convention intended
   for the paper; and
5. implement and audit any missing terms rather than weakening the manuscript
   claim.

No file in this directory may overwrite a production cache or frozen result.
Every generated artifact must carry a new tag and a source/configuration hash.

See [HANDOFF.md](HANDOFF.md) for the current state and gate definitions.
