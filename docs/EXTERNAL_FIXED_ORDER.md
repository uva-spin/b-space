# MCFM and DYTurbo fixed-order and W+Y workflow

This document is the operator-level recipe for the external fixed-order
calculations used in the isolated Tevatron and LHCb studies. It complements
MATCHING.md, which defines the TMD formulas and numerical conventions. The
external engines are not distributed in this repository: they are versioned
local installations, and every generated card, log, table, and
source-provenance record must be archived with the study that uses it.

The workflow has three deliberately separate roles:

1. DYTurbo provides the conventional fixed-order V+jet term and the
   conventional resummed-plus-fixed-order W+Y candidate.
2. MCFM provides an independent fixed-order Z+jet benchmark with the same
   bin, mass, beam, and scale inputs. It is a cross-check, not the N3LL W+Y
   oracle.
3. The Python scripts in systematics/ generate cards, run the executable,
   parse the result, convert units, and write fail-closed status metadata.

No command in this guide overwrites the frozen fixed-target production
package. Use a new output directory for every candidate or precision trial.

## 1. External prerequisites

The public checkout supplies the Python wrappers and small test data tables,
but not the external binaries or PDF grids. The production work used the
following local installations; different absolute paths are supported through
command-line options.

### Python and PDFs

From the repository root:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
~~~

Install LHAPDF and make the PDF set NNPDF40_nnlo_as_01180 visible to both
engines. The Python backend expects ordinary PDFs; if a wrapper receives an
xfxQ value it divides by x before using it as a density. Do not silently
substitute a PDF set or member: record the exact set, member, alpha_s provider,
and LHAPDF version in the run manifest.

### DYTurbo

The archived studies used DYTurbo 1.4.2. A valid installation must provide:

~~~text
DYROOT/bin/dyturbo       executable
DYROOT/lib/               runtime libraries, if the build uses them
~~~

The default paths in the archived scripts are /home/dustin/src/dyturbo-1.4.2
and /home/dustin/src/dyturbo-1.4.2/bin/dyturbo. For another installation,
pass both --dyturbo and --dyturbo-root; do not change a card by hand and
forget to record the change. The executable must be runnable from its root,
and the runtime library path should include both the active Python/conda
library directory and DYROOT/lib:

~~~bash
export DYROOT=/path/to/dyturbo-1.4.2
export DYTURBO="$DYROOT/bin/dyturbo"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYROOT/lib:$LD_LIBRARY_PATH"
"$DYTURBO" --help 2>&1 | head
~~~

The public source map records the DYTurbo implementation files inspected for
the order and switch semantics:
systematics/full_n3ll_wy_production_2026/scripts/build_dyturbo_n3ll_source_map.py.
The source map is provenance, not a source patch; the external engine remains
an input that must be archived or checksummed separately.

### MCFM

The independent benchmark used MCFM 10.3. A valid installation must provide:

~~~text
MCFM_BIN/mcfm             executable
MCFM_BIN/PDFs/            MCFM grid/PDF support files
~~~

The historical default was /home/dustin/work/MCFM-10.3/Bin. The wrapper also
needs LHAPDF_DATA_PATH when the selected MCFM build uses LHAPDF grids:

~~~bash
export MCFM_BIN=/path/to/MCFM-10.3/Bin
export LHAPDF_DATA_PATH=/path/to/lhapdf/share/LHAPDF
"$MCFM_BIN/mcfm" --help 2>&1 | head
~~~

MCFM is run with an unlimited stack, one OpenMP thread, the selected seed, and
PATH=/usr/bin:/bin in the archived reproducibility wrappers. This avoids a
machine-specific shell environment changing the integration result.

## 2. Input table and units

The working-tree campaign used the collider candidate tables under:

~~~text
Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/
  CDF_RUN_1.csv
  CDF_RUN_2.csv
  D0_RUN_1.csv
~~~

Those collider tables are archived inputs and are intentionally not committed
to the compact public checkout. Obtain the matching archived data directory
before running the commands below, then replace the Data/... path in each
command with that directory. The directory manifest and row identities must
match the campaign record; do not mix a CDF/D0 table from one candidate with a
LHCb table or covariance file from another.

Tevatron rows contain at least row_id, SqrtS, qT_low, qT_high, QM_Low,
QM_High, and the fit-ready cross section/error columns. LHCb rows also carry
the lepton rapidity and transverse-momentum acceptance fields. The
fixed-target quadrature probe uses the separate
Data/v23a_fixed_target_plus_tevatron_absolute_fit_ready/ tables.

All energies and bin edges in cards are GeV. The engines integrate over the
complete requested qT, rapidity, and invariant-mass bins. DYTurbo text tables
are normally reported in fb per bin; the wrappers convert them as

~~~text
value_pb_per_GeV = value_fb_per_bin / qT_bin_width / 1000
~~~

MCFM output may be labelled pb or fb depending on the build and output path;
the MCFM wrapper detects the unit, converts fb to pb, and divides by the qT
bin width. Never compare a raw engine number with a fit-ready point until the
unit and bin-width conversion have been recorded.

Before a grid run, verify that qT bins are contiguous and ordered. The full
DYTurbo grid refuses gaps, duplicate row identities, non-finite values, or a
wrong number of output bins.

## 3. DYTurbo fixed-order benchmark

The canonical Tevatron fixed-order wrapper is
v23/tools/run_tevatron_dyturbo_benchmark.py. It writes generated cards, logs,
parsed tables, and a summary under the requested output directory. A
single-row smoke test is:

~~~bash
PYTHONPATH=. python v23/tools/run_tevatron_dyturbo_benchmark.py \
  --data Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/CDF_RUN_2.csv \
  --rows CDF_RUN_2:17 \
  --out /tmp/bspace-dyturbo-cdf17 \
  --dyturbo "$DYTURBO" \
  --dyturbo-root "$DYROOT" \
  --pdf-set NNPDF40_nnlo_as_01180 \
  --pdf-member 0 \
  --mu-r-factor 1.0 \
  --mu-f-factor 1.0 \
  --cores 4 \
  --timeout 900
~~~

The generated Tevatron card is a fixed-order V+jet benchmark. Important
defaults are:

~~~text
beam: ih1=+1, ih2=-1, nproc=3, sqrt(s)=row.SqrtS
order=1, fixedorder_only=true, primed=true
doVJ=true; doBORN/doCT/doVJREAL/doVJVIRT=false; VJquad=true
PDF=NNPDF40_nnlo_as_01180, member=0
mu_r = row.QM * mu_r_factor; mu_f = row.QM * mu_f_factor
qt_bins = [row.qT_low, row.qT_high]
y_bins  = [-5, 5]
m_bins  = [row.QM_Low, row.QM_High]
makecuts=false
~~~

The remaining benchmark card entries are kept explicit rather than inherited
from a user installation. They include the electroweak input
(G_F=1.1663787e-5, mZ=91.1876, mW=80.385, sin2thetaW=0.23153,
alpha_em(mZ)=7.7585538055706e-3, Z width=2.4950, W width=2.091), the
DYTurbo running-width and photon settings, qT cutoff xqtcut=0.008, and the
integration controls below:

~~~text
fmuren=1, fmufac=1, fmures=1
kmuren=mu_r_factor, kmufac=mu_f_factor, kmures=1
rseed=seed, mcutoff=1e-3
intDimVJ=3, vegasncallsVJLO=10000000
vegasncallsBORN=1000, vegasncallsCT=100000
vegasncallsVJREAL=100000, vegasncallsVJVIRT=100000
vegascollect=true, vegascorr=false
pcubature=true, relaccuracy=1e-3, absaccuracy=0, level=3
threading=0, cores=chosen core count, cubanbatch=1000
texttable=true, redirect=false, silent=false
force_binsampling=true, ptbinwidth=false, ybinwidth=false, mbinwidth=false
~~~

The wrapper changes only the row-dependent values and command-line overrides;
it does not rely on a hidden global card. A generated card is therefore the
configuration file to cite in a result, not this prose default summary.

The wrapper writes cards/, logs/, and tables/, removes stale DYTurbo text
output before each attempt, runs from DYROOT, and parses the first finite
table value and its integration uncertainty. This benchmark is useful for
checking the V+jet normalization and bin convention; it is not a claim of
N3LL accuracy.

For an LHCb fiducial benchmark use
systematics/finite_y_tail_benchmark/scripts/run_lhcb7_dyturbo_benchmark.py:

~~~bash
PYTHONPATH=. python systematics/finite_y_tail_benchmark/scripts/run_lhcb7_dyturbo_benchmark.py \
  --data Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv \
  --rows LHCb_7:10 \
  --out /tmp/bspace-dyturbo-lhcb10 \
  --dyturbo "$DYTURBO" \
  --dyturbo-root "$DYROOT"
~~~

That card switches to pp beams (ih1=ih2=1), keeps the same electroweak and
scale conventions, and enables the measured fiducial cuts:

~~~text
makecuts=true
each lepton: pT >= 20 GeV and 2 <= rapidity <= 4.5
inclusive anti-kT jet/qT bin and the row's invariant-mass window
~~~

This LHCb script is an acceptance and fixed-order diagnostic. It is not the
finite-Y production path; the LHCb subtraction and covariance closure remain
outside the promoted production scope.

## 4. MCFM independent benchmark

The canonical Tevatron MCFM wrapper is
v23/tools/run_tevatron_mcfm_benchmark.py:

~~~bash
PYTHONPATH=. python v23/tools/run_tevatron_mcfm_benchmark.py \
  --data Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/CDF_RUN_2.csv \
  --rows CDF_RUN_2:17 \
  --out /tmp/bspace-mcfm-cdf17 \
  --mcfm-bin "$MCFM_BIN" \
  --mcfm-exe "$MCFM_BIN/mcfm" \
  --lhapdf-data "$LHAPDF_DATA_PATH" \
  --pdf-set NNPDF40_nnlo_as_01180 \
  --pdf-member 0 \
  --mu-r-factor 1.0 \
  --mu-f-factor 1.0 \
  --calls 1000000 \
  --seed 246810 \
  --timeout 1200
~~~

The generated MCFM 10.3 card uses nproc=41, part=lo, pbar-p beams
(ih1=+1, ih2=-1), and an inclusive anti-kT jet bin. The row's qT and mass
edges are inserted literally. Lepton cuts are disabled for the Tevatron
benchmark so the observable is comparable to the inclusive table. The card
sets renscale=row.QM*mu_r_factor and facscale=row.QM*mu_f_factor, with scale
variation disabled for the central benchmark. MCFM resummation grids are
enabled only for its own internal setup; this run is interpreted as an
independent fixed-order Z+jet reference.

Integration controls in the card are explicit: Sobol sampling, the requested
seed, precisiongoal=.003, no intermediate read/write, and a central call
count of one million (initcallslord). The wrapper removes stale total-cross
section files, runs with ulimit -s unlimited and OMP_NUM_THREADS=1, parses the
final cross section and unit, and records the conversion to pb/GeV.

For reference, the central MCFM card is organized as follows:

~~~text
mcfm_version=10.3, writerefs=false
[general] nproc=41, part=lo, sqrts=row.SqrtS
          ih1=+1, ih2=-1, zerowidth=false, removebr=false, ewcorr=none
[resummation] usegrid=true, makegrid=false, res_range=0..80
              resexp_range=1..80, fo_cutoff=1, transitionswitch=.4
[lhapdf] lhapdfset=NNPDF40_nnlo_as_01180, member=0, dopdferrors=false
[scales] dynamicscale=none
         renscale=row.QM*mu_r_factor, facscale=row.QM*mu_f_factor
         doscalevar=false, maxscalevar=6
[basicjets] inclusive=true, algorithm=ankt
             ptjetmin=row.qT_low, ptjetmax=row.qT_high, etajetmax=99, Rcutjet=.5
[masscuts] m34min=row.QM_Low, m34max=row.QM_High
[cuts] makecuts=false for Tevatron; all lepton and missing-pT cuts open
[histogram] writetxt=true, newstyle=true
[integration] initcallslord=calls, usesobol=true, seed=seed
              precisiongoal=.003, readin=false, writeintermediate=false
~~~

The additional MCFM integration budgets in the archived card are
initcallsnlo real=1,000,000, virtual=200,000, NNLO below=200,000, NNLO
virtual above=400,000, NNLO real above=2,000,000, and resummed components of
1,000 and 200,000 calls. They are retained in the generated card even though
the benchmark is interpreted as a fixed-order reference.

For an LHCb fiducial MCFM benchmark use:

~~~bash
PYTHONPATH=. python systematics/finite_y_tail_benchmark/scripts/run_lhcb7_mcfm_benchmark.py \
  --data Data/v23a_tevatron_plus_lhcb7_fiducial_candidate/LHCb_7.csv \
  --rows LHCb_7:10 \
  --out /tmp/bspace-mcfm-lhcb10 \
  --mcfm-bin "$MCFM_BIN" \
  --mcfm-exe "$MCFM_BIN/mcfm" \
  --lhapdf-data "$LHAPDF_DATA_PATH"
~~~

The LHCb card changes to pp beams, nproc=41, and enables the two-lepton
acceptance (20 GeV, rapidity 2 to 4.5) together with the row mass window.
The LHCb wrapper and DYTurbo wrapper must use the same row definition before
their numbers are compared. In particular, an absolute-rapidity MCFM result
can differ by a factor of two from a single positive-rapidity-arm convention;
do not rescale the data or a table until the convention is proven from the
card and integration limits.

## 5. Row-isolated external campaigns

To run a controlled set of benchmark rows without mixing stale or failed
outputs, use the campaign orchestrator:

~~~bash
PYTHONPATH=. python systematics/high_qt_direct_production_benchmark/scripts/run_external_campaign.py \
  --tier tier1_boundary \
  --codes dyturbo mcfm \
  --datasets CDF_RUN_2 \
  --limit 1 \
  --keep-going \
  --dyturbo-cores 4 \
  --dyturbo-timeout 900 \
  --mcfm-calls 1000000 \
  --mcfm-timeout 1200
~~~

The orchestrator selects the LHCb-specific or Tevatron-specific runner from
the dataset name, launches each row in an isolated directory, and writes a
JSONL campaign log. The three tiers are boundary/high-qT/exceptional test
sets listed in summaries/benchmark_batch_plan.csv. Completed summaries are
skipped unless --rerun is supplied. Its output root is the study's
systematics/high_qt_direct_production_benchmark/outputs/ tree; keep the
per-row cards/, logs/, and tables/ directories because they are part of the
provenance, not disposable cache. The orchestrator expects the archived
candidate data at the working-tree path named above; for a different archive,
stage or adapt that input path and record the change.

## 6. Conventional N3LL+NNLO W+Y candidate

The full conventional candidate is generated by
systematics/full_n3ll_wy_production_2026/scripts/run_tevatron_full_n3ll_nnlo_grid.py.
It uses one card containing all contiguous qT bins for each dataset:

~~~bash
PYTHONPATH=. python systematics/full_n3ll_wy_production_2026/scripts/run_tevatron_full_n3ll_nnlo_grid.py \
  --datasets CDF_RUN_1 CDF_RUN_2 D0_RUN_1 \
  --g1 1.017 \
  --calls 1000000 \
  --seed 246810 \
  --timeout 7200 \
  --out /tmp/bspace-tevatron-full-wy
~~~

The script first imports the canonical DYTurbo card builder, then applies the
candidate's production-level switches:

~~~text
fixedorder_only = false
order           = 3
primed          = false
doBORN          = true
doCT            = true
doVJ            = true
doVJREAL        = true
doVJVIRT        = true
VJquad          = false
intDimVJ        = -1
makecuts        = true
~~~

The candidate also raises the component Vegas calls (Born, counterterm, real
V+jet, and virtual V+jet) from their benchmark values to the requested
--calls, inserts the full contiguous qT edge list, preserves the row mass and
rapidity bins, and leaves the PDF and scale factors explicit in the card. The
DYTurbo output is raw fb/bin and is converted to pb/GeV only after a
successful row-count and finiteness check.

For unprimed order=3, the DYTurbo source map records that the resummation is
N3LL and the V+jet fixed-order order is NNLO (order_vjet=2 internally). The
term interpretation used by the study is:

~~~text
W   = RES
ASY = -CT
FO  = VJ
Y   = FO - ASY = VJ + CT
W+Y = RES + CT + VJ
~~~

This is an algebraic decomposition of the same DYTurbo card, not a mixture of
independent runs. A positive full W+Y table, finite MC uncertainties, and term
closure are required before a candidate is even considered for a fit. The
output contains the generated card, log, parsed grid, and grid_status.json;
it does not authorize production promotion.

The initial million-call grid is a screening calculation. Rows where RES, CT,
and VJ nearly cancel require controlled refinements, typically 30 million and
then 100 million calls, with the same card, seed convention, and output
validation. Refinements must be logged as new candidates rather than
replacing the first grid in place.

### Term decomposition check

For one row, run the independent term diagnostic:

~~~bash
PYTHONPATH=. python systematics/full_n3ll_wy_production_2026/scripts/run_tevatron_wy_term_decomposition.py \
  --dataset CDF_RUN_2 \
  --row-id CDF_RUN_2:17 \
  --g1 1.0 \
  --calls 3000000 \
  --out /tmp/bspace-wy-terms
~~~

The script writes separate RES, CT, and VJ cards and checks that their sum
reconstructs the all-terms convention within the propagated Monte Carlo
uncertainty. It reports the raw fb/bin values and the fit-ready pb/GeV value.
Use identical PDF, scale, bin, seed, and integration settings for all three
terms. A term sum that fails closure is a numerical/configuration failure, not
evidence for changing the matching formula.

## 7. Fixed-target quadrature and nuclear targets

The fixed-target exploratory quadrature driver is
systematics/full_n3ll_wy_production_2026/scripts/run_fixed_target_quadrature_probe.py.
It is intentionally separate from the Tevatron production candidate:

~~~bash
PYTHONPATH=. python systematics/full_n3ll_wy_production_2026/scripts/run_fixed_target_quadrature_probe.py \
  --dataset E288_200 \
  --row E288_200:0 \
  --g1 1.017 \
  --intdim-vj 3 \
  --calls 1000000 \
  --seed 20260821 \
  --out /tmp/bspace-fixed-target-quadrature
~~~

For a nuclear target, pass both --target-z and --target-a, for example
--target-z 6 --target-a 12. The driver inserts DYTurbo's nuclearpdf=true
card branch and records the target counts. DYTurbo's nuclear-PDF result is a
whole-nucleus proton-plus-neutron sum; divide by A before a per-nucleon
comparison. The fit-ready cross-section convention used in the handoff then
applies its recorded A/PreFactor conversion exactly once.

VJquad=true is a diagnostic integration method. The default no-cut probe uses
intDimVJ=3; the cut-aware alternative is requested with --makecuts and
--intdim-vj 5. The full finite-Y candidate uses non-quadrature integration
(VJquad=false) and a DYTurbo dimension selected for the active cuts. Do not
substitute the low-dimensional quadrature result into a production grid
without a dedicated closure and unit audit.

## 8. Reproducibility controls and fail-closed checks

For every engine run, archive:

- the exact generated card, executable path/version, external source map, and
  PDF/LHAPDF metadata;
- the command-line arguments, environment/library paths, seed, core count,
  Vegas/Sobol call counts, and timeout;
- raw engine output, parsed table, unit conversion, and qT-bin widths;
- row identifiers, mass/rapidity cuts, beam signs, and scale factors;
- integration uncertainties and an independent-seed repeat for any row that
  contributes appreciably to a cancellation;
- a status JSON that says whether the result is a benchmark, diagnostic, or
  candidate and whether any production promotion is authorized.

The wrappers fail closed when an output is missing, stale, partial, nonfinite,
has the wrong row count, has noncontiguous qT edges, or contains a nonpositive
full W+Y value. A successful process exit alone is not a physics validation.
For a conventional W+Y candidate also require:

~~~text
full W+Y = RES + CT + VJ within integration uncertainty
FO, ASY, and W use the same bins, cuts, PDF, scales, and electroweak inputs
MC uncertainties are small compared with the reported data/model comparison
the result is labelled isolated until the fit and covariance closure pass
~~~

The six LHCb rows retained in the larger 2026 diagnostic are W-only rows. They
are useful for locating acceptance and finite-Y problems, but they are not
silently converted into a universal finite-Y production prediction.

## 9. Source map

| Task | Source of truth |
| --- | --- |
| Tevatron DYTurbo fixed-order card and parser | v23/tools/run_tevatron_dyturbo_benchmark.py |
| Tevatron MCFM 10.3 card and parser | v23/tools/run_tevatron_mcfm_benchmark.py |
| LHCb DYTurbo fiducial benchmark | systematics/finite_y_tail_benchmark/scripts/run_lhcb7_dyturbo_benchmark.py |
| LHCb MCFM fiducial benchmark | systematics/finite_y_tail_benchmark/scripts/run_lhcb7_mcfm_benchmark.py |
| Row-isolated campaign orchestration | systematics/high_qt_direct_production_benchmark/scripts/run_external_campaign.py |
| Full Tevatron N3LL+NNLO grid | systematics/full_n3ll_wy_production_2026/scripts/run_tevatron_full_n3ll_nnlo_grid.py |
| DYTurbo N3LL switch/source inventory | systematics/full_n3ll_wy_production_2026/scripts/build_dyturbo_n3ll_source_map.py |
| RES/CT/VJ closure | systematics/full_n3ll_wy_production_2026/scripts/run_tevatron_wy_term_decomposition.py |
| Fixed-target quadrature/nuclear probe | systematics/full_n3ll_wy_production_2026/scripts/run_fixed_target_quadrature_probe.py |
| Candidate status and limitations | systematics/full_n3ll_wy_production_2026/HANDOFF.md |
