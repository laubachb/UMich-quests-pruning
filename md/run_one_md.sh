#!/bin/bash
# One NVT run of a compiled NequIP model at one carbon state point, RDF averaged in LAMMPS.
#   md/run_one_md.sh MODEL_DIR STATEPOINT_DIR OUT_DIR
# MODEL_DIR      directory holding compiled_best.nequip.pth
# STATEPOINT_DIR folder name under md/inputs (data.in + temperature from md/statepoints.csv)
# Protocol as in the paper's inputs: velocity create at T (seed 1111), fix nvt (Tdamp 0.05 ps),
# 1 fs x 6000 steps, newton off.  RDF: 200 bins to 4.0 A, sampled every 10 steps over steps
# 1000-6000 (fix ave/time), written to rdf.dat; thermo every 100 steps.  A run that loses
# atoms or produces NaN is recorded as unstable (status file) rather than retried.
set -uo pipefail
MODEL=$(readlink -f "$1")/compiled_best.nequip.pth; SP=$2; OUT=$3
REPO=$(cd "$(dirname "$0")/.." && pwd)
LMP=${LMP:-/work2/09982/blaubach/stampede3/software/lammps-nequip/install/bin/lmp}
export LD_LIBRARY_PATH=$REPO/env/venv-paper/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
T=$(awk -F, -v d="$SP" '$1==d{print $3}' $REPO/md/statepoints.csv)
mkdir -p "$OUT" && cd "$OUT"
[ -f status ] && grep -q -E "^(ok|unstable)" status && { echo "skip $OUT"; exit 0; }
cp $REPO/md/inputs/$SP/data.in .
cat > in.md <<IN
units           metal
atom_style      atomic
boundary        p p p
atom_modify     sort 0 0.0
newton          off
neighbor        1.0 bin
neigh_modify    delay 0 every 1 check yes
read_data       data.in
velocity        all create $T 1111 mom yes rot yes dist gaussian
pair_style      nequip
pair_coeff      * * $MODEL C
timestep        0.001
fix             1 all nvt temp $T $T 0.05
compute         g all rdf 200 cutoff 4.0
fix             r all ave/time 10 500 6000 c_g[*] file rdf.dat mode vector start 1000
thermo_style    custom step time temp pe press
thermo_modify   lost error norm yes
thermo          100
run             6000
IN
SECONDS=0
"$LMP" -in in.md > lmp.out 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -s rdf.dat ] && ! grep -qiE "nan|ERROR" lmp.out; then echo "ok $SECONDS s" > status
else echo "unstable rc=$rc $SECONDS s: $(grep -m1 -iE 'ERROR|nan' lmp.out | cut -c1-150)" > status; fi
cat status
