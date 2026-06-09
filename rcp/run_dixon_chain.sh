#!/usr/bin/env bash
# Container-side chain: rerun the CFR-evaluation post-processing after the
# Dixon FPED/FNED fix (2026-05-18).
#
# This is the entry point for the CPU-only `rcp/submit.sh dixon-chain`
# Run:ai job. Six CPU-bound pandas scripts in order. The chain is
# self-sufficient on a fresh PVC: it computes 03 (cfr) and 05 (group ECE)
# from the cache parquets first, so 06's three-input requirement is met
# even when /scratch has been wiped.
#
#   1. 03_compute_cfr.py          per-checkpoint CFR (hard, soft L1, soft JS)
#                                 -> per_checkpoint_cfr.parquet
#   2. 04_compute_fped_fned.py    Dixon sum-of-deviations rebuild
#                                 -> per_checkpoint_fped_fned_dixon.parquet
#   3. 05_compute_group_ece.py    per-group ECE
#                                 -> per_checkpoint_group_ece.parquet
#   4. 06_run_significance_suite  paired bootstrap + Bonferroni/Holm
#                                 -> significance_tests.csv, significance_summary.md
#   5. 07_build_main_table.py     paper-ready main results markdown
#                                 -> main_table.md
#   6. 09_write_report.py         REPORT.md rollup
#                                 -> analysis/REPORT.md
#
# Wall clock: ~5-10 minutes total on any CPU node.
#
# Usage on jumphost (CPU-only, no GPU slot wasted):
#   GPU_COUNT=0 bash rcp/submit.sh dixon-chain "bash rcp/run_dixon_chain.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[dixon-chain] CWD     : $(pwd)"
echo "[dixon-chain] python  : $(which python3) ($(python3 --version))"
echo ""

echo "===== Step 1/6: 03_compute_cfr.py ====="
python3 analysis/03_compute_cfr.py
echo ""

echo "===== Step 2/6: 04_compute_fped_fned.py (Dixon rebuild) ====="
python3 analysis/04_compute_fped_fned.py
echo ""

echo "===== Step 3/6: 05_compute_group_ece.py ====="
python3 analysis/05_compute_group_ece.py
echo ""

echo "===== Step 4/6: 06_run_significance_suite.py ====="
python3 analysis/06_run_significance_suite.py
echo ""

echo "===== Step 5/6: 07_build_main_table.py ====="
python3 analysis/07_build_main_table.py
echo ""

echo "===== Step 6/6: 09_write_report.py ====="
python3 analysis/09_write_report.py
echo ""

echo "[dixon-chain] all six scripts succeeded."
echo "[dixon-chain] Updated artifacts:"
echo "  analysis/results/per_checkpoint_cfr.parquet"
echo "  analysis/results/per_checkpoint_fped_fned_dixon.parquet"
echo "  analysis/results/fped_fned_summary.csv"
echo "  analysis/results/main_table_fped.md"
echo "  analysis/results/per_checkpoint_group_ece.parquet"
echo "  analysis/results/significance_tests.csv"
echo "  analysis/results/significance_summary.md"
echo "  analysis/results/main_table.md"
echo "  analysis/REPORT.md"
