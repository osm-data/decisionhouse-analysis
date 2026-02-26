#!/bin/bash
# Run caching benchmark for all scales: M, XXL, 3XL

set -e

echo "======================================================================"
echo "RUNNING CACHING BENCHMARK FOR ALL SCALES"
echo "======================================================================"
echo ""

SCALES=("M" "XXL" "3XL")
SEED=42

for SCALE in "${SCALES[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Running scale: $SCALE"
    echo "======================================================================"
    echo ""

    uv run python benchmark_caching.py --scale "$SCALE" --seed "$SEED"

    echo ""
    echo "✓ Completed scale $SCALE"
done

echo ""
echo "======================================================================"
echo "ALL SCALES COMPLETED"
echo "======================================================================"
echo ""
echo "Results saved:"
for SCALE in "${SCALES[@]}"; do
    echo "  - results/results_cache_experiment_${SCALE}_seed${SEED}.json"
    echo "  - results/caching_benefit_${SCALE}.pdf"
    echo "  - results/caching_benefit_${SCALE}.eps"
done
echo ""
