#!/bin/bash
# Download all required datasets.
# Run from repository root: bash data/download.sh

set -e
mkdir -p data/raw data/external

echo "=== Downloading GoldStandard2024 ==="
# Download from Zenodo
wget -O data/raw/GoldStandard2024.csv \
  "https://zenodo.org/records/14448399/files/GoldStandard2024.csv?download=1"
echo "GoldStandard2024 downloaded: $(wc -l < data/raw/GoldStandard2024.csv) lines"

echo ""
echo "=== External datasets ==="
echo "HateXplain: Clone from https://github.com/hate-alert/HateXplain"
echo "  Then extract Jewish-target subset and save as data/external/hatexplain_jewish.csv"
echo ""
echo "ToxiGen: Clone from https://github.com/microsoft/ToxiGen"
echo "  Then extract Jewish group and save as data/external/toxigen_jewish.csv"
echo ""
echo "ISCA Bias: Download from https://zenodo.org/records/15025646"
echo "  Save as data/external/isca_bias.csv"
echo ""
echo "=== Download complete ==="
