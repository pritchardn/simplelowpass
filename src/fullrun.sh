#!/bin/bash

mkdir -p ../data/
mkdir -p ../results/config/noisy
mkdir -p ../results/raw/noisy
echo "Generating Data"
python3 input_generator.py ../data/ 10
echo "Processing from Config"
# shellcheck disable=SC2125
CONF_FILES="../data/*.in"
RAW_FILES="../data/*.npz"
for f in $CONF_FILES
do
  echo "Processing $f"
  python3 lowpass.py 0 "$f" "../results/config/"
done

echo "Processing from direct files"
for f in $RAW_FILES
do
  echo "Processing $f"
  python3 lowpass.py 1 "$f" "../results/raw/"
done

echo "Generating Repeat Analysis"
python3 ../postProcessing/repetition_test.py "../results/raw/noisy/" "./repeat" "../results/raw/0_numpy_pointwise.out.npy"