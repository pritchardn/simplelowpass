#!/bin/bash

mkdir -p ../data/
mkdir -p ../results/double/config/noisy
mkdir -p ../results/double/config/clean
mkdir -p ../results/double/raw/noisy
mkdir -p ../results/double/raw/clean
mkdir -p ../results/single/config/noisy
mkdir -p ../results/single/config/clean
mkdir -p ../results/single/raw/noisy
mkdir -p ../results/single/raw/clean
echo "Generating Data"
python3 input_generator.py ../data/ 10
echo "Processing from Config"
# shellcheck disable=SC2125
CONF_FILES="../data/*.in"
RAW_FILES="../data/*.npz"
for f in $CONF_FILES
do
  echo "Processing $f"
  python3 lowpass.py 0 "$f" "../results/single/config/" 0
  python3 lowpass.py 0 "$f" "../results/double/config/" 1
done

echo "Processing from direct files"
for f in $RAW_FILES
do
  echo "Processing $f"
  python3 lowpass.py 1 "$f" "../results/single/raw/" 0
  python3 lowpass.py 1 "$f" "../results/double/raw/" 1
done

echo "Repetition Analysis"
python3 ./postProcessing/repetition_test.py "../results/double/raw/noisy/" "./postProcessing/repeat" "../results/double/raw/clean/3_numpy_pointwise.out.npy"

echo "Reproduction Analysis 1"
python3 ./postProcessing/reproduce_test_1.py "../results/double/raw/clean/" "./postProcessing/reproduce1"

echo "Reproduction Analysis 2"
python3 ./postProcessing/reproduce_test_2.py "../results/single/raw/clean/" "../results/double/raw/clean/" "./postProcessing/reproduce2"