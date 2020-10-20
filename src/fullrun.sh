#!/bin/bash

mkdir -p ../data/
python3 input_generator.py ../data/
# shellcheck disable=SC2125
FILES=../data/*
for f in $FILES
do
  echo "Processing $f"
  python3 lowpass.py ../data/"$f"
done