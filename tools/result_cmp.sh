#!/bin/bash

model=$1

echo "base and simfeedback"
python3.9 result_cmp.py -f1 "base_${model}2_sorted.txt" -f2 "simfeedback_${model}2_sorted.txt"
echo "base and UTfeedback"
python3.9 result_cmp.py -f1 "base_${model}2_sorted.txt" -f2 "UTfeedback_${model}2_sorted.txt"
echo "simfeedback and UTfeedback"
python3.9 result_cmp.py -f1 "simfeedback_${model}2_sorted.txt" -f2 "UTfeedback_${model}2_sorted.txt"
