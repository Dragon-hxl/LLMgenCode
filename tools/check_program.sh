#!/bin/bash

model=$2
kind=$1
#kind取值为base,sim,UT,expl的结果

python3.9 evaluate_functional_correctness.py "${kind}_${model}.jsonl" > "${kind}_${model}.txt" 2>&1
python3.9 sort_reslut.py -f "${kind}_${model}.txt"