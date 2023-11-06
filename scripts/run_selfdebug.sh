#! /bin/bash

models=("/lustre/S/hexiaolong/vicuna-7b-v0/" "/lustre/S/hexiaolong/vicuna-7b-v1.1/" "/lustre/S/hexiaolong/vicuna-13b-v0/" "/lustre/S/hexiaolong/vicuna-13b-v1.1/" "/lustre/S/hexiaolong/llama7b/" "/lustre/S/hexiaolong/llama13b/" "/lustre/S/hexiaolong/llama30b/" "/lustre/S/hexiaolong/llama65b/" )
model_kinds=("7bv0" "7bv1" "13bv0" "13bv1" "lla7b" "lla13b" "lla30b" "lla65b" )
kinds=("base" "simfeedback" "UTfeedback" "explfeedback")

# 获取传递的参数值
param1="$1"
param2="$2"

# 检查参数值是否在0到9之间
if [ "$param1" -ge 0 ] && [ "$param1" -le 8 ]; then
  # 获取选定模型的路径
  model_path="${models[$param1]}"
  model_kind="${model_kinds[$param1]}"
else
  echo "错误: 参数1必须在0到9之间"
  exit 1
fi

if [ "$param2" -ge 0 ] && [ "$param2" -le 3 ]; then
    kind="${kinds[$param2]}"
else
    echo "wrong: param2 must between 0 and 3"
    exit 1
fi

script="humaneval_$kind.py"

echo "run script $script -mf $model_path -o ${kind}_${model_kind}.jsonl"

# python3.9 $script -mf $model_path -o "$kind_$model_kind.jsonl"
