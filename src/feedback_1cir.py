# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import transformers, torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import read_problems
from human_eval.execution import run_code_with_output2
from concurrent.futures import ThreadPoolExecutor
from myutils import map_gpu_memory,get_args,code_clean,code_clean2
import os
from ast import literal_eval
from collections import defaultdict
os.environ["TOKENIZERS_PARALLELISM"] = "true"

prompt_file = "/home/S/hexiaolong/codex/human-eval/simfeedback_13bv1fd_cir1.jsonl"
ut_file = "/home/S/hexiaolong/codex/human-eval/tests_from_prompt.jsonl"


def main():
    # 获取参数
    args = get_args()
    output_file = args.output
    model_path = args.model_path
    
    feedback_dict={}

    with open(prompt_file,"r") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            tid = data["tid"]
            passed = data["passed"]
            feedback = data["feedback"]
            feedback_dict[tid] = data

    #为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
    max_memory_mapping = map_gpu_memory(used_gpu=[])

    #加载模型
    print("load model from ",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True)#, use_safetensors=True

    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    print(taskids)
    f = open(output_file,"w+",encoding='utf-8')
    for tid in taskids:
        print("get solution for task :",tid)
        data = feedback_dict[tid]
        passed = data["passed"]
        if passed:
            output = {"task_id": tid,"completion":data["feedback"]}
            f.write(json.dumps(output)+"\n")
            continue
        else:
            prompt = data["feedback"]
            input_len = len(prompt)
            inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
            inputs = inputs.to('cuda')
            pred = model.generate(**inputs, max_new_tokens=512, temperature=0)#,temperature=0.4,repetition_penalty=1.1
            ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:]
            solution = ans.strip("\n")
            print("====================prompt==================")
            print(prompt)
            print("+++++++++++origin solution++++++++++++++++++")
            print(solution)
            # 截取程序
            idx2 = solution.find("### Task End ###")
            if idx2 != -1:
                solution = solution[:idx2-1] #这里的减1是为了去掉前面的换行
            if len(solution.split("```"))>1:
                solution = solution.split("```")[1]
            else:
                print("code not wraped by  ```")
                # print(solution.split("```"))
            if solution.startswith("python"):
                solution = solution[6:]
            solution = solution.strip("\n")
            # 去除函数头和注释
            entry_point = "def " + problems[tid]["entry_point"]
            solution = code_clean2(code=solution,entry_point=entry_point)
            print("+++++++++++filter solution++++++++++++++++++")
            print(solution)
            print("++++++++++++++++++++++++++++++++++++++++++++")
            output = {"task_id": tid,"completion":solution}
            f.write(json.dumps(output)+"\n")
    f.close()


if __name__ == "__main__":
    main()
