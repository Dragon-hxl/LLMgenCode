# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# 来自codellama源码，使用UTfeedback进行了修改
import fire
import json
import gzip
import yaml
from typing import Dict,Iterable,Optional
import sys
sys.path.append("/home/S/hexiaolong/codex/human-eval")
sys.path.append("/home/S/hexiaolong/codex/codellama")
from llama import Llama
from human_eval.execution import run_code_with_output2, check_correctness
from human_eval.data import read_problems
from concurrent.futures import ThreadPoolExecutor
from myutils import code_clean2,get_unit_test,prompt_for_64
import time
import os
from collections import defaultdict
os.environ["TOKENIZERS_PARALLELISM"] = "true"

prompt_root = "/home/S/hexiaolong/codex/self-debug/prompt/"
data_root = "/home/S/hexiaolong/codex/self-debug/data/"
prompt_file = prompt_root + "prompt_base2.txt"
UTfeedback_file = "../prompt/prompt_UTfeedback.txt"
ut_file = "../data/test_from_prompt.jsonl"
true_tests_file = "../data/test_from_check.jsonl"
codeT_test_file = "../data/test_from_codeT_cola13bpy_t200_s100.jsonl"
humaneval_dataset = "/home/S/hexiaolong/codex/human-eval/data/HumanEval.jsonl.gz"

def get_truePass(problem,solution):
    with ThreadPoolExecutor(max_workers=1) as executor:
        args = (problem, solution, 3.0)
        future = executor.submit(check_correctness, *args)
        result = future.result()
        passed = result["passed"]
    return passed
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0,
    top_p: float = 1.0,
    max_seq_len: int = 4096,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
    output_file: str = None,
):
    with open("../configs/UTfeedback_config.yaml") as f:
        conf = yaml.safe_load(f)
        cfg = conf["codeT"]
        print(yaml.dump(cfg))
    debug_t = cfg["debug"]["temperature"]
    debug_p = cfg["debug"]["top_p"]
    debug_maxgen = cfg["debug"]["max_gen"]
    
    print(f"ckpt_dir: {ckpt_dir}\ntokenizer_path: {tokenizer_path}\nmax_seq_len: {max_seq_len}\nmax_batch_size: {max_batch_size}\noutput_file: {output_file}")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    unit_tests = {}
    assertions = defaultdict(list)
    assertion_strings = {}
    
    with open(prompt_file,"r") as f:
        preflex = f.read()

    with open(UTfeedback_file,"r") as af:
        UTfeedback_promt = af.read()
    def get_one_complication(problem,unit_test):#为模型的输入添加前缀串，看能否提高准确度。目前直接返回原串。
        res = preflex + unit_test + "\n\n# Complete the Python funtion:\n" + problem + "### result ###\n"
        return res

    base_unit_tests,base_assertions,base_assertion_strings = get_unit_test(ut_file)
    unit_tests,assertions,assertion_strings = get_unit_test(codeT_test_file,10,True)
    
    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    f = open(output_file,"w+",encoding='utf-8')
    full_output = output_file.split(".")[0] + "_full.jsonl" #记录每次轮回的结果
    fullf = open(full_output,"w+",encoding="utf-8")
    for tid in taskids:
        print(f"Gen code for tesk {tid}")
        start_time = time.time() #计时的
        tprompt = problems[tid]["prompt"]
        if tid == "HumanEval/64":
            tprompt = prompt_for_64
        problem = get_one_complication(tprompt,base_assertion_strings[tid])
        prompts = [problem]
        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        for prompt, result in zip(prompts, results):
            # print("===============origin code===================")
            # print(f"{result['generation']}")
            solution = result['generation']
            # 截取程序
            idx2 = solution.find("### Task End ###")
            if idx2 != -1:
                solution = solution[:idx2-1] #这里的减1是为了去掉前面的换行
            if len(solution.split("```"))>1:
                solution = solution.split("```")[1]
            else:
                print(solution.split("```"))
            if solution.startswith("python"):
                solution = solution[6:]
            solution = solution.strip("\n")
            # 去除函数头和注释
            entry_point = "def " + problems[tid]["entry_point"]
            solution = code_clean2(code=solution,entry_point=entry_point)
            print("\n======filter code======\n")
            print(solution)
            # print("-"*20)
        #self-debug step
        start_code = ""
        for line in tprompt.split("\n"):
            if line=="":
                continue
            if entry_point in line:
                start_code += line + "\n"
                break
            start_code += line + "\n"
        # print(f"entry_point: {entry_point}")
        # print("=====start code===========")
        # print(start_code)
        checkp = assertion_strings[tid]
        # for io in unit_tests[tid]:
        #     checkp += "assert " + io["tin"] + " == " + io["tout"] + "\n" # checkp把所有unit tests集合到一起用来判断程序对错
        ut = unit_tests[tid]
        run_test = [t["tin"] for t in ut] # 这个是用来执行的test，会返回代码执行它的结果和test_res比较得到UTfeedback
        # test_res = [literal_eval(t["tout"]) for t in ut]
        test_res = [t["tout"] for t in ut]
        #
        feedback_prompt = UTfeedback_promt + assertion_strings[tid] + "\n\n# Complete the Python funtion:\n" + tprompt+"### result ###\n```python\n" + start_code + "\n"
        
        cir = 1
        unchanged = 0 
        output_full = []
        # True_pass = get_truePass(problems[tid], solution)
        # 这里通过一次函数调用同时获得simple和UTfeedback
        with ThreadPoolExecutor(max_workers=1) as executor:
            args = (problems[tid], (start_code+solution), run_test, checkp, 3.0)
            future = executor.submit(run_code_with_output2, *args)
            result = future.result()
            passed = result["passed"]
            code_res = result["result"]
        print(f"task:{tid},cir:{cir},passed:{passed}")
        output_full.append({"cir":0, "solution": solution,"passed":passed})
        while cir < 11:
            completion = solution
            if passed:
                cir = 1024
            else:
                if type(code_res) is str:
                    prompt =  feedback_prompt +solution + "\n```\nFeedback: With the above function, " + unit_tests[tid][0]["tin"] +" returns the following error:\n\"\"\"\n"+code_res+ "\n\"\"\"\nSo the code does not pass the assertion. Please fix it.\n\n### fix result ###\n"
                else:
                    utFeedback = "\n```\nFeedback: With the above function,"
                    for i,cres in enumerate(code_res):
                        cres = str(cres)
                        if len(cres) > 1024:
                            print(f"too long code result for testcase with {cres}")
                            break
                        # print(f"code return res is {cres}. real res is {test_res[i]}")
                        if cres == test_res[i]:
                            utFeedback += f" {run_test[i]} == {cres} while the assertion is \"{assertions[tid][i]}\".The code pass this aasertion."
                            # pass
                        else:
                            utFeedback += f" {run_test[i]} == {cres} while the assertion is \"{assertions[tid][i]}\".The code does not pass this aasertion."
                    utFeedback += "\nSo the code is wrong. Please fix it.\n\n### fix result ###\n"
                    prompt = feedback_prompt +solution + utFeedback
                # print("---------------------feeedback prompt---------------------------")
                # print(utFeedback)
                # print("----------------------------------------------------------------")
                prompts = [prompt]
                results = generator.text_completion(
                    prompts,
                    max_gen_len=debug_maxgen,
                    temperature=debug_t,
                    top_p=debug_p,
                )
                for prompt, result in zip(prompts, results):
                    ans = result['generation'].strip()
                    if "```" in ans:
                        tmp = ans.split("```")
                        if len(tmp) > 1:
                            sol = tmp[1].strip()
                        else:
                            sol = ans
                    else:
                        sol = ans
                    if sol.startswith("python"):
                        sol = sol[6:]
                    sol = sol.strip("\n")
                    # 去除函数头和注释
                    solution = code_clean2(code=sol,entry_point=entry_point,start_code=start_code)
                    # print("==================fix ans======================")
                    # print(ans)
                    print("----------------filter fix ans---------------")
                    print(solution)
                    print("==================fix end======================")
                if completion==solution:
                    unchanged += 1
                    print(f"unchanged solution in cir[{cir}] with task {tid}")
                # True_pass = get_truePass(problems[tid], solution)
                # 这里通过一次函数调用同时获得simple和UTfeedback
                with ThreadPoolExecutor(max_workers=1) as executor:
                    args = (problems[tid], (start_code+solution), run_test, checkp, 3.0)
                    future = executor.submit(run_code_with_output2, *args)
                    result = future.result()
                    passed = result["passed"]
                    code_res = result["result"]
                print(f"task:{tid},cir:{cir},passed:{passed}")
                output_full.append({"cir":cir, "solution": solution, "passed": passed})
            cir += 1
        end_time = time.time()
        used_time = (end_time - start_time)/60
        print(f"task {tid} spends {used_time} min")
        fullf.write(json.dumps({"task_id": tid,"completion":output_full})+"\n")
        output = {"task_id": tid,"completion":solution}
        f.write(json.dumps(output)+"\n")
    f.close()
    fullf.close()

if __name__ == "__main__":
    fire.Fire(main)
