import transformers, torch
import json
import argparse
import ast
import sys
import time
sys.path.append("/home/S/hexiaolong/codex/self-debug/humaneval")
sys.path.append("/home/S/hexiaolong/codex/self-debug")
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import read_problems
from human_eval.execution import check_test_correctness
from concurrent.futures import ThreadPoolExecutor
import os
from myutils import map_gpu_memory,prompt_for_64,make_printv,print_with_tag
os.environ["TOKENIZERS_PARALLELISM"] = "true"

shot_file = "gen_test_shot_20240313.txt"
# shot_file = "gen_test_shot_codellama.txt"
def get_one_shot():
    with open(shot_file,"r") as f:
        shot = f.read()
    return shot

def check_test_valid(problem, test_in, test_out):
    checkp = problem["prompt"] + f"    pass\nassert {test_in} == {test_out}"
    # print(checkp)
    with ThreadPoolExecutor(max_workers=1) as executor:
        args = (checkp,3.0)
        future = executor.submit(check_test_correctness, *args)
        result = future.result()
        passed = result["passed"]
        code_res = result["result"]
    print(f"passed:{passed},result:{code_res}")
    if passed or "AssertionError" in code_res:
        return True
    else:
        return False
    
def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="human_eval test")
    parser.add_argument('-mf','--model_path',default='/lustre/S/hexiaolong/vicuna-7b-v1.1',required=True, help="file path where the model used locates")
    parser.add_argument('-o','--output',default="ouput.json",required=True, help="file where the output store")

    args = parser.parse_args()
    output_file = args.output
    model_path = args.model_path
    unit_tests = {}

    printv = make_printv(True)
    
    #为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
    max_memory_mapping = map_gpu_memory(used_gpu=[])

    #加载模型
    printv("load model from ",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)#,add_eos_token=True,add_bos_token=True,padding_side='left', trust_remote_code=True,legacy=False
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)#, use_safetensors=True

    #获取solution
    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    printv("task num: ",num_task )
    f = open(output_file,"w+",encoding='utf-8')
    shot = get_one_shot()
    for tid in taskids:
        num_id = int(tid.split("/")[1])
        if num_id < 0 or num_id > 164:
            continue
        cir =0
        tests = []
        test_in_set = set()
        test_get = set()
        already_gen =""
        temperature = 0.8
        top_k = 50
        end_cir = 20
        tpromt = problems[tid]["prompt"]
        if tid == "HumanEval/64":
            tpromt = prompt_for_64
        while cir < end_cir:
            printv(f"start cir : {cir}")
            prompt = f"{shot}\n\nfunc signature:\n{tpromt}\nunit tests:\n"
            # prompt = f"{shot}\nfunc signature:\n{tpromt}[/INST]"
            # problem = prompt + "\tpass\n\n" + "# Check the correctness of " + problems[id]["entry_point"] +"\n" + already_gen +"\nassert"
            print_with_tag(prompt,"prompt",True)
            input_len = len(prompt)
            inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
            inputs = inputs.to('cuda')
            st = time.time()
            # pred = model.generate(**inputs, max_new_tokens=2048,top_p=0.95,temperature=temperature,repetition_penalty=1.1)#,temperature=0.4
            pred = model.generate(**inputs, max_new_tokens=512,top_k=top_k,do_sample=True,temperature=temperature,num_return_sequences=8,pad_token_id=tokenizer.eos_token_id,repetition_penalty=1.1)#,repetition_penalty=1.1
            interval = time.time() - st
            printv(f"gen time : {interval}")
            ans = ""
            for p in pred:
                ans += tokenizer.decode(p, skip_special_tokens=True) + "\nnext ans :\n"#[input_len-7:].strip()
            print_with_tag(ans,"ans",True)
            entry_point = "assert " + problems[tid]["entry_point"] + "("
            for line in ans.split("\n"):
                line = line.replace("\\_","_")
                if entry_point in line and "==" in line and "# assert" not in line:
                    test_in = line.split("==")[0]
                    test_in = line.split("==")[0][test_in.index("assert ")+7:].strip()
                    test_out = line.split("==")[1].strip()
                    # if test_in in test_in_set:
                    #     continue
                    if (test_in,test_out) in test_get:
                        continue
                    testcase = f"assert {test_in} == {test_out}"
                    flag = py_is_syntax_valid(testcase)
                    if not flag:
                        printv(f"gen wrong testcase :  {test_in} == {test_out}")
                        continue
                    else:
                        printv(f"gen testcase :  {test_in} == {test_out}")
                    tests.append({"tin":test_in,"tout":test_out})
                    # test_in_set.add(test_in)
                    test_get.add((test_in,test_out))
                    printv("++++++++++++++++++++++++++++++++++++++++")
            if len(tests) > 100:
                cir = 1024
                f.write(json.dumps({tid:tests})+"\n")
                f.flush()
                printv(f"for task {tid} gen tests num: {len(tests)}\n")
            else:
                cir += 1
                # temperature -=0.05
                top_k += 5
                if cir == end_cir:
                    f.write(json.dumps({tid:tests})+"\n")
                    f.flush()
                    printv(f"for task {tid} gen tests failed with num: {len(tests)}\n")
    f.close()

