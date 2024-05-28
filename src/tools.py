from dataset import read_problems
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append("/home/S/hexiaolong/codex/self-debug/humaneval")
from executor_utils import check_correctness,check_test_correctness
import json
import numpy as np
from myutils import *
from resfiles_record import res_root,data_files,res_7b16k,res_cola7bpy,res_cola34bpy,res_llama7b,tmp
from evaluate import *
def load_gened_testcase(results):
    tescase_gened = {}
    testcase_num = []
    for result in results:
        tid = result["task_id"]
        is_MBPP = False
        if "MBPP" in tid:
            is_MBPP = True
        gened_testcase = result["gened_testcase"]
        testcase_num.append(len(gened_testcase))
        tescase_gened[tid] = gened_testcase
    print(f"testcase num : mean {sum(testcase_num)/len(testcase_num)}, min {min(testcase_num)}, max {max(testcase_num)}")
    return tescase_gened

def gened_testcase_compare(testcase1,testcase2):
    k1,k2 = testcase1.keys(),testcase2.keys()
    print(f"length t1:{len(k1)},t2:{len(t2)}")
    k = k1 if len(k1) < len(k2) else k2
    for key in k:
        if key=="HumanEval/89":
            continue
        gened_testcase1 = set(testcase1[key])
        gened_testcase2 = set(testcase2[key])
        equal_len = len(gened_testcase1)==len(gened_testcase2)
        if gened_testcase1==gened_testcase2:
            print(f"Two testcases are equal t1:{len(gened_testcase1)}, t2:{len(gened_testcase2)}")
            pass
        else:
            print(f"Two testcases are not equal!")
            print(f"t1 - t2 \[len:{len(gened_testcase1-gened_testcase2)}\]:\n{gened_testcase1-gened_testcase2}")
            print(f"t2 - t1 \[len:{len(gened_testcase2-gened_testcase1)}\]:\n{gened_testcase2-gened_testcase1}")
    return




if __name__=="__main__":
    f1 = res_root + tmp[-3]
    f2 = res_root + res_cola7bpy[3]
    
    r1 = load_results(f1)
    r2 = load_results(f2)
    t1 = load_gened_testcase(r1)
    t2 = load_gened_testcase(r2)
    
    gened_testcase_compare(t1,t2)
    
    # with open("/home/S/hexiaolong/codex/self-debug/src/gened_testcase_codellama7bpy_humaneval.jsonl","w") as f:
    #     for tid,testcase in t2.items():
    #         f.write(json.dumps({tid:testcase})+"\n")
    # with open("/home/S/hexiaolong/codex/self-debug/src/gened_testcase_codellama7bpy_humaneval.jsonl","a+") as f:
    #     for tid,testcase in t1.items():
    #         if tid=="HumanEval/89":
    #             f.write(json.dumps({tid:testcase})+"\n")
    
    