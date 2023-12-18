import json
import sys
sys.path.append("/home/S/hexiaolong/codex/human-eval")
from human_eval.data import read_problems
from human_eval.execution import run_code_with_output2, check_correctness
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
from data_tools import data_analysis,Counter_with_base


def load_testcase(test_file):
    with open(test_file) as f:
        data = json.loads(f.read())
    return data

def testfile_trans(test_file,resfile):
    tests = []
    data = load_testcase(test_file)
    for tid,v in data.items():
        ios = []
        for test in v:
            if "\n" in test:
                # print(test)
                test = test.replace("\n","")
                continue
            if "==" in test:
                tin = test.split("==")[0].replace("assert","").strip()
                tout = test.split("==")[1].strip()
                ios.append({"tin":tin,"tout":tout})
        print(f"task {tid} has {len(ios)} tests")
        tests.append({"task_id":tid,"ios":ios})
    with open(resfile,"w+")as f:
        for t in tests:
            f.write(json.dumps(t)+"\n")
    return
      
def testcase_num(test_file):
    test_num = []
    test_less_10 = []
    test_less_100 = []
    with open(test_file) as f:
        data = json.loads(f.read())
        for k,v in data.items():
            v = sum(v,[])
            test_num.append(len(v))
            if len(v) < 10:
                test_less_10.append(k)
            if len(v) < 100:
                test_less_100.append(k)
    data_analysis(test_num)
    print(Counter_with_base(test_num,100))
    print(Counter_with_base([x for x in test_num if x < 100],10))
    print(f"{len(test_less_10)} task has less than 10 testcases.They are:\n{test_less_10}")
    print(f"{len(test_less_100)} task has less than 100 testcases.They are:\n{test_less_100}")
    
def testcase_num2(test_file):
    test_num = []
    test_less_10 = []
    test_less_100 = []
    limit = 5
    with open(test_file) as f:
        for line in f.readlines():
            data = json.loads(line)
            for k,v in data.items():
                limit_v = [t[:limit] for t in v]
                v = sum(v,[])
                test_num.append(len(v))
                if len(v) < 10:
                    test_less_10.append(k)
                if len(v) < 100:
                    test_less_100.append(k)
    data_analysis(test_num)
    print(Counter_with_base(test_num,100))
    print(Counter_with_base([x for x in test_num if x < 100],10))
    print(f"{len(test_less_10)} task has less than 10 testcases.They are:\n{test_less_10}")
    print(f"{len(test_less_100)} task has less than 100 testcases.They are:\n{test_less_100}")

def testcases_merge(file1,file2,resfile):
    data1 = {}
    data2 = {}
    with open(file1,"r") as f:
        data1 = json.loads(f.read())
    with open(file2,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            for k,v in data.items():
                data2[k] = v
    for k,v in data1.items():
        if k in data2.keys():
            data1[k] = data1[k] + data2[k]
    with open(resfile,"w+") as f:
        f.write(json.dumps(data1))
    return data1
            
        

if __name__ == "__main__":
    CODET_tescase_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_filter_add.jsonl"
    # add_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test3_add.jsonl"
    # res_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_filter_add.jsonl"
    # testcases_merge(CODET_tescase_file,add_file,res_file)
    # testcase_num2(CODET_tescase_file)
    res_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_uniform.jsonl"
    testfile_trans(CODET_tescase_file,res_file)