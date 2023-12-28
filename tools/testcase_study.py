import json
import sys
sys.path.append("/home/S/hexiaolong/codex/human-eval")
sys.path.append("/home/S/hexiaolong/codex/self-debug")
from human_eval.data import read_problems
from human_eval.execution import run_code_with_output2, check_correctness
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
from data_tools import data_analysis,Counter_with_base
from myutils import get_unit_test
from collections import defaultdict
# testcase files
test_from_check = "../data/test_from_check.jsonl"
test_from_prompt = "../data/test_from_prompt.jsonl"
gened_testcase = "../try/gen_test_t0.8_topp0.95_sample100_max300.jsonl"

correct_testcase_gened = "../try/gen_test_t0.8_topp0.95_sample100_max300_filter.jsonl"
correct_testcase_gened2 = "../try/gen_test_t0.8_topp0.95_sample100_max300_correct2.jsonl"
# problems
problems = read_problems()
def load_testcase_uniform(test_file):
    testcase = {}
    with open(test_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["task_id"]
            ios = data["ios"]
            uts = []
            for io in ios:
                tin = io["tin"]
                tout = io["tout"]
                t = tin + " == " + tout
                uts.append(t)
            testcase[tid] = uts
    return testcase

def load_testcase_gened(test_file):
    testcase = {}
    with open(test_file,"r") as f:
        for line in f.readlines():
            testcases = json.loads(line)
            for tid,ios in testcases.items():
                uts = []
                for io in ios:
                    io = io.replace("assert","").strip()
                    uts.append(io)
                testcase[tid] = uts
    return testcase

def load_testcase_gened2(test_file):
    testcase = {}
    with open(test_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["task_id"]
            tests = data["testcases"]
            testcase[tid] = tests
    return testcase

def testcase_compare(problems,testcases):
    taskids = list(problems.keys())
    for tid in taskids:
        print(f"++++++++++++++++++{tid}++++++++++++++++++")
        print(problems[tid]["prompt"])
        print(problems[tid]["canonical_solution"])
        for label,testcase in testcases.items():
            print(f"----------------{label}---------------------- ")
            ios = testcase[tid]
            for io in ios:
                print(io)
    return
            



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
            
def show_gened_testcase(test_file,verbose):
    testcases_load = {}
    with open(test_file,"r")as f:
        for line in f.readlines():
            data = json.loads(line)
            for k,v in data.items():
                cases = sum(v,[])
                one_line_cases = [c.replace("\n","") for c in cases]
                testcases_load[k] = one_line_cases
                if verbose:
                    print(f"+++++++++++++++++++{k}+++++++++++++++++++++++++")
                    for c in one_line_cases:
                        print(c)
    return testcases_load

never_passed_task = [1, 6, 10, 17, 19, 32, 36, 38, 64, 70, 71, 74, 75, 81, 83, 93, 102, 103, 104, 106, 107, 113, 114, 118, 119, 123, 125, 129, 130, 131, 141, 144, 145, 146, 147, 148, 151, 154, 156, 160, 161, 163]
pt_not_passed_task = [3, 9, 16, 18, 20, 26, 54, 57, 88, 90, 91, 92, 98, 99, 100, 105, 108, 109, 110, 116, 128, 135, 136, 138, 139, 142, 149, 150, 153, 155, 157]
gened_test_not_passed_task = [3, 9, 16, 21, 62, 87, 89, 90, 91, 92, 97, 99, 100, 109, 110, 116, 120, 127, 128, 133, 138, 139, 142, 149, 153, 155, 157]
more_test_not_passed = [1, 3, 6, 9, 10, 16, 17, 19, 21, 32, 36, 38, 39, 46, 50, 62, 64, 65, 68, 69, 70, 71, 74, 75, 76, 77, 81, 83, 87, 89, 90, 91, 92, 93, 94, 97, 99, 100, 102, 103, 104, 106, 107, 109, 110, 113, 114, 115, 116, 118, 119, 120, 123, 125, 127, 128, 129, 130, 131, 132, 133, 134, 138, 139, 141, 142, 144, 145, 146, 147, 148, 149, 151, 153, 154, 155, 156, 157, 158, 160, 161, 163]

def correct_percent_analysis(test_file):
    
    base = 0.1
    percent_counter = defaultdict(list)
    number_counter = defaultdict(list)
    zeros = []
    with open(test_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["task_id"]
            testcase_num = len(data["testcases"])
            tid = int(tid.split("/")[1].strip())
            if tid not in more_test_not_passed:
                continue
            percent = data["correct_percent"]
            if percent==0.0:
                zeros.append((tid,percent))
            number_counter[testcase_num].append(tid)
            percent_counter[percent//base].append((tid,percent))
    print(f"no correct testcase:{zeros}")
    for k,v in percent_counter.items():
        print(f"percent between {k*base} and {k*base+base} has {len(v)} tasks.")
        print(v)
    print(number_counter)

if __name__ == "__main__":
    # CODET_tescase_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_filter_add.jsonl"
    # add_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test3_add.jsonl"
    # res_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_filter_add.jsonl"
    # testcases_merge(CODET_tescase_file,add_file,res_file)
    # testcase_num2(CODET_tescase_file)
    # res_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_uniform.jsonl"
    # testfile_trans(CODET_tescase_file,res_file)
    # t1 = load_testcase_uniform(test_from_check)
    # t2 = load_testcase_uniform(test_from_prompt)
    # # t3 = load_testcase_gened(correct_testcase_gened)
    # t3 = load_testcase_gened2(correct_testcase_gened2)
    # ts = {"check":t1,"prompt test":t2,"gened":t3}
    # testcase_compare(problems,ts)
    # show_gened_testcase(gened_testcase)
    print(len(never_passed_task))
    print(len(pt_not_passed_task))
    print(len(never_passed_task))
    correct_percent_analysis(correct_testcase_gened2)