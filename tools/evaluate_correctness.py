import json
import sys
sys.path.append("/home/S/hexiaolong/codex/human-eval")
from human_eval.data import read_problems
from human_eval.execution import run_code_with_output2, check_correctness
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import matplotlib.pyplot as plt

def get_truePass(problem,solution):
    with ThreadPoolExecutor(max_workers=1) as executor:
        args = (problem, solution, 3.0)
        future = executor.submit(check_correctness, *args)
        result = future.result()
        passed = result["passed"]
    return passed

def draw_plots(data,image_path):
    # data format: {"label:value"} value is a dict: {cir:[task_id1,task_id2...]}
    xs = []
    ys = []
    labels = []
    for k,v in data.items():
        labels.append(k)
        x = range(11)
        xs.append(x)
        y = [len(v[i]) for i in x]
        ys.append(y)
    fig = plt.figure(figsize=(5,3),dpi=400)
    plt.xlabel("Cirs")
    plt.ylabel("number of task")
    plt.title(image_path.split(".")[0])
    for i in range(len(data.keys())):
        x = xs[i]
        y = ys[i]
        plt.plot(x,y)
        for xz,yz in zip(x,y):
            plt.text(xz,yz,yz)
    plt.legend(labels,loc="upper left")
    fig.savefig(image_path)
    return

def get_pass_n(result_file,output_file=None):
    passed_per_cir = defaultdict(set)
    task_cir = defaultdict(list)
    task_cir_pass = defaultdict(list)
    problems = read_problems()
    passed_num = 0
    data_back = []
    with open(result_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            task_id = data["task_id"]
            completion = data["completion"]
            problem = problems[task_id]
            com_back = {}
            for cir,solutions in completion.items():
                cir = int(cir)
                print(f"Task {task_id} gens {len(solutions)} solutions in cir {cir}")
                total_passed = False
                solutions_back = []
                for i,solution in enumerate(solutions):
                    solution = solution["solution"]
                    passed = get_truePass(problem,solution)
                    print(f"solution {i} passed {passed}")
                    if passed:
                        total_passed =True
                    solutions_back.append((solution,passed))
                task_cir_pass[task_id].append(total_passed)
                com_back[cir] = solutions_back
                # if total_passed:
                #     passed_per_cir[cir].add(task_id)
                task_cir[task_id].append(cir)
            data_back.append({"task_id":task_id, "complletion":com_back})
    # for i in range(1,11):
    #     passed_per_cir[i] = passed_per_cir[i].union(passed_per_cir[i-1])
    for t,v in task_cir_pass.items():
        n = len(v)
        for i in range(n,11):
            task_cir_pass[t].append(v[n-1])
    for t,v in task_cir_pass.items():
        for i,passed in enumerate(v):
            if passed:
                passed_per_cir[i].add(t)
    for k,v in passed_per_cir.items():
        tid_int = [int(x.split("/")[1]) for x in v]
        print(f"cir {k},passed {len(v)} tasks, pass rate is {len(v)/164}")
        print(f"pass tasks are:\n{sorted(list(tid_int))}")
    for k,v in task_cir_pass.items():
        print(f"task {k} pass or not for each cir: {v}")
    print("--------------------------------------------")
    all_True = 0
    all_False = 0
    Both_True_False = 0
    for k,v in task_cir_pass.items():
        if True in v and False in v:
            Both_True_False += 1
            print(f"task {k} pass or not for each cir: {v}")
        elif True in v and False not in v:
            all_True += 1
        else:
            all_False += 1
    print(f"all true: {all_True}, all false: {all_False}, both: {Both_True_False}")
    print("--------------------------------------------")
    
    if output_file:
        with open(output_file,"w+") as f:
            for data in data_back:
                f.write(json.dumps(data)+"\n")
    
    return passed_per_cir,task_cir

def get_pass_1(result_file):
    passed_per_cir = defaultdict(set)
    task_cir = defaultdict(list)
    task_cir_pass = defaultdict(list)
    problems = read_problems()
    with open(result_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            task_id = data["task_id"]
            completion = data["completion"]
            problem = problems[task_id]
            for cir,solutions in completion.items():
                cir = int(cir)
                print(f"Task {task_id} gens {len(solutions)} solutions in cir {cir}")
                total_passed = False
                solution = solutions[0]["solution"]
                passed = get_truePass(problem,solution)
                if passed:
                    total_passed =True
                task_cir_pass[task_id].append(total_passed)
                # if total_passed:
                #     passed_per_cir[cir].add(task_id)
                task_cir[task_id].append(cir)
    # for i in range(1,11):
    #     passed_per_cir[i] = passed_per_cir[i].union(passed_per_cir[i-1])
    for t,v in task_cir_pass.items():
        n = len(v)
        for i in range(n,11):
            task_cir_pass[t].append(v[n-1])
    for t,v in task_cir_pass.items():
        for i,passed in enumerate(v):
            if passed:
                passed_per_cir[i].add(t)
    for k,v in passed_per_cir.items():
        tid_int = [int(x.split("/")[1]) for x in v]
        print(f"cir {k},passed {len(v)} tasks, pass rate is {len(v)/164}")
        print(f"pass tasks are:\n{sorted(list(tid_int))}")
    # for k,v in task_cir_pass.items():
    #     print(f"task {k} pass or not for each cir: {v}")
    print("--------------------------------------------")
    all_True = 0
    all_False = 0
    Both_True_False = 0
    for k,v in task_cir_pass.items():
        if True in v and False in v:
            Both_True_False += 1
            print(f"task {k} pass or not for each cir: {v}")
        elif True in v and False not in v:
            all_True += 1
        else:
            all_False += 1
    print(f"all true: {all_True}, all false: {all_False}, both: {Both_True_False}")
    print("--------------------------------------------")
    
    return passed_per_cir,task_cir



def show_info(file):
    with open(file,"r")as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["task_id"]
            completion = data["completion"]
    return

def draw_plots2(data,image_path):
    # data format: {"label:value"} value is a dict: {cir:[task_id1,task_id2...]}
    xs = []
    ys = []
    labels = []
    for k,v in data.items():
        labels.append(k)
        x = range(len(v))
        xs.append(x)
        y = v
        ys.append(y)
    fig = plt.figure(figsize=(5,3),dpi=400)
    plt.xlabel("Cirs")
    plt.ylabel("number of task")
    plt.title(image_path.split(".")[0])
    for i in range(len(data.keys())):
        x = xs[i]
        y = ys[i]
        plt.plot(x,y)
        for xz,yz in zip(x,y):
            plt.text(xz,yz,yz)
    plt.legend(labels,loc="upper left")
    fig.savefig(image_path)
    return



if __name__=="__main__":
    # file = "../res/UTfeedback_multi_7b16k_full.jsonl"
    file = "../res/UTfeedback_multiCODET2_7b16k_pT.jsonl"
    # output_file = "../res/multi_with_pass/UTfeedback_multiSBSP10_7b16k_tT.jsonl"
    passed_per_cir,task_cir = get_pass_1(file)
    # passed_per_cir,task_cir = get_pass_n(file)
    # data = {"SBSP10_7b16k_tT":passed_per_cir}
    # draw_plots(data=data, image_path="../image/UTfeedback_multiSBSP10_7b16k_tT.jpg")
                
                    