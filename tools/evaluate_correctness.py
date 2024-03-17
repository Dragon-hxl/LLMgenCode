import json
import sys
# sys.path.append("/home/S/hexiaolong/codex/human-eval")
sys.path.append("/home/S/hexiaolong/codex/self-debug")
sys.path.append("/home/S/hexiaolong/codex/self-debug/humaneval")
from human_eval.data import read_problems
from human_eval.execution import run_code_with_output2, check_correctness
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
from data_tools import data_analysis,Counter_with_base
import fire
import numpy as np


def get_truePass(problem,solution):
    with ThreadPoolExecutor(max_workers=1) as executor:
        args = (problem, solution, 1.0)
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

def get_pass_all(resfile):
    passed_task = set()
    problems = read_problems()
    with open(resfile,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["task_id"]
            completions = data["completion"]
            problem = problems[tid]
            last_cir = max(list(completions.keys()))
            solutions = completions[last_cir]
            total_passed = False
            for solution in solutions:
                solution = solution["solution"]
                passed = get_truePass(problem,solution)
                if passed:
                    total_passed =True
                    passed_task.add(tid)
                    break
    # print(f"get_pass_all pass {len(passed_task)} tasks.They are:\n{list(passed_task)}")
    return passed_task

def get_pass_all_multi_files(files):
    passed_tasks = []
    for file in files:
        passed_task = get_pass_all(file)
        passed_tasks += list(passed_task)
    pass_ids = [int(t.split("/")[1]) for t in passed_tasks]
    pass_ids = sorted(pass_ids)
    print(f"get_pass_all pass {len(pass_ids)} tasks.They are:\n{pass_ids}")
            

def tid_to_int(tid):
    return int(tid.split("/")[1])

def get_pass_n(result_file,output_file=None):
    passed_per_cir = defaultdict(set)
    task_cir = defaultdict(list)
    task_cir_pass = defaultdict(list)
    problems = read_problems()
    data_back = []
    checked_task = set()
    with open(result_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            task_id = tid_to_int(data["task_id"])#data["task_id"]
            if task_id in checked_task:
                continue
            checked_task.add(task_id)
            completion = data["completion"]
            problem = problems[data["task_id"]]
            com_back = {}
            cirs = []
            for cir,solutions in completion.items():
                cir = int(cir)
                cirs.append(cir)
                print(f"Task {task_id} gens {len(solutions)} solutions in cir {cir}")
                assert cir <= 10
                solutions_back = []
                total_passed = False
                for i,solution in enumerate(solutions):
                    solution = solution["solution"]
                    passed = get_truePass(problem,solution)
                    print(f"solution {i} passed {passed}")
                    if passed:
                        total_passed =True
                        for c in range(cir,11):
                            passed_per_cir[c].add(task_id)
                    solutions_back.append((solution,passed))
                com_back[cir] = solutions_back
                task_cir_pass[task_id].append(total_passed)
            cirs = sorted(cirs)
            task_cir[task_id] = cirs
            data_back.append({"task_id":task_id, "completion":com_back})
    lack_task = []
    task_cir_pass = dict(sorted(task_cir_pass.items(),key=lambda x:x[0]))
    passed_per_cir = dict(sorted(passed_per_cir.items(),key=lambda x:x[0]))
    task_has_solution = task_cir_pass.keys()
    print(f"task_has_solution: {task_has_solution}")
    for tid in range(164):
        if tid not in task_has_solution:
            lack_task.append(tid)
    pass_task_num = [0 for i in range(11)]
    for k,v in passed_per_cir.items():
        print(f"cir {k},passed {len(v)} tasks, pass rate is {len(v)/164}")
        print(f"pass tasks are:\n{sorted(v)}")
        pass_task_num[k] = len(v)
    print(f"pass task num: {pass_task_num}")
    pass_task_rate = [x/164 for x in pass_task_num]
    print(f"pass task rate: {pass_task_rate}")
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
    print(f"lack task : {lack_task}")
    if output_file:
        with open(output_file,"w+") as f:
            for data in data_back:
                f.write(json.dumps(data)+"\n")
    
    return passed_per_cir,task_cir,lack_task


def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def get_pass_k(result_file,output_file=None,k=10,n=10):
    passed_per_cir = defaultdict(set)
    task_cir = defaultdict(list)
    task_cir_pass = defaultdict(list)
    problems = read_problems()
    data_back = []
    checked_task = set()
    pass_k_list = defaultdict(list)
    with open(result_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            task_id = tid_to_int(data["task_id"])#data["task_id"]
            if task_id in checked_task:
                continue
            checked_task.add(task_id)
            completion = data["completion"]
            problem = problems[data["task_id"]]
            com_back = {}
            cirs = []
            for cir,solutions in completion.items():
                cir = int(cir)
                cirs.append(cir)
                print(f"Task {task_id} gens {len(solutions)} solutions in cir {cir}")
                assert cir <= 10
                solutions_back = []
                total_passed = False
                solutions = solutions[:k]
                passed_num = 0
                for i,solution in enumerate(solutions):
                    solution = solution["solution"]
                    passed = get_truePass(problem,solution)
                    print(f"solution {i} passed {passed}")
                    if passed:
                        passed_num += 1
                        total_passed =True
                        for c in range(cir,11):
                            passed_per_cir[c].add(task_id)
                    solutions_back.append((solution,passed))
                pass_k_list[cir].append(estimator(n,passed_num,k))
                com_back[cir] = solutions_back
                task_cir_pass[task_id].append(total_passed)
            cirs = sorted(cirs)
            task_cir[task_id] = cirs
            data_back.append({"task_id":task_id, "completion":com_back})
    lack_task = []
    task_cir_pass = dict(sorted(task_cir_pass.items(),key=lambda x:x[0]))
    passed_per_cir = dict(sorted(passed_per_cir.items(),key=lambda x:x[0]))
    task_has_solution = task_cir_pass.keys()
    print(f"task_has_solution: {task_has_solution}")
    for tid in range(164):
        if tid not in task_has_solution:
            lack_task.append(tid)
    pass_task_num = [0 for i in range(11)]
    for k,v in passed_per_cir.items():
        print(f"cir {k},passed {len(v)} tasks, pass rate is {len(v)/164}")
        print(f"pass tasks are:\n{sorted(v)}")
        pass_task_num[k] = len(v)
    print(f"pass task num: {pass_task_num}")
    pass_task_rate = [x/164 for x in pass_task_num]
    print(f"pass task rate: {pass_task_rate}")
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
    print(f"lack task : {lack_task}")
    if output_file:
        with open(output_file,"w+") as f:
            for data in data_back:
                f.write(json.dumps(data)+"\n")
    
    for cir,v in pass_k_list.items():
        pass_at_k = sum(v)/len(v)
        print(f"cir {cir}, pass at k {k} rate: {pass_at_k}")
    
    return passed_per_cir,task_cir,lack_task,pass_k_list




def get_pass_1(result_file,output_file=None):
    passed_per_cir = defaultdict(set)
    task_cir = defaultdict(list)
    task_cir_pass = defaultdict(list)
    problems = read_problems()
    data_back = []
    checked_task = set()
    with open(result_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            task_id = tid_to_int(data["task_id"])#data["task_id"]
            if task_id in checked_task:
                continue
            checked_task.add(task_id)
            completion = data["completion"]
            problem = problems[data["task_id"]]
            com_back = {}
            cirs = []
            for cir,solutions in completion.items():
                cir = int(cir)
                cirs.append(cir)
                print(f"Task {task_id} gens {len(solutions)} solutions in cir {cir}")
                assert cir <= 10
                solutions_back = []
                total_passed = False
                solution = solutions[0]["solution"]
                passed = get_truePass(problem,solution)
                if passed:
                    total_passed =True
                    for c in range(cir,11):
                        passed_per_cir[c].add(task_id)
                solutions_back.append((solution,passed))
                com_back[cir] = solutions_back
                task_cir_pass[task_id].append(total_passed)
            cirs = sorted(cirs)
            task_cir[task_id] = cirs
            data_back.append({"task_id":task_id, "completion":com_back})
    lack_task = []
    task_cir_pass = dict(sorted(task_cir_pass.items(),key=lambda x:x[0]))
    passed_per_cir = dict(sorted(passed_per_cir.items(),key=lambda x:x[0]))
    task_has_solution = task_cir_pass.keys()
    print(f"task_has_solution: {task_has_solution}")
    for tid in range(164):
        if tid not in task_has_solution:
            lack_task.append(tid)
    pass_task_num = [0 for i in range(11)]
    for k,v in passed_per_cir.items():
        print(f"cir {k},passed {len(v)} tasks, pass rate is {len(v)/164}")
        print(f"pass tasks are:\n{sorted(v)}")
        pass_task_num[k] = len(v)
    print(f"pass task num: {pass_task_num}")
    pass_task_rate = [x/164 for x in pass_task_num]
    print(f"pass task rate: {pass_task_rate}")
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
    print(f"lack task : {lack_task}")
    if output_file:
        with open(output_file,"w+") as f:
            for data in data_back:
                f.write(json.dumps(data)+"\n")
    
    return passed_per_cir,task_cir,lack_task


# def get_pass_1(result_file):
#     passed_per_cir = defaultdict(set)
#     task_cir = defaultdict(list)
#     task_cir_pass = defaultdict(list)
#     problems = read_problems()
#     with open(result_file,"r") as f:
#         for line in f.readlines():
#             data = json.loads(line)
#             task_id = data["task_id"]
#             completion = data["completion"]
#             problem = problems[task_id]
#             for cir,solutions in completion.items():
#                 cir = int(cir)
#                 print(f"Task {task_id} gens {len(solutions)} solutions in cir {cir}")
#                 total_passed = False
#                 solution = solutions[0]["solution"]
#                 passed = get_truePass(problem,solution)
#                 if passed:
#                     total_passed =True
#                 task_cir_pass[task_id].append(total_passed)
#                 # if total_passed:
#                 #     passed_per_cir[cir].add(task_id)
#                 task_cir[task_id].append(cir)
#     # for i in range(1,11):
#     #     passed_per_cir[i] = passed_per_cir[i].union(passed_per_cir[i-1])
#     task_id_list = ["HumanEval/" + str(i) for i in range(164)]
#     lack_task = []
#     for t,v in task_cir_pass.items():
#         if t not in task_id_list:
#             lack_task.append(t)
#         n = len(v)
#         for i in range(n,11):
#             task_cir_pass[t].append(v[n-1])
#     for t,v in task_cir_pass.items():
#         for i,passed in enumerate(v):
#             if passed:
#                 passed_per_cir[i].add(t)
#     for k,v in passed_per_cir.items():
#         tid_int = [int(x.split("/")[1]) for x in v]
#         print(f"cir {k},passed {len(v)} tasks, pass rate is {len(v)/164}")
#         print(f"pass tasks are:\n{sorted(list(tid_int))}")
#     # for k,v in task_cir_pass.items():
#     #     print(f"task {k} pass or not for each cir: {v}")
#     print("--------------------------------------------")
#     all_True = 0
#     all_False = 0
#     Both_True_False = 0
#     for k,v in task_cir_pass.items():
#         if True in v and False in v:
#             Both_True_False += 1
#             print(f"task {k} pass or not for each cir: {v}")
#         elif True in v and False not in v:
#             all_True += 1
#         else:
#             all_False += 1
#     print(f"all true: {all_True}, all false: {all_False}, both: {Both_True_False}")
#     print("--------------------------------------------")
#     print(f"lack task : {lack_task}")
#     return passed_per_cir,task_cir,lack_task



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

def draw_bars(data,image_path):
    xs = []
    ys = []
    for k,v in data.items():
        xs.append(k)
        ys.append(v)
    fig = plt.figure(figsize=(5,3),dpi=400)
    plt.xlabel("模型输入固定部分所占百分比%")
    plt.ylabel("数量")
    title = image_path.split("/")[-1].split(".")[0]
    plt.title(title)
    plt.bar(x=xs,height=ys,color="blue")
    for xt,yt in zip(xs,ys):
        plt.text(xt,yt,yt)
    fig.savefig(image_path)
    return


def data_counter(data:list,data_to_task:dict):
    split_number = [0, 1, 10, 100]
    data_average = sum(data)/len(data)
    print(f"average: {data_average}")
    for i in range(1,len(split_number)):
        left = split_number[i-1]
        right = split_number[i]
        filter = [ x for x in data if x >= left and x <=right]
        if len(filter)==0:
            print(f"There are {len(filter)} number in data between {left} and {right}")
        else:
            filter_average = sum(filter)/len(filter)
            print(f"There are {len(filter)} number in data between {left} and {right}, with average {filter_average}")
            filter_uniform = [int(x/(0.1*right)) for x in filter]
            print(f"counter : {Counter(filter_uniform)}")
            if len(filter) < 30:
                filter_task = [data_to_task[x]["task_id"] for x in filter]
                print(f"task id : {filter_task}")
    return
def time_evaluate(resfile):
    run_solution = []
    choose_solution = []
    model_inference = []
    rs_to_data = {}
    cs_to_data = {}
    ms_to_data = {}
    with open(resfile,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            task_id = data["task_id"]
            time_records = data["time_record"]
            for time_record in time_records:
                run_solution.append(time_record["run_solutions_time"])
                rs_to_data[time_record["run_solutions_time"]] = data
                choose_solution.append(time_record["choose_solution_time"])
                cs_to_data[time_record["choose_solution_time"]] = data
                model_inference.append(time_record["model_inference_time"])
                ms_to_data[time_record["model_inference_time"]] = data
    print("run solution time:")
    data_counter(run_solution,rs_to_data)
    print("choose solution time:")
    data_counter(choose_solution,cs_to_data)
    print("model inference time:")
    data_counter(model_inference,ms_to_data)
    return

def fix_percent_eval(resfile):
    fix_percents = []
    fix_lengths = []
    change_lengths = []
    with open(resfile,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            task_id = data["task_id"]
            fix_records = data["fix_record"]
            for fix_record in fix_records:
                for fix_percent in fix_record["fix_percents"]:
                    total_length = fix_percent[0]
                    fix_length = fix_percent[1]
                    change_length = total_length - fix_length
                    fp = fix_percent[2]
                    fix_lengths.append(fix_length)
                    change_lengths.append(change_length)
                    fix_percents.append(fp)
    fix_lengths = [int(x/100) for x in fix_lengths]
    change_lengths = [int(x/100) for x in change_lengths]
    fix_percents = [int(x/0.1) for x in fix_percents]
    print(f"fix_lengths: {Counter(fix_lengths)}")
    print(f"change_lengths: {Counter(change_lengths)}")
    print(f"fix percents: {Counter(fix_percents)}")
    

def print_anwser(result_file,task_list):
    problems = read_problems()
    with open(result_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            task_id = data["task_id"]
            completion = data["completion"]
            problem = problems[task_id]
            if task_id in task_list:
                for cir,solutions in completion.items():
                    cir = int(cir)
                    print(f"Task {task_id} gens {len(solutions)} solutions in cir {cir}")
                    total_passed = False
                    solutions_back = []
                    for i,solution in enumerate(solutions):
                        solution = solution["solution"]
                        print("+++++++++++++++++++++++++")
                        print(solution)
                        print("++++++++++++++++++++++")
    return 
   
def main(res_file=""):
    # file = "../res/UTfeedback_multi_7b16k_full.jsonl"
    file = "../res/treesearch_SBSP10_codellama7binst_pT.jsonl"
    # file = "/home/S/hexiaolong/codex/self-debug/res/SBP/UTfeedback_multiSBP10_7b16k_tT.jsonl"
    # resfiles = ["UTfeedback_multiCODETfilter_7b16k_pT_29_full.jsonl","UTfeedback_multiCODETfilter_7b16k_pT_58_full.jsonl","UTfeedback_multiCODETfilter_7b16k_pT_86_full.jsonl","UTfeedback_multiCODETfilter_7b16k_pT_113_full.jsonl","UTfeedback_multiCODETfilter_7b16k_pT_140_full.jsonl","UTfeedback_multiCODETfilter_7b16k_pT_163_full.jsonl"]
    # res_root = "../res/"
    # resfiles = [res_root+f for f in resfiles]
    # get_pass_all_multi_files(resfiles)
    # output_file = "../res/multi_with_pass/UTfeedback_multiSBSP10_7b16k_tT.jsonl"
    # print_anwser(file,["HumanEval/76"])
    # passed_per_cir,task_cir,lack_task = get_pass_n(file)
    passed_per_cir,task_cir,lack_task,pass_k_list = get_pass_k(file,k=1,n=10)
    # passed_per_cir,task_cir = get_pass_n(file)
    # data = {"SBSP10_7b16k_tT":passed_per_cir}
    # time_evaluate(file)
    # fix_percent_eval(file)
    # draw_plots(data=data, image_path="../image/UTfeedback_multiSBSP10_7b16k_tT.jpg")
    file_list = [
        "UTfeedback_PassRate_correctrm10_7b16k_pT.jsonl",
        "UTfeedback_checkrate_7b16k_pT.jsonl",
        "UTfeedback_CODETRate_correctrm_mix09_7b16k_pT.jsonl",
        "UTfeedback_CODETRate_correctrm_mix05_7b16k_pT.jsonl",
        "UTfeedback_CODETPoint_alltestcaserm_7b16k_tT.jsonl",
        "UTfeedback_PassRate_correctrm10_7b16k_pT.jsonl",
        "UTfeedback_PassRate_mix05_10_7b16k_pT.jsonl",
        "UTfeedback_PassRate_mix09_10_7b16k_pT.jsonl",
        "UTfeedback_PassRate_mix092_10_7b16k_pT.jsonl",
        "UTfeedback_PassRate_mix10_10_7b16k_pT.jsonl",
        "UTfeedback_SBSP_7b16k_halftT.jsonl",
        "UTfeedback_SBSP_7b16k_halftT2.jsonl",#一般check test测试2
        "UTfeedback_SBSP_7b16k_halftT3_s1000.jsonl",
        "UTfeedback_SBSP_7b16k_halftT4.jsonl",#随机一半check 36,37,129,130,153,156
        "UTfeedback_SBSP_7b16k_halftT5_s10.jsonl",# 33,94,129,130,153,163
        "UTfeedback_CODETPointtry2_7b16k_pT.jsonl",
        "UTfeedback_SBSP_7b16k_halftT6s256.jsonl",
        "UTfeedback_CODETv3_7b16k_pT.jsonl",
        "UTfeedback_CODETv3_t3_7b16k_pT.jsonl",
        "UTfeedback_CODETv3_t8_7b16k_pT.jsonl",
        "UTfeedback_CODETv3_sortby_solution_num_7b16k_pT.jsonl",
        "treesearch_SBSP10_7b16k_pT.jsonl",
    ]
   
if __name__=="__main__":
    fire.Fire(main)
    # lack = [0, 2, 4, 7, 12, 15, 23, 28, 29, 30, 31, 34, 35, 37, 40, 42, 43, 44, 45, 48, 51, 52, 53, 55, 58, 60, 67, 72, 74, 94, 104, 105, 107, 113, 115, 116, 124, 129, 162]
    # ignore = [0, 2, 4, 7, 12, 15, 23, 28, 29, 30, 31, 34, 35, 40, 42, 43, 44, 45, 48, 51, 52, 53, 55, 58, 60, 124, 162, 72, 6, 39, 105,115]
    # passed = [0, 8,27, 37, 42, 46, 48, 47, 47, 47, 47]
    # print(len(ignore))
    # print(set(lack) - set(ignore))
    # print(set(ignore) - set(lack))
    # passed = [x+27 for x in passed]
    # passed[7] += 1
    # passed[8] += 1
    # passed[9] += 2
    # passed[10] += 2
    # print(passed)
    
    