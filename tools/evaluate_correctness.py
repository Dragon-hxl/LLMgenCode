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
    
                
if __name__=="__main__":
    # file = "../res/UTfeedback_multi_7b16k_full.jsonl"
    file = "../res/UTfeedback_CODETPassfirst_7b16k_pT.jsonl"
    # resfiles = ["UTfeedback_multiCODETfilter_7b16k_pT_29_full.jsonl","UTfeedback_multiCODETfilter_7b16k_pT_58_full.jsonl","UTfeedback_multiCODETfilter_7b16k_pT_86_full.jsonl","UTfeedback_multiCODETfilter_7b16k_pT_113_full.jsonl","UTfeedback_multiCODETfilter_7b16k_pT_140_full.jsonl","UTfeedback_multiCODETfilter_7b16k_pT_163_full.jsonl"]
    # res_root = "../res/"
    # resfiles = [res_root+f for f in resfiles]
    # get_pass_all_multi_files(resfiles)
    # output_file = "../res/multi_with_pass/UTfeedback_multiSBSP10_7b16k_tT.jsonl"
    passed_per_cir,task_cir = get_pass_n(file)
    # passed_per_cir,task_cir = get_pass_n(file)
    # data = {"SBSP10_7b16k_tT":passed_per_cir}
    # time_evaluate(file)
    # fix_percent_eval(file)
    # draw_plots(data=data, image_path="../image/UTfeedback_multiSBSP10_7b16k_tT.jpg")
                
                    