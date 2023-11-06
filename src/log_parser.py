import re
from collections import defaultdict
import time
import json
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append("/home/S/hexiaolong/codex/human-eval")
sys.path.append("/home/S/hexiaolong/codex/human-eval/human_eval")
from execution import run_code_with_output2, check_correctness
from data import  read_problems
from concurrent.futures import ThreadPoolExecutor
"""
此文件包含分析humaneval self-debug实验中输出的log的各类函数。
"""
# 这个函数主要是找到input file中的unchanged solution行，用来统计每个task出现生成的代码不变的轮次
def grep_solution_changed(input_file):
    # input_file = "humaneval_UTfeedback_13bv02.bak"
    s = "unchanged solution"#    #"solution unchanged"#"result:{\'task_id\'"
    # c = defaultdict(list)
    # rc = defaultdict(list)
    changed_task_per_cir = defaultdict(list)
    unchanged_task_per_cir = defaultdict(list)
    start = time.time()
    
    with open(input_file,"r") as inf:
        lines = inf.readlines()
        for line in lines:
            if s in line:
                tid = int(line.split("/")[1])
                # tid = int(line.split)
                # cir_match = re.search(r"solution unchanged in cir (.*?) task",line) # solution unchanged in cir 8 task HumanEval/163
                cir_match = re.search(r"unchanged solution in cir\[(.*?)\] with",line) #unchanged solution in cir[0] with task HumanEval/17
                cir = cir_match.group(1)
                cir = int(cir)
                # c[tid].append(cir)
                unchanged_task_per_cir[cir].append(tid)
    for cir in range(10):
        for i in range(164):
            if i not in unchanged_task_per_cir[cir]:
                changed_task_per_cir[cir].append(i)
        
    # for i in range(164):
    #     rc[len(c[i])].append(i)
    # print("="*56)
    # x = []
    # y = []
    # for k,v in changed_task_per_cir.items():
    #     print(f"In cir {cir},task {v} changed")
    #     x.append(k)
    #     y.append(len(v))

    # fig = plt.figure(figsize=(4,4),dpi=400)
    # plt.plot(x,y)
    # plt.xlabel("unchanged cir number")
    # plt.ylabel("task numbers")
    # plt.title(input_file.split(".")[0])
    
    # image_path = "./image/" + input_file.split(".")[0] + "_unchanged_cir.jpg"
    # print(f"save image to {image_path}")
    # fig.savefig(image_path)
    # end = time.time()
    # print("used time:",(end-start)/60)
    for k,v in changed_task_per_cir.items():
        print(f"In cir {k},changed tasks are: {v} ")
    return changed_task_per_cir

# 统计每轮中，判定为通过的task数量
def extract_cir_info(file):
    passed_per_cir = defaultdict(set)
    with open(file,"r") as f:
        lines = f.readlines()
        for line in lines:
            if "task:HumanEval" in line:
                # print(line.strip("\n"))
                taskid_match = re.search(r'task:HumanEval/(.*?),',line)
                if taskid_match is None:
                    print("unparsered taskid")
                cir_match = re.search(r'cir:(.*?),',line) #cir_match = re.search(r'cir(\d+)',line)# 
                if cir_match is None:
                    print("unparsered cir_match")
                passed_match = re.search(r"passed:(\w+)",line)
                if passed_match is None:
                    print("unparsered passed_match")
                # result_match = re.search(r"'passed': (.*?), 'result': (.*?)",line) #'passed': False, 'result': 
                # if result_match is None:
                #     print("unparsered result_match")
                if taskid_match and passed_match  and cir_match:#and result_match
                    taskid = taskid_match.group(1)
                    cir = int(cir_match.group(1))
                    passed = passed_match.group(1)
                    # print(passed)
                    # result_value = result_match.group(2)
                    # print(passed)
                    if passed=="True":
                        if taskid not in passed_per_cir[cir]:
                            passed_per_cir[cir].add(taskid)
    # print("="*56)
    print(f"Cir 0 passed task number is {len(passed_per_cir[0])}.List of passed task:\n {sorted(passed_per_cir[0],key=lambda x: int(x)) }")
    for i in range(1,10):
        print(f"Cir {i} passed task {passed_per_cir[i]}")
        passed_per_cir[i] = passed_per_cir[i].union(passed_per_cir[i-1])
    # print("="*56)
    # x = range(10)
    # x_ticks = [f"cir{cir}" for cir in x]
    # y = [len(passed_per_cir[cir]) for cir in x]
    # figure = plt.figure(figsize=(4,4),dpi=400)
    # # 设置坐标轴刻度
    # plt.xticks(x,x_ticks)
    # plt.yticks(y)
    # plt.xlabel("Cirs")
    # plt.ylabel("Passed number")
    # plt.title(file.split(".")[0])
    
    # plt.plot(x,y,color='red', linewidth=2, linestyle='-',label="passed task per cir",marker="o")
    
    # image_path = "./image/" + file.split(".")[0] + "_passed_cir.jpg"
    # print(f"save image to {image_path}")
    # figure.savefig(image_path)  
    return passed_per_cir              

def draw_plots(data,image_path):
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
        
        
def get_true_pass():
    file = "./human-eval/UTfeedback_13bv1_t1_trueTest_full.jsonl"
    truePass_per_cir = defaultdict(list)
    d = dict()
    with open(file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            task_id = data["task_id"]
            completions = data["completion"]
            for completion in completions:
                cir = completion["cir"]
                solution = completion["solution"]
                passed = completion["passed"]
                if passed:
                    print(passed)
                    for c in range(cir,10):
                        truePass_per_cir[c].append(task_id)
    # for i in range(1,10):
    #     truePass_per_cir[i] += truePass_per_cir[i-1]
    d["t=1.0"] = truePass_per_cir
    draw_plots(d,"UTfeedback_13bv1_t1_trueTest_truePass.jpg")
    return truePass_per_cir
                
def get_truePass(problem,solution):
    with ThreadPoolExecutor(max_workers=1) as executor:
        args = (problem, solution, 3.0)
        future = executor.submit(check_correctness, *args)
        result = future.result()
        passed = result["passed"]
    return passed
   
def extract_solutions_from_log(log_file):
    print(f"extract_solutions_from_log {log_file}")
    full_solution_file = log_file[:-4]+ "_full.jsonl"
    problems = read_problems()
    truePass_per_cir = defaultdict(set)
    solution_per_cir = defaultdict(set)
    d = dict()
    s = ""
    with open(log_file,"r") as f: 
        # lines = f.readlines()
        tid = ""
        origin_code_flag = False
        fix_code_flag = False
        origin_code = ""
        fix_code = ""
        fix_code_count = 0
        codes = []
        solutions_per_task = {}
        for line in f.readlines():
            if line.startswith("get solution"):
                if tid!="":
                    solutions_per_task[tid] = codes
                    codes = []
                tid = line.split(":")[1].strip()
                # print(f"task id:{tid}")
                fix_code_count=0
            elif "filter code" in line: # filter solution
                origin_code_flag = True
                # print("start solution")
            elif line == "--------------------\n" and origin_code_flag: #++++++++++++++++++++++++++++++++++++++++++++\n
                origin_code_flag = False
                codes.append({"cir":-1,"solution":origin_code})
                # print(f"origin_code:\n{origin_code}")
                origin_code = ""
            elif "filter fix ans" in line: #line == "----------------filter fix ans---------------\n":# -------------filter fix ans----------------
                fix_code_flag = True
                # print("start fix solution")
            elif "fix end" in line: #line == "==================fix end======================\n" and fix_code_flag:# ============fix end===============
                fix_code_flag = False
                codes.append({"cir":fix_code_count,"solution":fix_code})
                # print(f"fix_code_{fix_code_count}:\n{fix_code}")
                fix_code = ""
                fix_code_count +=1
            elif origin_code_flag:
                origin_code += line
            elif fix_code_flag:
                fix_code += line
        # with open(full_solution_file,"w+") as f:
        #     for k,v in solutions_per_task.items():
        #         f.write(json.dumps({"task_id":k,"completion":v}))
        
        for k,v in solutions_per_task.items():
            n = len(v)
            s = v[n-1]["solution"]
            for c in range(n-1,10):
                solutions_per_task[k].append({"cir":c,"solution":s})
        for k,v in solutions_per_task.items():
            for solution in v:
                passed = get_truePass(problems[k],solution["solution"])
                solution_per_cir[solution["cir"]+1].add(k)
                if passed:
                    truePass_per_cir[solution["cir"]+1].add(k)
        # for i in range(0,11):
        #     print(f"cir i total task num is {len(solution_per_cir[i])}")
        #     print(f"cir i passed {len(truePass_per_cir[i])} tasks:{truePass_per_cir[i]}")
        # for i in range(1,11):
        #     truePass_per_cir[i] = truePass_per_cir[i].union(truePass_per_cir[i-1])
        # for i in range(0,11):
        #     print(f"After union cir i passed {len(truePass_per_cir[i])} tasks:{truePass_per_cir[i]}")
        print(truePass_per_cir.keys())
    return truePass_per_cir

    # with open(file,"r") as f:
    #     for line in f.readlines():
    #         data = json.loads(line)
    #         task_id = data["task_id"]
    #         completions = data["completion"]
    #         for completion in completions:
    #             cir = completion["cir"]
    #             passed = completion["passed"]
    #             if passed:
    #                 truePass_per_cir[cir].append(task_id)
    # d["t=1.0"] = truePass_per_cir
    # draw_plots(d,"UTfeedback_13bv1_truePass.jpg")
    # return truePass_per_cir
ROOT = "./2023-10-16/" 
codellama_root = "./codellama/"
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k","--kind",required=True,type=int,help="function kind")
    args = parser.parse_args()
    kind = args.kind
    
    if kind==0:
        # file = args.file#"humaneval_simfeedback_lla13b_t08.out"
        data1 = dict()
        data2 = dict()
        # files = ["humaneval_simfeedback_13bv1.out","humaneval_simfeedback_13bv1_t04.out","humaneval_simfeedback_13bv1_t08.out","humaneval_simfeedback_13bv1_t1.out"]
        files = ["humaneval_UTfeedback_13bv1_t1_codeTTest.out","humaneval_UTfeedback_13bv1_t1_trueTest.out",ROOT+"humaneval_UTfeedback_13bv1_t1.out"]
        # files = [codellama_root+"UTfeedback_cola7bpy_t1_trueTest.out",codellama_root+"UTfeedback_cola7bpy_t1.out"]
        labels = ["codeTTest","trueTest","promptTest"]# "t=0.0","t=0.4","t=0.8",
        for i,file in enumerate(files):
            print("-"*50)
            changed_task_per_cir = grep_solution_changed(file)
            print("")
            print("*"*50)
            print("")
            passed_per_cir = extract_cir_info(file)
            print("-"*50)
            data1[labels[i]] = changed_task_per_cir
            data2[labels[i]] = passed_per_cir
        draw_plots(data1,"UTfeedback_13bv1_t1_changedtask.jpg")
        draw_plots(data2,"UTfeedback_13bv1_t1_passedtask.jpg")
    elif kind==1:
        get_true_pass()
    elif kind==2:
        data = {}
        codellma_root = "./codellama/"
        log_files = ["UTfeedback_cola34bpy_t1.out","UTfeedback_cola34bpy_t1_trueTest.out"]#, "humaneval_UTfeedback_cola34b_t1_codeTTest.out"
        labels = ["promptTest","trueTest"]#,"codeTTest"
        for logf,label in zip(log_files,labels):
            if "cola" in logf:
                logf = codellama_root + logf
            truePass_per_cir = extract_solutions_from_log(logf)
            data[label] = truePass_per_cir
        draw_plots(data,"UTfeedback_cola34b_t1_difftests.jpg")
        