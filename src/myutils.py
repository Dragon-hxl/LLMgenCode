import torch
import json
import argparse
import re
import time
import math
import sys
import random
import numpy as np
sys.path.append("/home/S/hexiaolong/codex/self-debug")
sys.path.append("/home/S/hexiaolong/codex/self-debug/humaneval")
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from utils.obj import Node
from human_eval.data import  read_problems
from human_eval.execution import check_test_correctness,run_code_with_output_CODET,run_code_with_output_CODET3,_pack_test_cases


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return

def make_printv(verbose: bool, log_file = ""):
    def print_v(*args, **kwargs):
        if verbose:
            kwargs["flush"] = True
            if log_file != "":
                f = open(log_file,"w+")
                kwargs["file"] = f
            print(*args, **kwargs)
        else:
            pass
    return print_v

def print_with_tag(content,tag,verbose: bool):
    print_v = make_printv(verbose)
    num = 60
    if verbose:
        part = (num - len(tag))//2
        print_v("+"*part + tag + "+"*part)
        print_v(content)
        print_v("+"*num)
    else:
        pass
    return

def load_testcase(test_file,type:int = 0):
    
    testcase = {}           
    if type == 0: 
        # {tid:testcase} each line
        with open(test_file) as f:
            for line in f.readlines():
                d = json.loads(line)
                for k,v in d.items():
                    testcase[k]=v
    elif type == 1:
        # {"task_id":tid,"ios":[{"tin":tin,"tout":tout}]} each line (uniform format)
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
    elif type ==2:
        # {"task_id":tid,"testcases":[...]} each line
        with open(test_file,"r") as f:
            for line in f.readlines():
                data = json.loads(line)
                tid = data["task_id"]
                tests = data["testcases"]
                testcase[tid] = tests
    return testcase

def get_args():
    """
    get args : model_path, output_file , verbose
    """
    parser = argparse.ArgumentParser(description="human_eval simplefeedback")
    parser.add_argument('-mf','--model_path',default='/lustre/S/hexiaolong/vicuna-7b-v1.1',required=True, help="file path where the model used locates")
    parser.add_argument('-o','--output',default="ouput.json",required=True, help="file where the output store")
    parser.add_argument("-v","--verbose", default=False, action="store_true", help="show log")

    args = parser.parse_args()
    return args

def map_gpu_memory(used_gpu:list=[],low_0: bool=True):
    """
    为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分.used_gpu参数指定map的GPU
    """
    gpu_memory = []
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    max_memory_mapping = {
                        i: str(int(gpu_memory[i] * 0.85)) + "GiB" # 在这里设置每个GPU的显存分配比例
                        for i in range(num_gpus)
                    }
    memory_mapping ={}
    if used_gpu!=[]:
        for i in used_gpu:
            memory_mapping[i] = max_memory_mapping[i]
        max_memory_mapping = memory_mapping
    # if low_0 and (0 in max_memory_mapping.keys()):
    #     max_memory_mapping[0] = gpu_memory[0] * 0.5
    print(max_memory_mapping)
    return max_memory_mapping

def get_unit_test(ut_file, chosen_num=1000,verbose=False):
    """
    从文件中加载unit test，参数ut_file是文件名， chosen_num每个task加载的test数量，verbose是否打印细节信息
    """
    unit_tests = {}
    assertions = defaultdict(list)
    assertion_string = {}
    with open(ut_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["task_id"]
            ios = data["ios"][:chosen_num]
            unit_tests[tid]=ios
            for io in ios:
                assertions[tid].append("assert "+io["tin"] + " == " + io["tout"])
            assertion_string[tid] = "\n".join(assertions[tid])
            if verbose:
                print(f"Read {len(assertions[tid])} tests from file for task {tid}")
    return unit_tests,assertions,assertion_string


# 接受一个完整的函数代码，去除其中的函数头和注释（humaneval检测需要）
def code_clean(code,entry_point,start_code=""):
    # regex = "\"\"\"[.\n]*\"\"\""
    # 使用正则表达式匹配和移除单行注释
    code = re.sub(r'#.*', '', code)

    # 使用正则表达式匹配和移除多行注释
    code = re.sub(r'(\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\")', '', code, flags=re.DOTALL)

    # code = re.sub(regex,"",code)
    if entry_point in code:
        res = ""
        prefix = "    "
        for line in code.split("\n"):
            if line=="" or line=="\t" or line=="    ":
                continue
            if entry_point in line:
                prefix = ""
                continue
            if line in start_code:
                continue
            if line[0]!=" " and line[0]!="\t" and not line.startswith("def") and not line.startswith("import") and not line.startswith("from"):
                continue
            res += prefix + line + "\n"
        return res
    else:
        res = ""
        prefix = "    "
        for line in code.split("\n"):
            res += prefix + line + "\n"
        return res
    return code

def code_clean2(code,entry_point,start_code=""):
    """去除code中的注释和空行，去除code中和start_code重复的部分。entry_point是一个函数名，code中在entry_point这一行之前的行要加上缩进，之后的行中的函数体要保留。
    """
    # regex = "\"\"\"[.\n]*\"\"\""
    # 使用正则表达式匹配和移除单行注释
    code = re.sub(r'#.*', '', code)

    # 使用正则表达式匹配和移除多行注释
    code = re.sub(r'(\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\")', '', code, flags=re.DOTALL)

    # code = re.sub(regex,"",code)
    if entry_point in code:
        res = ""
        prefix = "    "
        for line in code.split("\n"):
            if line=="" or line=="\t" or line=="    ":
                continue
            if entry_point in line:
                prefix = ""
                continue
            if line in start_code:# "import" in line and l
                continue
            if line[0]!=" " and line[0]!="\t" and not line.startswith("def") and not line.startswith("import") and not line.startswith("from"):
                continue
            res += prefix + line + "\n"
        return res
    else:
        res = ""
        prefix = "    "
        for line in code.split("\n"):
            res += prefix + line + "\n"
        return res
    return code

prompt_for_64 = '''def vowels_count(s):
    """Write a function vowels_count which takes a string representing
    a word as input and returns the number of vowels in the string.
    Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a
    vowel, but only when it is at the end of the given word.

    Example:
    >>> vowels_count("abcde")
    2
    >>> vowels_count("ACEDY")
    3
    """ '''
    
def get_UTfeedback_prompt(feedback_prompt, solution, code_res, run_test, test_res, assertions):
    total_tests = len(run_test)
    pass_tests = 0
    passn = 0.0
    if type(code_res) is str:
        prompt =  feedback_prompt +solution + "\n```\nFeedback: With the above function, " + run_test[0] +" returns the following error:\n\"\"\"\n"+code_res+ "\n\"\"\"\nSo the code does not pass the assertion. Please fix it.\n\n### fix result ###\n"
    else:
        utFeedback = "\n```\nFeedback: With the above function,"
        for i,cres in enumerate(code_res):
            try:
                cres = str(cres)
            except:
                continue
            if len(cres) >1024:
                print("Too long feedback, maybe the result of code is too wired!")
                continue
            # print(f"code return res is {cres}. real res is {test_res[i]}")
            if cres == test_res[i]:
                utFeedback += f" {run_test[i]} == {cres} while the assertion is \"{assertions[i]}\".The code pass this aasertion."
                pass_tests += 1
            else:
                utFeedback += f" {run_test[i]} == {cres} while the assertion is \"{assertions[i]}\".The code does not pass this aasertion."
        utFeedback += "\nSo the code is wrong. Please fix it.\n\n### fix result ###\n"
        prompt = feedback_prompt +solution + utFeedback
        if total_tests == 0:
            passn = 0.0
        else:
            passn = (1.0*pass_tests)/total_tests
    return prompt,passn

prompt_root = "/home/S/hexiaolong/codex/self-debug/data/prompt/"

def get_UTfeedback_prompt_v1(feedback_prompt, solution, passed, final_res, run_test, assertions, type="UT"):
    print(f"Gen {type} feedback prompt!")
    total_tests = len(assertions)
    pass_tests = 0
    passn = 0.0
    if type == "UT" or type=="expl":
    # print(f"run_test:{len(run_test)},pass_result:{len(pass_result)},test_result:{len(test_result)},assertions:{len(assertions)}")
        if passed:
            utFeedback = "\n```\nFeedback: With the above function,"
            pass_result = final_res["pass_result"]
            test_result = final_res["test_result"]
            # print("test_result : ",test_result)
            for i,p in enumerate(pass_result):
                try:# 避免一些极端情况的出现
                    test_res = str(test_result[i])
                except:
                    continue
                if len(test_res) >1024:
                    print("Too long feedback, maybe the result of code is too wired!")
                    continue
                if p:
                    utFeedback += f" {run_test[i]} == {test_result[i]} while the assertion is \"{assertions[i]}\".The code pass this aasertion."
                    pass_tests += 1
                else:
                    utFeedback += f" {run_test[i]} == {test_result[i]} while the assertion is \"{assertions[i]}\".The code does not pass this aasertion." 
            utFeedback += "\nSo the code is wrong. Please fix it.\n\n### fix result ###\n"
            prompt = feedback_prompt +solution + utFeedback
            if total_tests == 0:
                passn = 0.0
            else:
                passn = (1.0*pass_tests)/total_tests
        else:
            prompt =  feedback_prompt +solution + "\n```\nFeedback: With the above function, " + run_test[0] +" returns the following error:\n\"\"\"\n"+final_res+ "\n\"\"\"\nSo the code does not pass the assertion. Please fix it.\n\n### fix result ###\n"
    elif type=="simple":
        if passed:
            pass_result = final_res["pass_result"]
            test_result = final_res["test_result"]
            pass_flag = True
            for i,p in enumerate(pass_result):
                try:# 避免一些极端情况的出现
                    test_res = str(test_result[i])
                except:
                    continue
                if p:
                    pass_tests += 1
                else:
                    pass_flag = False
            if not pass_flag:
                utFeedback = "\n```\nFeedback: The code above is wrong. Please fix it.\n### fix result ###\n"
            else:
                utFeedback = "\n```\nFeedback: The code above is correct.\n"
            prompt = feedback_prompt +solution + utFeedback
            if total_tests == 0:
                passn = 0.0
            else:
                passn = (1.0*pass_tests)/total_tests
        else:
            utFeedback = "\n```\nFeedback: The code above is wrong. Please fix it.\n### fix result ###\n"
            prompt =  feedback_prompt +solution + utFeedback
        # print(f"simple feedback prompt:\n{prompt}")
    return prompt,passn,pass_tests

def get_UTfeedback_prompt_v2(feedback_prompt, solution, passed, final_res, run_test, assertions):
    total_tests = len(run_test)
    pass_tests = 0
    passn = 0.0
    # print(f"run_test:{len(run_test)},pass_result:{len(pass_result)},test_result:{len(test_result)},assertions:{len(assertions)}")
    if passed:
        utFeedback = "\n```\nFeedback: With the above function,"
        pass_result = final_res["pass_result"]
        test_result = final_res["test_result"]
        # print("test_result : ",test_result)
        for i,p in enumerate(pass_result):
            try:# 避免一些极端情况的出现
                test_res = str(test_result[i])
            except:
                continue
            if p:
                # utFeedback += f" {run_test[i]} == {test_result[i]} while the assertion is \"{assertions[i]}\".The code pass this aasertion."
                pass_tests += 1
            else:
                utFeedback += f" {run_test[i]} == {test_result[i]} while the assertion is \"{assertions[i]}\".The code does not pass this aasertion." 
        utFeedback += "\nSo the code is wrong. Please fix it.\n\n### fix result ###\n"
        prompt = feedback_prompt +solution + utFeedback
        if total_tests == 0:
            passn = 0.0
        else:
            passn = (1.0*pass_tests)/total_tests
    else:
        prompt =  feedback_prompt +solution + "\n```\nFeedback: With the above function, " + run_test[0] +" returns the following error:\n\"\"\"\n"+final_res+ "\n\"\"\"\nSo the code does not pass the assertion. Please fix it.\n\n### fix result ###\n"
    return prompt,passn,pass_tests

def filter_fix_ans(ans, entry_point, start_code,verbose=False):
    if "```" in ans: # 过滤掉生成的无用信息
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
    sol = code_clean2(code=sol,entry_point=entry_point,start_code=start_code)
    if verbose:
        print("-------------filter fix ans----------------")
        print(sol)
        print("============fix end===============")
    return sol

def log_message(message,verbose):
    if verbose:
        print(message)
    return

problems = read_problems()

def exec_solution_testcase(task_id,solution,testcase):
    problem = problems[task_id]
    test_string = f"assert {testcase['tin']} == {testcase['tout']}"
    check_string = problem["prompt"] + solution + test_string
    with ThreadPoolExecutor(max_workers=1) as executor:
        args = (check_string, 1.0)
        future = executor.submit(check_test_correctness, *args)
        result = future.result()
        passed = result["passed"]
    return passed



def print_checkp(problem,testcases):
    exec_globals = {}
    check_program = (
                problem["prompt"]+ "    pass\n" 
            )
    #对每个unit test构建语句将其执行结果赋给一个全局变量方便获取
    for i,ut in enumerate(testcases):
        exec_globals["ans_"+str(i)] = 0
        check_program = check_program + "try:\n    ans_" + str(i) + " = " + f"({ut['tin']} == {ut['tout']})" + "\nexcept:\n    ans_" + str(i) + " = False\n"
    print(check_program)
    return
def get_pass_rate(data,Node_list, testcases) -> None:
    """
    给定nodelist和testcases，计算nodelist的node代码的solution在testcases上的通过率
    """
    n = len(Node_list)
    m = len(testcases)
    for i in range(n):
        with ThreadPoolExecutor(max_workers=1) as executor:
            args = (data,Node_list[i].solution,testcases,"",0.1)
            future = executor.submit(run_code_with_output_CODET, *args)
            result = future.result()
            code_res = result["result"]
        pass_num = 0
        for j,res in enumerate(code_res):
                if type(res) is bool:
                    if res:
                        pass_num+=1
        Node_list[i].CODET_pass_rate = (1.0*pass_num)/m
    return

def get_CODET_point_v1(Node_list, testcases, task_id) -> None:
    """
    给定nodelist也就是solutions和testcases，通过CODET方法计算各自权重得分。
    """
    start_time = time.time()
    solution_pass_test = defaultdict(set)
    # solution_group = defaultdict(set)
    solution_group = []
    # group_score = defaultdict(int)
    grouped = []
    verbose = True
    n = len(Node_list)
    m = len(testcases)
    
    for i in range(n):
        grouped.append(False)
    log_message("Run solution and test case...",verbose)
    for i in range(n):
        if Node_list[i].already_CODET:
            solution_pass_test[i] =  Node_list[i].CODET_pass_testcase
            continue
        with ThreadPoolExecutor(max_workers=1) as executor:
            args = (problems[task_id],Node_list[i].solution,testcases,"",0.1)
            future = executor.submit(run_code_with_output_CODET, *args)
            result = future.result()
            passed = result["passed"]
            code_res = result["result"]
        # result = run_code_with_output_CODET(problems[task_id],Node_list[i].solution,testcases,"",0.1)
        # code_res = result["result"]
        # print(f"task:{task_id},solution:{i},passed:{result["passed"]},result:{code_res}")
        if type(code_res) is str:
            pass
        else:
            for j,res in enumerate(code_res):
                if type(res) is bool:
                    if res:
                        solution_pass_test[i].add(j)
        Node_list[i].CODET_pass_testcase = solution_pass_test[i]
        Node_list[i].already_CODET = True
    log_message("Group solution...",verbose)            
    for i in range(n):
        if grouped[i]:
            continue
        group = set()
        group.add(i)
        grouped[i] = True
        for j in range(i+1,n):
            if solution_pass_test[i] == solution_pass_test[j]:
                group.add(j)
                grouped[j]=True
        score = math.sqrt(len(group)) * len(solution_pass_test[i])
        pass_testcase = solution_pass_test[i]
        solution_group.append((group,pass_testcase,score))
        # group_score[i] = math.sqrt(len(group)) * len(solution_pass_test[i])
        log_message(f"group {group} scores {score}",verbose)
        for sol_id in group:
            Node_list[sol_id].CODET_point = score
            Node_list[sol_id].CODET_total_test_num = m
    end_time = time.time()
    log_message(f"Spends {(end_time-start_time)/60} mins",verbose)
    sorted_group = sorted(solution_group,key=lambda x: x[1],reverse=True)
    for i,s in enumerate(sorted_group):
        print(f"The {i} rate CODET group pass testcases are {s}")
    
    return sorted_group

def get_CODET_point_v2(Node_list, testcases, task_id , limit=1) -> None:
    """
    给定solutions和testcases，通过CODET计算各自权重分数，limit会指定从排名前limit的集合中选取solution返回
    """
    start_time = time.time()
    solution_pass_test = defaultdict(set)
    solution_group = []
    group_score = []
    grouped = []
    verbose = True
    n = len(Node_list)
    m = len(testcases)
    
    for i in range(n):
        grouped.append(False)
    log_message("Run solution and test case...",verbose)
    # print_checkp(problems[task_id],testcases)    
    for i in range(n):
        if Node_list[i].already_CODET:
            solution_pass_test[i] =  Node_list[i].CODET_pass_testcase
            continue
        with ThreadPoolExecutor(max_workers=1) as executor:
            args = (problems[task_id],Node_list[i].solution,testcases,"",0.1)
            future = executor.submit(run_code_with_output_CODET, *args)
            result = future.result()
            passed = result["passed"]
            code_res = result["result"]
        # print(f"task:{task_id},solution:{i},passed:{passed},result:{code_res}")
        # result = run_code_with_output_CODET(problems[task_id],Node_list[i].solution,testcases,"",0.1)
        code_res = result["result"]
        # print(f"task:{task_id},solution:{i},passed:{result["passed"]},result:{code_res}")
        if type(code_res) is str:
            pass
        else:
            for j,res in enumerate(code_res):
                if type(res) is bool:
                    if res:
                        solution_pass_test[i].add(j)
        Node_list[i].CODET_pass_testcase = solution_pass_test[i]
        Node_list[i].already_CODET = True
    log_message("Group solution...",verbose)            
    for i in range(n):
        if grouped[i]:
            continue
        group = set()
        group.add(i)
        grouped[i] = True
        for j in range(i+1,n):
            if solution_pass_test[i] == solution_pass_test[j]:
                group.add(j)
                grouped[j]=True
        score = math.sqrt(len(group)) * len(solution_pass_test[i]) * (1.0/m)
        solution_group.append((group,score))
        log_message(f"{group} scores {score}",verbose)
        for sol_id in group:
            Node_list[sol_id].CODET_point = score
            Node_list[sol_id].CODET_total_test_num = m
    end_time = time.time()
    log_message(f"Spends {(end_time-start_time)/60} mins",verbose)
    
    sorted_group = sorted(solution_group,key=lambda x: x[1],reverse=True)
    sorted_nodes = []
    tmplimit = 0
    for i,(k,v) in enumerate(sorted_group):
        sgroup = k
        nodes = [Node_list[nid] for nid in sgroup]
        nodes = sorted(nodes,key=lambda x: (x.passT_rate,x.prob),reverse=True)
        sorted_nodes.append(nodes)
        if v==0:
            tmplimit = i
    if n <= 10:
        return sum(sorted_nodes,[])
    if tmplimit < limit and tmplimit!=0:
        limit = tmplimit
    if limit < 1: #防止limit为0导致后面的死循环
        limit = 1
    print(f"When choose nodes according to CODET, limit = {limit}")
    limit_sorted_nodes = sorted_nodes[:limit]
    left_sorted_nodes = sorted_nodes[limit:]
    idx_record = []
    for nodes in limit_sorted_nodes:
        idx_record.append(0)
    chosen_nodes = []
    lack_num = 0
    stop = False
    # all_limit_nodes = sum(limit_sorted_nodes)
    # all_limit_nodes_len = len(all_limit_nodes)
    # if all_limit_nodes_len <= 10:
    #     chosen_nodes = all_limit_nodes
    #     left = 10 - all_limit_nodes_len
    # else:
    #     h = math.floor(10/limit)
    #     g = 10%limit
    while True:
        for i,nodes in enumerate(limit_sorted_nodes):
            if idx_record[i] >= len(nodes):
                lack_num+=1
                if lack_num > 100:
                    stop = True
                    break
                continue
            chosen_nodes.append(nodes[idx_record[i]])
            idx_record[i] = idx_record[i] + 1
            if len(chosen_nodes) == 10:
                stop = True
                break
        if stop:
            break
    left = 10 - len(chosen_nodes)
    if left > 0:
        for nodes in left_sorted_nodes:
            for node in nodes:
                chosen_nodes.append(node)
                left = left - 1
                if left == 0:
                    break
            if left == 0:
                break    
    end_time = time.time()
    log_message(f"Spends {(end_time-start_time)/60} mins",verbose)
    return chosen_nodes

def get_CODET_point_v3(Node_list, testcases, problem,chosen_num=8,sort_len=False,count_solution_num = False,verbose = False, debug = False) -> None:
    """
    给定nodelist也就是solutions和testcases，通过CODET方法计算各自权重得分。
    """
    start_time = time.time()
    solution_pass_test = defaultdict(set)
    test_pass_solution_num = {i:0 for i in range(len(testcases))}
    solution_group = []
    # group_score = defaultdict(int)
    grouped = []
    if debug:
        verbose = True
    # verbose = True
    print_v = make_printv(verbose)
    n = len(Node_list)
    m = len(testcases)
    print_v(f"node num: {n}, testcase num: {m}")
    
    for i in range(n):
        grouped.append(False)
    print_v("Run solution and test case...")
    for i in range(n):
        if Node_list[i].already_CODET:
            print_v(f"ignore solution {i}...")
            solution_pass_test[i] =  Node_list[i].CODET_pass_testcase
            continue
        try:
            if debug:
                print_v("debug step")
                print_with_tag(Node_list[i].solution,"solution",True)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    args = (problem,Node_list[i].solution,testcases,"",0.1)
                    future = executor.submit(run_code_with_output_CODET3, *args)
                    result = future.result()
            else:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    args = (problem,Node_list[i].solution,testcases,"",0.1)
                    future = executor.submit(run_code_with_output_CODET, *args)
                    result = future.result()
        except Exception as e:
            print_v(f"Run solution {i} error: {e}")
            continue
        print_v("gather result...")
        code_res = result["result"]
        print_v(f"code_res is {code_res}")
        if type(code_res) is str:
            pass
        else:
            for j,res in enumerate(code_res):
                if type(res) is bool:
                    if res:
                        solution_pass_test[i].add(j)
        Node_list[i].CODET_pass_testcase = solution_pass_test[i]
        Node_list[i].already_CODET = True
    print_v("Group solution...")            
    for i in range(n):
        if grouped[i]:
            continue
        group = set()
        group.add(i)
        grouped[i] = True
        for j in range(i+1,n):
            if solution_pass_test[i] == solution_pass_test[j]:
                group.add(j)
                grouped[j]=True
        score = math.sqrt(len(group)) * len(solution_pass_test[i])
        pass_testcase = solution_pass_test[i]
        solution_group.append((group,pass_testcase,score))
        print_v(f"group {group} scores {score}")
        for sol_id in group:
            Node_list[sol_id].CODET_point = score
            Node_list[sol_id].CODET_total_test_num = m
    end_time = time.time()
    print_v(f"Run all soluiton spends {(end_time-start_time)/60} mins")
    sorted_group = sorted(solution_group,key=lambda x: x[1],reverse=True)
    for i,s in enumerate(sorted_group):
        print_v(f"The {i} rate CODET group pass testcases are {s}")
        g = len(s[0])
        ptestcase = s[1]
        score = s[2]
        for t in ptestcase:
            if count_solution_num:
                test_pass_solution_num[t] += g
            else:
                test_pass_solution_num[t] += score
    if sort_len:
        sorted_pass_test_solution_num = sorted(test_pass_solution_num.items(),key=lambda x: (x[1],-len(testcases[x[0]])),reverse=True)#(x[1],-len(testcases[x[0]]))
    else:
        sorted_pass_test_solution_num = sorted(test_pass_solution_num.items(),key=lambda x: x[1],reverse=True)#(x[1],-len(testcases[x[0]]))
    chosen_testcase = [x[0] for x in sorted_pass_test_solution_num[:chosen_num]]
    
    return sorted_group,chosen_testcase

def get_CODET_point_v4(Node_list, testcases, task_id,chosen_num=8,sort_len=False,count_solution_num = False,verbose = False, debug = False) -> None:
    """
    给定nodelist也就是solutions和testcases，通过CODET方法计算各自权重得分。
    """
    start_time = time.time()
    solution_pass_test = defaultdict(set)
    test_pass_solution_num = {i:0 for i in range(len(testcases))}
    solution_group = []
    # group_score = defaultdict(int)
    grouped = []
    if debug:
        verbose = True
    # verbose = True
    n = len(Node_list)
    m = len(testcases)
    print(f"node num: {n}, testcase num: {m}")
    
    for i in range(n):
        grouped.append(False)
    log_message("Run solution and test case...",verbose)
    for i in range(n):
        log_message(f"ignore before {i}...",verbose)
        if Node_list[i].already_CODET:
            log_message(f"ignore solution {i}...",verbose)
            solution_pass_test[i] =  Node_list[i].CODET_pass_testcase
            continue
        try:
            if debug:
                print("debug step")
                print_with_tag(Node_list[i].solution,"solution",True)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    args = (problems[task_id],Node_list[i].solution,testcases,"",0.1)
                    future = executor.submit(run_code_with_output_CODET3, *args)
                    result = future.result()
                    # passed = result["passed"]
                    # code_res = result["result"]
                # result = run_code_with_output_CODET3(problems[task_id],Node_list[i].solution,testcases,"",0.1)
                log_message(f"result res is {result['result']}",verbose)
            else:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    args = (problems[task_id],Node_list[i].solution,testcases,"",0.1)
                    future = executor.submit(run_code_with_output_CODET, *args)
                    result = future.result()
        except Exception as e:
            # print(f"Run solution {i} error: {e}")
            continue
        log_message("gather result...",verbose)
        code_res = result["result"]
        log_message(f"code_res is {code_res}",verbose)
        # print(f"task:{task_id},solution:{i},passed:{result["passed"]},result:{code_res}")
        if type(code_res) is str:
            pass
        else:
            for j,res in enumerate(code_res):
                if type(res) is bool:
                    if res:
                        solution_pass_test[i].add(j)
        Node_list[i].CODET_pass_testcase = solution_pass_test[i]
        Node_list[i].already_CODET = True
    log_message("Group solution...",verbose)            
    for i in range(n):
        if grouped[i]:
            continue
        group = set()
        group.add(i)
        grouped[i] = True
        for j in range(i+1,n):
            if solution_pass_test[i] == solution_pass_test[j]:
                group.add(j)
                grouped[j]=True
        score = math.sqrt(len(group)) * len(solution_pass_test[i])
        pass_testcase = solution_pass_test[i]
        solution_group.append((group,pass_testcase,score))
        # group_score[i] = math.sqrt(len(group)) * len(solution_pass_test[i])
        log_message(f"group {group} scores {score}",verbose)
        for sol_id in group:
            Node_list[sol_id].CODET_point = score
            Node_list[sol_id].CODET_total_test_num = m
    end_time = time.time()
    log_message(f"Spends {(end_time-start_time)/60} mins",verbose)
    sorted_group = sorted(solution_group,key=lambda x: x[1],reverse=True)
    for i,s in enumerate(sorted_group):
        print(f"The {i} rate CODET group pass testcases are {s}")
        g = len(s[0])
        ptestcase = s[1]
        for t in ptestcase:
            if count_solution_num:
                print("count solution num")
                test_pass_solution_num[t] += g
            else:
                print("count group num")
                test_pass_solution_num[t] += 1
    if sort_len:
        sorted_pass_test_solution_num = sorted(test_pass_solution_num.items(),key=lambda x: (x[1],-len(testcases[x[0]])),reverse=True)#(x[1],-len(testcases[x[0]]))
    else:
        sorted_pass_test_solution_num = sorted(test_pass_solution_num.items(),key=lambda x: x[1],reverse=True)#(x[1],-len(testcases[x[0]]))
    chosen_testcase = [x[0] for x in sorted_pass_test_solution_num[:chosen_num]]
    
    return sorted_group,chosen_testcase

def get_CODET_point_origin(Node_list, testcases, task_id) -> None:
    start_time = time.time()
    solution_id_to_data = dict()
    test_id_to_data = dict()
    solution_pass_test = defaultdict(set)
    sol_ids = []
    test_ids = []
    solution_group = defaultdict(set)
    # group_score = defaultdict(int)
    grouped = dict()
    verbose = True
    
    for i,node in enumerate(Node_list):
        solution_id_to_data[i]=node.solution
        sol_ids.append(i)
        grouped[i]=False
    for i,test in enumerate(testcases):
        test_id_to_data[i]=test
        test_ids.append(i)
    log_message("Run solution and test case...",verbose)    
    for i in sol_ids:
        if Node_list[i].already_CODET:
            solution_pass_test[i] = solution_pass_test[i] | Node_list[i].CODET_pass_testcase
            continue
        for j in test_ids:
            passed = exec_solution_testcase(task_id,solution_id_to_data[i],test_id_to_data[j])
            if passed:
                solution_pass_test[i].add(j)
        Node_list[i].CODET_pass_testcase = solution_pass_test[i]
        Node_list[i].already_CODET = True
    log_message("Group solution...",verbose)            
    for idx,i in enumerate(sol_ids):#range(len(sol_ids)):
        if grouped[i]:
            continue
        group = set()
        group.add(i)
        grouped[i] = True
        for j in range(idx+1,len(sol_ids)):
            solution_id_2 = sol_ids[j]
            if solution_pass_test[i] == solution_pass_test[solution_id_2]:
                group.add(solution_id_2)
                grouped[solution_id_2]=True
        solution_group[i] = group
        group_score = math.sqrt(len(group)) * len(solution_pass_test[i])
        log_message(f"group {i} : {group} scores {group_score}",verbose)
        for sol_id in group:
            Node_list[sol_id].CODET_point = group_score
    end_time = time.time()
    log_message(f"Spends {(end_time-start_time)/60} mins",verbose)
    return


def start_code_extract(tprompt,entry_point):
    # 去掉problems[tid]["prompt"]中的注释和空行
    start_code = ""
    for line in tprompt.split("\n"):
        if line=="":
            continue
        if line.startswith("#"):
            continue
        if entry_point in line:
            start_code += line + "\n"
            break
        start_code += line + "\n"
    print("=====start code===========")
    print(start_code)
    return start_code