import torch
import json
import argparse
import re
import time
import math
import sys
sys.path.append("/home/S/hexiaolong/codex/human-eval")
sys.path.append("/home/S/hexiaolong/codex/self-debug")
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from utils.obj import Node
from human_eval.data import  read_problems
from human_eval.execution import check_test_correctness,run_code_with_output_CODET


# get args for humaneval test on LLM
def get_args():
    parser = argparse.ArgumentParser(description="human_eval simplefeedback")
    parser.add_argument('-mf','--model_path',default='/lustre/S/hexiaolong/vicuna-7b-v1.1',required=True, help="file path where the model used locates")
    parser.add_argument('-o','--output',default="ouput.json",required=True, help="file where the output store")
    parser.add_argument("-v","--verbose", default=False, action="store_true", help="show log")

    args = parser.parse_args()
    
    return args

#为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
def map_gpu_memory(used_gpu):
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
                        i: str(int(gpu_memory[i] * 0.85)) + "GiB"
                        for i in range(num_gpus)
                    }
    # used_gpu = []
    memory_mapping ={}
    if used_gpu!=[]:
        for i in used_gpu:
            memory_mapping[i] = max_memory_mapping[i]
        max_memory_mapping = memory_mapping
    print(max_memory_mapping)
    return max_memory_mapping

def get_unit_test(ut_file, chosen_num=1000,verbose=False):
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
        passn = (1.0*pass_tests)/total_tests
    return prompt,passn
def filter_fix_ans(ans, entry_point, start_code,verbose=False):
    # print("=================fix ans=====================")
    # print(ans)
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


def get_CODET_point(Node_list, testcases, task_id) -> None:
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

def get_CODET_point2(Node_list, testcases, task_id) -> None:
    start_time = time.time()
    solution_id_to_data = dict()
    test_id_to_data = dict()
    solution_pass_test = defaultdict(set)
    sol_ids = []
    test_ids = []
    solution_group = defaultdict(set)
    group_score = defaultdict(int)
    grouped = dict()
    verbose = True
    
    for i,node in enumerate(Node_list):
        solution_id_to_data[i]=node
        sol_ids.append(i)
        grouped[i]=False
    for i,test in enumerate(testcases):
        test_id_to_data[i]=test
        test_ids.append(i)
    log_message("Run solution and test case...",verbose)
    # print_checkp(problems[task_id],testcases)    
    for i in sol_ids:
        if Node_list[i].already_CODET:
            solution_pass_test[i] =  Node_list[i].CODET_pass_testcase
            continue
        with ThreadPoolExecutor(max_workers=1) as executor:
            args = (problems[task_id],solution_id_to_data[i].solution,testcases,"",0.1)
            future = executor.submit(run_code_with_output_CODET, *args)
            result = future.result()
            passed = result["passed"]
            code_res = result["result"]
            checkp = result["check_program"]
        print(f"task:{task_id},solution:{i},passed:{passed},result:{code_res}")
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
        group_score[i] = math.sqrt(len(group)) * len(solution_pass_test[i])
        log_message(f"group {i} : {group} scores {group_score[i]}",verbose)
        for sol_id in group:
            Node_list[sol_id].CODET_point = group_score[i]
    end_time = time.time()
    log_message(f"Spends {(end_time-start_time)/60} mins",verbose)
    return group_score


def get_CODET_point3(Node_list, testcases, task_id) -> None:
    start_time = time.time()
    solution_id_to_data = dict()
    test_id_to_data = dict()
    solution_pass_test = defaultdict(set)
    sol_ids = []
    test_ids = []
    solution_group = defaultdict(set)
    group_score = defaultdict(int)
    grouped = dict()
    verbose = True
    
    for i,node in enumerate(Node_list):
        solution_id_to_data[i]=node
        sol_ids.append(i)
        grouped[i]=False
    for i,test in enumerate(testcases):
        test_id_to_data[i]=test
        test_ids.append(i)
    log_message("Run solution and test case...",verbose)
    print_checkp(problems[task_id],testcases)    
    for i in sol_ids:
        if Node_list[i].already_CODET:
            solution_pass_test[i] =  Node_list[i].CODET_pass_testcase
            continue
        # for j in test_ids:
        #     passed = exec_solution_testcase(task_id,solution_id_to_data[i],test_id_to_data[j])
        #     if passed:
        #         solution_pass_test[i].add(j)
        # run_code_with_output_CODET(problems[task_id],solution_id_to_data[i],testcases,"",300.0)
        with ThreadPoolExecutor(max_workers=1) as executor:
            args = (problems[task_id],solution_id_to_data[i].solution,testcases,"",30.0)
            future = executor.submit(run_code_with_output_CODET, *args)
            result = future.result()
            passed = result["passed"]
            code_res = result["result"]
        print(f"task:{task_id},solution:{i},passed:{passed},result:{code_res}")
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
        group_score[i] = math.sqrt(len(group)) * len(solution_pass_test[i])
        log_message(f"group {i} : {group} scores {group_score[i]}",verbose)
        for sol_id in group:
            Node_list[sol_id].CODET_point = group_score[i]
    log_message("Sort group and get result...",verbose)    
    sorted_group = sorted(group_score.items(),key=lambda x: x[1],reverse=True)
    sorted_nodes = []
    for k,v in sorted_group:
        sgroup = solution_group[k]
        nodes = [solution_id_to_data[i] for i in sgroup]
        nodes = sorted(nodes,key=lambda x: (x.passT_rate,x.prob),reverse=True)
        sorted_nodes.append(nodes)
    idx_record = []
    for nodes in sorted_nodes:
        idx_record.append(0)
    chosen_nodes = []
    lack_num = 0
    stop = False
    while True:
        for i,nodes in enumerate(sorted_nodes):
            if idx_record[i] >= len(nodes):
                lack_num+=1
                if lack_num > 999:
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
    end_time = time.time()
    log_message(f"Spends {(end_time-start_time)/60} mins",verbose)
    return chosen_nodes


def test_filter(test_file):
    resfile = test_file[:-6]+"_filter.jsonl"
    testcases = {}
    with open(tests_for_CODET_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            for k,v in data.items():
                task_test_cases = sum(v, [])
                print(f"task {k} gen {len(task_test_cases)} testcases")
                testcases[k] = task_test_cases
    filter_tests = {}
    for k,v in testcases.items():
        task_id = k
        with ThreadPoolExecutor(max_workers=1) as executor:
            args = (problems[task_id],problems[task_id]["canonical_solution"],v,"",0.1)
            future = executor.submit(run_code_with_output_CODET, *args)
            result = future.result()
            passed = result["passed"]
            code_res = result["result"]
            checkp = result["check_program"]
        filter_testcases = []
        for i,test in enumerate(v):
            if code_res[i] == True:
                filter_testcases.append(test)
        filter_tests[task_id] = filter_testcases
        print(f"For task {task_id}, there are {len(filter_testcases)} testcases left after filter!")
        with open(resfile,"w+") as f:
            f.write(json.dumps(filter_tests))
    return
    
if __name__=="__main__":
    tests_for_CODET_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300.jsonl"
    test_filter(test_file=tests_for_CODET_file)
    