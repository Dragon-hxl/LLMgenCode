"""
CodeT论文中dual execution agreement部分的实现.
"""

import json
import argparse
import time
import random
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from human_eval.data import  read_problems
from human_eval.execution import check_test_correctness
from pathos.helpers import mp as multip


problems = read_problems()
random.seed(1024)

def load_tests(file):
    """
    加载模型生成的test cases
    """
    testcases = {}
    with open(file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            for k,v in data.items():
                tid = k
                test = v
                testcases[k] = v
    return testcases

def load_solutions(file):
    """
    加载模型生成的solutions
    """
    solutions = {}
    with open(file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["task_id"]
            sols = data["completion"]
            solutions[tid] = sols
    return solutions

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

def dual_execution_agreement(task_id,tests,solutions,verbose=False):
    log_message(f"Start dual_execution_agreement for {task_id}",verbose)
    start_time = time.time()
    solution_id_to_data = dict()
    test_id_to_data = dict()
    solution_pass_test = defaultdict(set)
    sol_ids = []
    test_ids = []
    solution_group = defaultdict(set)
    group_score = defaultdict(int)
    grouped = dict()
    
    for i,solution in enumerate(solutions):
        solution_id_to_data[i]=solution
        sol_ids.append(i)
        grouped[i]=False
    for i,test in enumerate(tests):
        test_id_to_data[i]=test
        test_ids.append(i)
    log_message("Run solution and test case...",verbose)    
    for i in sol_ids:
        for j in test_ids:
            passed = exec_solution_testcase(task_id,solution_id_to_data[i],test_id_to_data[j])
            if passed:
                solution_pass_test[i].add(j)
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
        group_score[i] = len(group) * len(solution_pass_test[i])
        log_message(f"group {i} scores {group_score[i]}",verbose)
    log_message("Sort group and get result...",verbose)    
    sorted_group = sorted(group_score.items(),key=lambda x: x[1],reverse=True)
    for k,v in sorted_group:
        result_test_id = solution_pass_test[k]
        result_test = [test_id_to_data[i] for i in result_test_id]
        break
    log_message(f"After dual_execution_agreement, chosen {len(result_test)} testcases",verbose)
    
    end_time = time.time()
    log_message(f"Spends {(end_time-start_time)/60} mins",verbose)
    return result_test

def show_data(data,kind):
    if kind == 'solution':
        for k,v in data.items():
            print(f"solutions for task {k}, total num is {len(v)}")
            for i,solution in enumerate(v):
                print(f"[solution [{i}]]")
                print(solution)
            print("----------------------------------")
    elif kind == 'testcase':
        for k,v in data.items():
            print(f"testcase for task {k} total num is {len(v)}")
            for i,testcase in enumerate(v):
                test = f"assert {testcase['tin']} == {testcase['tout']}"
                print(test)
            print("----------------------------------")
                

def log_message(message,verbose):
    if verbose:
        print(message)

def init_worker(*args):# 初始化，这里通常是定义全局变量，且在worker中不能修改这些变量
    # global finished, left
    # finished, left = args
    pass

def worker(args):
    task_id,tests,solutions,verbose = args
    testcases = dual_execution_agreement(task_id,tests,solutions,verbose)
    return {"task_id":task_id,"testcase":testcases}

class Config():
    def __init__(self):
        self.tests_file = "gen_tests_13bv1_num300.jsonl"
        self.solutions_file = "gen_solutions_13bv1_num100.jsonl"
        self.output_file = "test_from_codeT_13bv1_t300_s100.jsonl"
        self.test_num = 300
        self.solution_num = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="show detail")
    parser.add_argument("-show", "--show", action="store_true", help="when set, will show solutions and testcases in detail.")
    args = parser.parse_args()
    
    verbose = args.verbose
    show = args.show
    multi_core = False
    multi_core2 = True
    
    config = Config()
    output_file = config.output_file
    #读取数据：humaneval数据集，生成的solution，生成的testcases
    log_message(f"Load solutions and testcases from {config.solutions_file} and {config.tests_file}",verbose)
    solutions_gened = load_solutions(config.solutions_file)
    testcases_gened = load_tests(config.tests_file)
    f = open(output_file,"w+",encoding='utf-8')
    final_testcases = dict()
    if show:
        show_data(solutions_gened, 'solution')
        show_data(testcases_gened, 'testcase')
    elif multi_core:
        num_workers = 64
        task_ids = list(problems.keys())
        num_jobs = len(task_ids)
        # 开始dual_execution_agreement过程
        rs = []
        finished = 0
        while num_jobs:
            if num_jobs > num_workers:
                jobs = [] # 列表的形式，其中的每个元素就是一个进程的参数
                for i in range(finished,finished + num_workers):
                    tid = task_ids[i]
                    tlen = len(testcases_gened[tid])
                    slen = len(solutions_gened[tid])
                    if tlen >= config.test_num:
                        chosen_testcases = random.sample(testcases_gened[tid],k=config.test_num)
                    else:
                        chosen_testcases = testcases_gened[tid]
                    if slen >= config.solution_num:
                        chosen_solution = random.sample(solutions_gened[tid], k=config.solution_num)
                    else:
                        chosen_solution = solutions_gened[tid]
                    args = (tid,chosen_testcases,chosen_solution,verbose)
                    jobs.append(args)
                pools = multip.Pool(processes=num_workers)# ,initializer=init_worker,initargs=None
                res = pools.map(worker, jobs)
                print("res :",res)
                rs = rs + res
                num_jobs -= num_workers
                finished += num_workers
                pools.close() #关闭进程池
                pools.join() #等待所有进程结束
            else:
                for i in range(finished,finished + num_jobs):
                    tid = task_ids[i]
                    tlen = len(testcases_gened[tid])
                    slen = len(solutions_gened[tid])
                    if tlen >= config.test_num:
                        chosen_testcases = random.sample(testcases_gened[tid],k=config.test_num)
                    else:
                        chosen_testcases = testcases_gened[tid]
                    if slen >= config.solution_num:
                        chosen_solution = random.sample(solutions_gened[tid], k=config.solution_num)
                    else:
                        chosen_solution = solutions_gened[tid]
                    args = (tid,chosen_testcases,chosen_solution,verbose)
                    jobs.append(args)
                pools = multip.Pool(processes=num_jobs)#,initializer=init_worker,initargs=None
                res = pools.map(worker, jobs)
                print("res :",res)
                rs = rs + res
                num_jobs -= num_jobs
                finished += num_jobs
                pools.close()
                pools.join()
        final_testcases = dict()
        for result in rs:
            final_testcases[result["task_id"]] = result["testcase"]
            f.write(json.dumps({"task_id":result["task_id"],"testcases":result["testcase"]})+"\n")
    elif multi_core2:
        task_ids = list(problems.keys())
        num_tasks = len(task_ids)
        num_processes = 64
        jobs = [] # 列表的形式，其中的每个元素就是一个进程的参数
        for i in range(num_tasks):
            tid = task_ids[i]
            tlen = len(testcases_gened[tid])
            slen = len(solutions_gened[tid])
            if tlen >= config.test_num:
                chosen_testcases = random.sample(testcases_gened[tid],k=config.test_num)
            else:
                chosen_testcases = testcases_gened[tid]
            if slen >= config.solution_num:
                chosen_solution = random.sample(solutions_gened[tid], k=config.solution_num)
            else:
                chosen_solution = solutions_gened[tid]
            args = (tid,chosen_testcases,chosen_solution,verbose)
            jobs.append(args)
        pool = multip.Pool(processes=num_processes)
        results = pool.map(worker,jobs)
        pool.close() #关闭进程池
        pool.join() #等待所有进程结束
        final_testcases = dict()
        for result in results:
            final_testcases[result["task_id"]] = result["testcase"]
            f.write(json.dumps({"task_id":result["task_id"],"testcases":result["testcase"]})+"\n")
    else:
        task_ids = list(problems.keys())
        # 开始dual_execution_agreement过程
        for tid in task_ids:
            tlen = len(testcases_gened[tid])
            slen = len(solutions_gened[tid])
            if tlen >= config.test_num:
                chosen_testcases = random.sample(testcases_gened[tid],k=config.test_num)
            else:
                chosen_testcases = testcases_gened[tid]
            if slen >= config.solution_num:
                chosen_solution = random.sample(solutions_gened[tid], k=config.solution_num)
            else:
                chosen_solution = solutions_gened[tid]
            testcases = dual_execution_agreement(tid,chosen_testcases,chosen_solution,verbose)
            final_testcases[tid] = testcases
            for tid in task_ids:
                f.write(json.dumps(final_testcases[tid])+"\n")
    f.close()
    print("dual_execution_agreement finished!")
    
        
        