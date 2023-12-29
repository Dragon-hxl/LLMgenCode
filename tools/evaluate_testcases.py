# import multiprocessing as mp
from pathos.helpers import mp as multip
import time
import sys
sys.path.append("/home/S/hexiaolong/codex/self-debug")
sys.path.append("/home/S/hexiaolong/codex/self-debug/humaneval")
from testcase_study import *
from human_eval.data import  read_problems
from human_eval.execution import check_test_correctness,run_code_with_output_CODET
from data_tools import Counter_with_base,data_analysis

def worker(args):
    problem,testcases = args
    print(f"start evaluate testcase for task : {problem['task_id']}.")
    tin_set = set()
    correct_testcases = []
    entry_point = problem["entry_point"] + "("
    for testcase in testcases:
        if entry_point not in testcase or "==" not in testcase:
            continue
        check_program = problem["prompt"] + problem["canonical_solution"] + testcase
        # print(check_program)
        result = check_test_correctness(check_program,1.0)
        passed = result["passed"]
        tin = testcase.split("==")[0].strip()
        if passed and tin not in tin_set:
            correct_testcases.append(testcase)
            tin_set.add(tin)
    print(f"evaluat testcase for task : {problem['task_id']} end!")
    return (problem["task_id"],correct_testcases)
        
    


def main(res_file):
    testcases_gened = load_testcase("../try/gen_test_t0.8_topp0.95_sample100_max300_rm.jsonl")
    problems = read_problems()
    task_ids = list(problems.keys())
    # init job args
    jobs = []
    for tid in task_ids:
        jobs.append((problems[tid],testcases_gened[tid]))
    # init pool
    worker_num = 64
    pools = multip.Pool(worker_num)
    # run
    result = []
    for i in range(len(jobs)):
        res = pools.apply_async(worker,args=(jobs[i],))
        result.append(res.get())
    pools.close()
    pools.join()
    # handle result
    print("write result")
    if res_file!= "":
        f = open(res_file,"w+")
    percents = []
    for tid,correct_testcase in result:
        total_num = len(testcases_gened[tid])
        correct_num = len(correct_testcase)
        correct_percent = correct_num/total_num
        print(f"++++++++++{tid}++++++++++")
        print(f"total testcase num : {total_num}")
        print(f"correct testcase num : {correct_num}")
        print(f"correct percent {correct_percent}")
        percents.append(correct_percent)
        # correct_testcase = list(correct_testcase)
        if res_file!= "":
            f.write(json.dumps({"task_id":tid,"testcases":correct_testcase,"total_num":total_num,"correct_num":correct_num,"correct_percent":correct_percent})+"\n")
    data_analysis(percents)
    Counter_with_base(percents,0.01)
    return

if __name__=="__main__":
    main("../try/gen_test_t0.8_topp0.95_sample100_max300_rm_correct.jsonl")