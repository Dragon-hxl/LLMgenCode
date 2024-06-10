from dataset import read_problems
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append("/home/S/hexiaolong/codex/self-debug/humaneval")
from executor_utils import check_correctness,check_test_correctness
import json
import numpy as np
from myutils import make_printv,print_with_tag
from resfiles_record import res_root,data_files,res_7b16k,res_cola7bpy,res_cola34bpy,res_llama7b,tmp

def get_truePass(problem,solution):
    check_program = (
                problem["prompt"] +"\n" + solution + "\n" +
                problem["test"] + "\n" +
                f"check({problem['entry_point']})"
            )
    # print(f"check_program: \n{check_program}")
    with ThreadPoolExecutor(max_workers=1) as executor:
        # args = (problem, solution, 1.0)
        # future = executor.submit(check_correctness, *args)
        args = (check_program,1.0)
        future = executor.submit(check_test_correctness, *args)
        result = future.result()
        passed = result["passed"]
        error_message = result["result"]
    return passed,error_message

def tid_to_int(tid):
    return int(tid.split("/")[1])

def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def error_analysis(error_dict):
    error_list = []
    for cir,errors in error_dict.items():
        print(f"In cir {cir} , errors counter is:")
        print(Counter(errors))
        error_list += errors
    print("Total error messages Counter is : ")
    print(Counter(error_list))
    return

def get_pass_k(results,data,k=10,n=10,ignore_task=[],verbose=False):
    print_v = make_printv(verbose=verbose)
    passed_per_cir = defaultdict(set)# 每个cir通过的task
    task_cir = defaultdict(list)
    task_cir_pass = defaultdict(list)
    checked_task = set()
    pass_k_list = defaultdict(list)
    result_num = len(results)
    error_dict = defaultdict(list)
    cir_nums = []
    for result in results:
        task_id = tid_to_int(result["task_id"])
        if task_id in checked_task:
            continue
        checked_task.add(task_id)
        completion = result["completion"]
        problem = data[result["task_id"]]
        cirs = []
        for cir,solutions in completion.items():
            cir = int(cir)
            cirs.append(cir)
            print(f"Task {task_id} gens {len(solutions)} solutions in cir {cir}")
            if cir==5 and task_id==100:
                for i,solution in enumerate(solutions):
                    print(f"In cir 5 ,solution {i} is :\n{solution}")
            total_passed = False
            solutions = solutions[:k]
            passed_num = 0
            
            for i,solution in enumerate(solutions):
                solution = solution["solution"]
                passed,error_message = get_truePass(problem,solution)
                print_v(f"solution {i} passed {passed}")
                
                if passed:
                    passed_num += 1
                    total_passed =True
                    for c in range(cir,11):
                        passed_per_cir[c].add(task_id)
                    cir_nums.append(cir)
                else:
                    error_dict[cir].append(error_message)
            pass_k_list[cir].append(estimator(n,passed_num,k))
            task_cir_pass[task_id].append(total_passed)
        cirs = sorted(cirs)
        task_cir[task_id] = cirs
    task_cir_pass = dict(sorted(task_cir_pass.items(),key=lambda x:x[0]))
    passed_per_cir = dict(sorted(passed_per_cir.items(),key=lambda x:x[0]))
    # find lack task
    task_has_solution = task_cir_pass.keys()
    # print(f"task_has_solution: {task_has_solution}")
    tid_list = data.keys()
    tid_list = [tid_to_int(tid) for tid in tid_list]
    tid_list = [x for x in tid_list if x not in ignore_task]
    lack_task = [tid for tid in tid_list if tid not in task_has_solution]
    # for tid in tid_list:
    #     if tid not in task_has_solution:
    #         lack_task.append(tid)
    
    pass_task_num = [0 for i in range(11)]
    for k,v in passed_per_cir.items():
        print(f"cir {k},passed {len(v)} tasks, pass rate is {len(v)/result_num}")
        print(f"pass tasks are:\n{sorted(v)}")
        pass_task_num[k] = len(v)
    
    for k,v in task_cir_pass.items():
        print(f"task {k} pass or not for each cir: {v}")
    print("--------------------------------------------")
    
    print(f"lack task : {lack_task}")
    
    
    print(f"pass task num: {pass_task_num}")
    # pass_task_rate = [x/result_num for x in pass_task_num]
    # print(f"pass task rate: {pass_task_rate}")
    
    #计算无差的pass@k
    # for cir,v in pass_k_list.items():
    #     pass_at_k = sum(v)/len(v)
    #     print(f"cir {cir}, pass at k {k} rate: {pass_at_k}")
    #错误信息分析
    # error_analysis(error_dict)
    return passed_per_cir,task_cir,lack_task,pass_k_list

def evaluate_gened_testcase(results,data,verbose):
    printv = make_printv(verbose)
    pass_rate_list = []
    for result in results:
        tid = result["task_id"]
        is_MBPP = False
        if "MBPP" in tid:
            is_MBPP = True
        gened_testcase = result["gened_testcase"]
        n = len(gened_testcase)
        if is_MBPP:
            solution = data[tid]["canonical_solution"]
        else:
            solution = data[tid]["prompt"] + data[tid]["canonical_solution"]
        pass_test_num = 0
        for testcase in gened_testcase:
            check_program = solution + "\n" + testcase
            with ThreadPoolExecutor(max_workers=1) as executor:
                args = (check_program, 1.0)
                future = executor.submit(check_test_correctness, *args)
                result = future.result()
                passed = result["passed"]
                if passed:
                    pass_test_num += 1
        if n == 0:
            pass_rate = 0
        else:
            pass_rate = pass_test_num/n
        printv(f"Task {tid} gened {n} testcases, {pass_test_num} of them is correct, pass rate is {pass_rate}")
        pass_rate_list.append(pass_rate)
    mean_pass_rate = np.mean(pass_rate_list)
    max_pass_rate = np.max(pass_rate_list)
    min_pass_rate = np.min(pass_rate_list)
    all_true_num = sum([1 for x in pass_rate_list if x == 1.0])
    printv(f"mean pass rate is {mean_pass_rate}, max pass rate is {max_pass_rate}, min pass rate is {min_pass_rate}, all true is {all_true_num}")
    return pass_rate_list
def evaluate_testcase_nofilter(results,data,verbose):
    printv = make_printv(verbose)
    pass_rate_list = []
    for result in results:
        tid = result["task_id"]
        is_MBPP = False
        if "MBPP" in tid:
            is_MBPP = True
        gened_testcase = result["gened_testcase"]
        np.random.seed(2024)
        # if len(gened_testcase) > 10:
        #     gened_testcase = np.random.choice(gened_testcase,10,replace=False)
        gened_testcase = gened_testcase[:10]
        n = len(gened_testcase)
        if is_MBPP:
            solution = data[tid]["canonical_solution"]
        else:
            solution = data[tid]["prompt"] + data[tid]["canonical_solution"]
        pass_test_num = 0
        for testcase in gened_testcase:
            check_program = solution + "\n" + testcase
            with ThreadPoolExecutor(max_workers=1) as executor:
                args = (check_program, 1.0)
                future = executor.submit(check_test_correctness, *args)
                result = future.result()
                passed = result["passed"]
                if passed:
                    pass_test_num += 1
        if n==0:
            pass_rate = 0
        else:
            pass_rate = pass_test_num/n
        printv(f"Task {tid} gened {n} testcases, {pass_test_num} of them is correct, pass rate is {pass_rate}")
        pass_rate_list.append(pass_rate)
    mean_pass_rate = np.mean(pass_rate_list)
    max_pass_rate = np.max(pass_rate_list)
    min_pass_rate = np.min(pass_rate_list)
    all_true_num = sum([1 for x in pass_rate_list if x == 1.0])
    printv(f"mean pass rate is {mean_pass_rate}, max pass rate is {max_pass_rate}, min pass rate is {min_pass_rate}, all true is {all_true_num}")
    return pass_rate_list

def evaluate_testcase_filter(results,data,verbose):
    printv = make_printv(verbose)
    pass_rate_dict = defaultdict(list)
    for result in results:
        tid = result["task_id"]
        is_MBPP = False
        if "MBPP" in tid:
            is_MBPP = True
        chosen_testcase_dict = result["chosen_testcase_dict"]
        if is_MBPP:
            solution = data[tid]["canonical_solution"]
        else:
            solution = data[tid]["prompt"] +"\n"+ data[tid]["canonical_solution"]
        for cir,internal_tests in chosen_testcase_dict.items():
            n = len(internal_tests)
            pass_test_num = 0
            for testcase in internal_tests:
                check_program = solution + "\n" + testcase
                with ThreadPoolExecutor(max_workers=1) as executor:
                    args = (check_program, 1.0)
                    future = executor.submit(check_test_correctness, *args)
                    result = future.result()
                    passed = result["passed"]
                    if passed:
                        pass_test_num += 1
            if n == 0:
                pass_rate = 0
            else:
                pass_rate = pass_test_num/n
            pass_rate_dict[cir].append(pass_rate)
    for cir,pass_rate_list in pass_rate_dict.items():
        mean_pass_rate = np.mean(pass_rate_list)
        max_pass_rate = np.max(pass_rate_list)
        min_pass_rate = np.min(pass_rate_list)
        all_true_num = sum([1 for x in pass_rate_list if x == 1.0])
        printv(f" cir {cir} mean pass rate is {mean_pass_rate}, max pass rate is {max_pass_rate}, min pass rate is {min_pass_rate}, all true is {all_true_num}")
    return pass_rate_dict

def evaluate_internal_tests(results,data,verbose):
    printv = make_printv(verbose)
    pass_rate_list = []
    for result in results:
        tid = result["task_id"]
        is_MBPP = False
        if "MBPP" in tid:
            is_MBPP = True
        internal_tests = result["internal_tests"]
        n = len(internal_tests)
        if n==0:
            print(f"Task {tid} has no internal tests")
            continue
        if is_MBPP:
            solution = data[tid]["canonical_solution"]
        else:
            solution = data[tid]["prompt"] +"\n"+ data[tid]["canonical_solution"]
        pass_test_num = 0
        for testcase in internal_tests:
            check_program = solution + "\n" + testcase
            with ThreadPoolExecutor(max_workers=1) as executor:
                args = (check_program, 1.0)
                future = executor.submit(check_test_correctness, *args)
                result = future.result()
                passed = result["passed"]
                if passed:
                    pass_test_num += 1
        if n == 0:
            pass_rate = 0
        else:
            pass_rate = pass_test_num/n
        printv(f"Task {tid} use {n} internal_tests, {pass_test_num} of them is correct, pass rate is {pass_rate}")
        pass_rate_list.append(pass_rate)
    mean_pass_rate = np.mean(pass_rate_list)
    max_pass_rate = np.max(pass_rate_list)
    min_pass_rate = np.min(pass_rate_list)
    all_true_num = sum([1 for x in pass_rate_list if x == 1.0])
    printv(f"mean pass rate is {mean_pass_rate}, max pass rate is {max_pass_rate}, min pass rate is {min_pass_rate}, all true is {all_true_num}")
    return pass_rate_list

def load_results(res_file):
    results = []
    with open(res_file,"r") as f:
        for line in f.readlines():
            result = json.loads(line)
            # print(result["task_id"])
            results.append(result)
    return results

def show_certian_task(results,tid):
    print("Show debug progress of task {}".format(tid))
    for result in results:
        task_id = tid_to_int(result["task_id"])
        if task_id != tid:
            continue
        completion = result["completion"]
        problem = data[result["task_id"]]
        print(f"Task NL: \n{problem['prompt']}")
        for cir,solutions in completion.items():
            cir = int(cir)
            solution = solutions[0]
            print("-----------------------------")
            print(f"Cir {cir} :\n {solution['solution']}\npassT_rate = {solution['passT_rate']}")
            print("-----------------------------")
    return

chosen_data_idx = [240, 93, 372, 296, 155, 102, 454, 370, 209, 387, 366, 388, 135, 272, 125, 325, 416, 376, 255, 181, 212, 269, 497, 315, 111, 158, 278, 360, 169, 265, 38, 374, 396, 443, 105, 352, 385, 477, 239, 363, 425, 446, 334, 75, 486, 108, 444, 210, 29, 394, 178, 321, 213, 238, 63, 371, 380, 71, 390, 167, 199, 471, 176, 406, 494, 166, 218, 479, 162, 290, 109, 208, 117, 104, 20, 383, 115, 441, 9, 132, 258, 163, 395, 291, 411, 361, 215, 314, 57, 438, 457, 310, 399, 118, 120, 237, 187, 69, 103, 188, 252, 304, 448, 72, 134, 198, 319, 172, 171, 362, 364, 458, 86, 350, 356, 67, 410, 465, 297, 351, 33, 50, 88, 2, 77, 224, 472, 405, 179, 427, 41, 100, 145, 122, 355, 236, 308, 417, 246, 268, 223, 339, 432, 435, 36, 154, 354, 142, 402, 289, 338, 128, 478, 51, 253, 475, 368, 450, 90, 263, 114, 418, 480, 23, 496, 473, 193, 324, 37, 60, 492, 28, 470, 64, 107, 412, 44, 419, 377, 462, 249, 298, 84, 82, 323, 326, 53, 398, 287, 309, 15, 312, 55, 286, 92, 409, 161, 0, 62, 143]
base_pass_task_mbpp = [17, 23, 27, 35, 40, 41, 46, 51, 52, 58, 62, 66, 79, 82, 85, 88, 89, 93, 96, 99, 105, 112, 113, 127, 133, 144, 145, 161, 168, 171, 173, 174, 175, 176, 183, 195, 204, 210, 212, 214, 221, 222, 227, 230, 234, 249, 250, 255, 258, 261, 263, 269, 273, 281, 284, 293, 297, 309, 319, 322, 329, 332, 333, 341, 361, 373, 375, 394, 403, 404, 412, 422, 425, 443, 447, 457, 458, 459, 465, 474, 476, 478, 480, 487, 489, 495, 496, 498, 502, 504, 507, 509]
different_task = ['MBPP/18', 'MBPP/30', 'MBPP/45', 'MBPP/56', 'MBPP/70', 'MBPP/148', 'MBPP/151', 'MBPP/152', 'MBPP/164', 'MBPP/181', 'MBPP/323', 'MBPP/338', 'MBPP/342', 'MBPP/348', 'MBPP/364', 'MBPP/367', 'MBPP/466', 'MBPP/485', 'MBPP/486', 'MBPP/501']
humaneval_cola7bpy_base = [0, 2, 3, 4, 5, 7, 9, 12, 13, 15, 16, 17, 21, 22, 23, 24, 27, 28, 29, 30, 31, 34, 35, 42, 43, 44, 45, 46, 47, 48, 49, 53, 55, 56, 58, 60, 61, 63, 66, 71, 72, 74, 77, 79, 80, 86, 87, 104, 107, 112, 113, 116, 121, 122, 124, 136, 142, 147, 152, 153, 156, 159, 162]


if __name__ == "__main__":    
    res_file = res_root + tmp[-1] #res_cola7bpy[3]#res_7b16k[1]#res_cola34bpy[5]# res_cola7bpy[3]
    
    if "mbpp" in res_file:
        data_file = data_files["mbpp"]
        data = read_problems(data_file)
        new_data = {}
        for tid,d in data.items():
            tid_num = tid_to_int(tid)
            idx = tid_num - 11
            if idx  not in chosen_data_idx:
                continue
            new_data[tid] = d
        data = new_data
    elif "humaneval" in res_file:
        data_file = data_files["humaneval"]
        data = read_problems(data_file)
    elif "mtpb" in res_file:
        data_file = data_files["mbpt"]
        data = read_problems(data_file)
    elif "bigbench" in res_file:
        data_file = data_files["bigbench"]
        data = read_problems(data_file)
        
    results = load_results(res_file=res_file)
    #get pass@k
    get_pass_k(results,data,1,10,ignore_task=[])#humaneval_cola7bpy_base
    # show_certian_task(results,40)
    # evaluate_gened_testcase(results=results,data=data,verbose=True)
    if "TFTS" in res_file:
        # print("------------------------no filter------------------------")
        # evaluate_testcase_nofilter(results=results,data=data,verbose=True)
        print("-----------------------filter-------------------------")
        evaluate_testcase_filter(results=results,data=data,verbose=True)
        # evaluate_internal_tests(results=results,data=data,verbose=True)