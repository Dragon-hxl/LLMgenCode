import json
import math
import random
import numpy as np
import sys
# sys.path.append("/home/S/hexiaolong/codex/human-eval")
sys.path.append("/home/S/hexiaolong/codex/self-debug/humaneval")
sys.path.append("/home/S/hexiaolong/codex/self-debug")
sys.path.append("/home/S/hexiaolong/codex/self-debug/src")
from human_eval.data import read_problems,write_jsonl
from human_eval.execution import run_code_with_output2, check_correctness,check_test_correctness,run_code_with_output_CODET,_pack_test_cases
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
from data_tools import data_analysis,Counter_with_base
from myutils import get_unit_test
from collections import defaultdict
from matplotlib import colormaps
import re
# problems
problems = read_problems()

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
                    t = "assert " + tin + " == " + tout
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
    elif type == 3:
        # {"task_id":[{"tin":tin,"tout":tout}]} each line (uniform format)
         with open(test_file,"r") as f:
            for line in f.readlines():
                data = json.loads(line)
                for tid,ios in data.items():
                    uts = []
                    for io in ios:
                        tin = io["tin"]
                        tout = io["tout"]
                        t = "assert " + tin + " == " + tout
                        uts.append(t)
                    testcase[tid] = uts
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

def testfile_trans(test_file,resfile):
    tests = []
    data = load_testcase(test_file)
    for tid,v in data.items():
        ios = []
        for test in v:
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
      
def testcase_num(testcases):
    test_num = []
    test_less_10 = []
    test_less_100 = []
    for k,v in testcases.items():
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

def count_and_remove_duplicate(testcases):
    in_percent_list = []
    out_percent_list = []
    unique_percent_list = []
    valid_percent_list = []
    testcases_rmduplicate = {}
    unique_test_num_list = []
    unique_in_num_list = []
    unique_out_num_list = []
    valid_num_list = []
    for tid,testcase in testcases.items():
        out_set = set()
        in_set = set()
        in_out_set = set()
        valid_num = 0
        for t in testcase:
            if "assert"not in t or "==" not in t:
                continue
            valid_num += 1
            t = t.replace("assert","")
            tin = t.split("==")[0].strip()
            tout = t.split("==")[1].strip()
            if tin not in in_set:
                in_set.add(tin)
            if tout not in out_set:
                out_set.add(tout)
            if (tin,tout) not in in_out_set:
                in_out_set.add((tin,tout))
        ios = [ "assert " + tin + " == " + tout for tin,tout in in_out_set]
        testcases_rmduplicate[tid]=ios
        
        # 打印数据
        test_num = len(testcase)
        unique_in_num = len(in_set)
        unique_out_num = len(out_set)
        unique_test_num = len(in_out_set)
        unique_in_percent = unique_in_num/test_num
        unique_out_percent = unique_out_num/test_num
        unique_test_percent = unique_test_num/test_num
        valid_percent = valid_num/test_num
        in_percent_list.append(unique_in_percent)
        out_percent_list.append(unique_out_percent)
        unique_percent_list.append(unique_test_percent)
        valid_percent_list.append(valid_percent)
        unique_test_num_list.append(unique_test_num)
        unique_in_num_list.append(unique_in_num)
        unique_out_num_list.append(unique_out_num)
        valid_num_list.append(valid_num)
        print("---------------------------------------------------------------------------------------------------------------------------------")
        print(f"For task {tid}, there are {test_num}, valid num : {valid_num}.")
        print(f"unique test is {unique_test_num}, unique in : {unique_in_num}, unique out : {unique_out_num}")
        print(f"unique test percent is {unique_test_percent}, unique in percent : {unique_in_percent}, unique out percent : {unique_out_percent}, valid percent : {valid_percent}")
    # 数据分析和绘图
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("in percent")
    data_analysis(in_percent_list)
    draw_hist(data=in_percent_list,image_path="../image/in_percent_list.png")
    print("out percent")
    data_analysis(out_percent_list)
    draw_hist(data=out_percent_list,image_path="../image/out_percent_list.png")
    print("unique percent")
    data_analysis(unique_percent_list)
    draw_hist(data=unique_percent_list,image_path="../image/unique_test_percent.png")
    print("valid percent")
    draw_hist(data=valid_percent_list,image_path="../image/valid_percent_list.png")
    data_analysis(valid_percent_list)
    print("unique test num")
    data_analysis(unique_test_num_list)
    draw_hist(data=unique_test_num_list,image_path="../image/unique_test_num_list.png")
    print("unique in num")
    data_analysis(unique_in_num_list)
    draw_hist(data=unique_in_num_list,image_path="../image/unique_in_num_list.png")
    print("unique out num")
    data_analysis(unique_out_num_list)
    draw_hist(data=unique_out_num_list,image_path="../image/unique_out_num_list.png")
    print("valid test num")
    data_analysis(valid_num_list)
    draw_hist(data=valid_num_list,image_path="../image/valid_num_list.png")
    return testcases_rmduplicate

def draw_hist(data,image_path):
    import math
    m = max(data)
    range_max = 0
    bin_num = 10
    if m <= 1:
        data_range = (0,1)
        range_max = 1
    else:
        m = round(m)
        n = len(str(m)) - 2
        base = math.pow(10,n)
        if m/base > 30:
            base = base * 10
        bin_num = math.ceil(m/base)
        range_max = bin_num*base
        print(f"data max {m},base {base}, range max : {range_max}")
        data_range = (0,range_max)
    figure = plt.figure(figsize=(9,16),dpi=400)
    plt.xlabel("value")
    plt.ylabel("number")
    colors = plt.get_cmap("Reds")
    n, bins, patches = plt.hist(x=data,bins=bin_num,range=data_range,align="mid",color="Red") # hist里所有的bin默认同颜色
    print(f"bins:\n{bins}")
    # 为bin设置不同的颜色
    for c,p in zip(bins,patches):
        p.set_facecolor(colors(c/range_max))
    for i in range(len(n)):
        plt.text(bins[i]+range_max/(bin_num*2), n[i]*1.02, int(n[i]), fontsize=8, horizontalalignment="center")
    title = image_path.split("/")[-1].split(".")[0]
    plt.title(title)
    xt = [(range_max/bin_num)*i for i in range(bin_num+1)]
    plt.xticks(xt)
    figure.savefig(image_path)


def write_testcase(test_cases,fpath,type:int=0):
    if type==0:
        # test_cases {tid:testcases}
        with open(fpath,"w+") as f:
            for tid,testcase in test_cases.items():
                data = {tid:testcase}
                f.write(json.dumps(data)+"\n")
    elif type==1:
        tests = []
        for tid,v in test_cases.items():
            ios = []
            for test in v:
                if "==" in test:
                    tin = test.split("==")[0].replace("assert","").strip()
                    tout = test.split("==")[1].strip()
                    ios.append({"tin":tin,"tout":tout})
            print(f"task {tid} has {len(ios)} tests")
            tests.append({"task_id":tid,"ios":ios})
        with open(fpath,"w+")as f:
            for t in tests:
                f.write(json.dumps(t)+"\n")
    return
            

def mix_testcase(correct_testcases,all_testcases,correct_percent:float=0.5,keep_num:bool=False):
    """
    Add incorrect testcases to correct testcases.The returned mix_testcases have correct percent correct_testcases.
    if keep_num is True, the returned mix_testcases have the number of testcase as same as the input correct_testcases.
    """
    assert correct_percent > 0.0000001
    
    task_ids = all_testcases.keys()
    special_task = []
    zero_correct_task = []
    mix_testcases = {}
    for tid in task_ids:
        correct_testcase = correct_testcases[tid]
        testcases = all_testcases[tid]
        incorrect_testcase = []
        for t in testcases:
            if t in correct_testcase:
                continue
            else:
                incorrect_testcase.append(t)
        correct_num = len(correct_testcase)
        incorrect_num = len(incorrect_testcase)
        
        if keep_num:
            res_num = correct_num
            add_num = math.ceil(res_num*(1-correct_percent))
            correct_num = res_num - add_num
        else:
            res_num = math.ceil(correct_num/correct_percent)
            add_num = res_num - correct_num
        if correct_num == 0 :
            zero_correct_task.append(tid)
        if add_num > incorrect_num:
            special_task.append(tid)
        
        add_testcase = incorrect_testcase[:add_num]
        correct_testcase = correct_testcase[:correct_num]
        final_add_num = len(add_testcase)
        mix_testcases[tid] = correct_testcase + add_testcase
        print("-----------------------------------------------------------------------------------------------------")
        print(f"In task {tid}, there are {correct_num} corrects and {incorrect_num} incorrects.")
        print(f"When correct percent is {correct_percent}, should add {add_num} testcases, final add {final_add_num}, final testcase num is {len(mix_testcases[tid])}")
    print(f"zeros_correct_task is : {zero_correct_task}, and special task is {special_task}")
        
    return mix_testcases

def check_correct_percent(testcases_correct,test_wait_for_check):
    total_correct_num = 0
    total_num = 0
    cp_list = []
    for tid,testcases in test_wait_for_check.items():
        print("++++++++++++++++++++++++++++++++++++++++++++++")
        testcase_string = "\n".join(testcases)
        print(f"task {tid} has testcase num {len(testcases)}:\n{testcase_string}")
        correct_num = 0
        num = len(testcases)
        ct = testcases_correct[tid]
        for t in testcases:
            if t in ct:
                correct_num += 1
        total_correct_num += correct_num
        total_num += num
        cp = correct_num/num
        cp_list.append(cp)
        print(f"task {tid} correct percent is {cp}")
    correct_percent = total_correct_num/total_num
    print(f"correct percent : {correct_percent}")
    data_analysis(cp_list)
    draw_hist(data=cp_list,image_path="../image/check_correct_percent.png")
    return 

def testcase_clean(testcases):
    
    def remove_common(t):
        tout = t.split("==")[1].strip()
        tin = t.split("==")[0].strip()
        cidx = tout.rfind(", \"")
        if cidx!=-1 and tout[-1]=="\"":
            tout = tout[:cidx+1]
        if tout[-1] == ",":
            tout = tout[:-1]
        return tin + " == " + tout
    
    final_gened_testcases = defaultdict(list)
    special_task = ["HumanEval/81","HumanEval/74","HumanEval/29","HumanEval/125","HumanEval/58","HumanEval/14","HumanEval/7","HumanEval/1","HumanEval/101","HumanEval/105","HumanEval/111","HumanEval/113","HumanEval/117","HumanEval/148","HumanEval/149"]
    final_test_num = []
    usual_task = ["HumanEval/22","HumanEval/57","HumanEval/72","HumanEval/81"]
    cp_list = []
    for tid in testcases.keys():
        entry_point = "assert " + problems[tid]["entry_point"] + "("
        print(f"========================={tid}============================")
        print(f"total testcases : {len(testcases[tid])}")
        invalid_test = []
        tin_set = set()
        t_set = set()
        for t in testcases[tid]:
            t = t.replace("\n","")
            re.sub(r'\s{2,}'," ", t)
            if "assert " not in t or "==" not in t or entry_point not in t:
                invalid_test.append(t)
                continue
            # if tid not in special_task:
            #     t = remove_common(t)
            # else:
            #     print(t)
            t = remove_common(t)
            if tid in special_task:
                print(t)
            tin = t.split("==")[0].replace("assert","").strip()
            # if tin not in tin_set:
            #     tin_set.add(tin)
            #     final_gened_testcases[tid].append(t)
            if t not in t_set:
                t_set.add(t)
                final_gened_testcases[tid].append(t)
        print(f"invalid testcases : {len(invalid_test)}")
        
        correct_t = []
        invalid_test = []
        tsolution = problems[tid]["prompt"]+problems[tid]["canonical_solution"]
        for t in final_gened_testcases[tid]:
            checkp = tsolution + "\n" + t +" \n"
            res = check_test_correctness(checkp,0.1)
            if res["passed"]:
                correct_t.append(t)
            elif res["result"] != "failed: AssertionError" and res["result"] !="timed out":
                # print(f"error test : {t} with result : {res['result']}")
                invalid_test.append(t)
        print(f"testcase tiwh synax error: {len(invalid_test)}")
        
        for t in invalid_test:
            final_gened_testcases[tid].remove(t)
        print(f"final gened testcases : {len(final_gened_testcases[tid])}")
        
        final_test_num.append(len(final_gened_testcases[tid]))
        if len(final_gened_testcases[tid]) == 0:
            print(f"task {tid} has no testcases.")
            continue
        print(f"correct testcases : {len(correct_t)},correct percent : {len(correct_t)/len(final_gened_testcases[tid])}")
        
        correct_t2 = []
        result = run_code_with_output_CODET(problems[tid],problems[tid]["canonical_solution"],final_gened_testcases[tid],"",0.1)
        code_res = result["result"]
        checkp = (
                problems[tid]["prompt"] + problems[tid]["canonical_solution"] + "\n" +
                _pack_test_cases(final_gened_testcases[tid], 0.1)
            )
        if type(code_res) is str:
            print(f"task {tid} has error : {code_res}")
            print(f"check program is \n{checkp}")
            pass
        else:
            for j,res in enumerate(code_res):
                if type(res) is bool:
                    if res:
                        correct_t2.append(final_gened_testcases[tid][j])
                else:
                    print(f"result is not bool : {res}")
        print(f"correct testcases2 : {len(correct_t2)},correct percent : {len(correct_t2)/len(final_gened_testcases[tid])}")
        # assert len(correct_t) == len(correct_t2)
        if(len(correct_t) != len(correct_t2)):
            print(f"task {tid} has different correct testcases.")
        cp_list.append(len(correct_t)/len(final_gened_testcases[tid]))
    data_analysis(final_test_num)
    data_analysis(cp_list)
    return final_gened_testcases


def get_correct_and_wrong_idx(testcase,idx_file):
    f = open(idx_file,"w+")
    for tid,tests in testcase.items():
        print("get idx for task",tid)
        correct_idx = []
        wrong_idx = []
        tsolution = problems[tid]["prompt"]+problems[tid]["canonical_solution"]
        for i,t in enumerate(tests):
            checkp = tsolution + "\n" + t +" \n"
            res = check_test_correctness(checkp,0.1)
            if res["passed"]:
                correct_idx.append(i)
            else:
                wrong_idx.append(i)
        f.write(json.dumps({"task_id":tid,"correct_idx":correct_idx,"wrong_idx":wrong_idx})+"\n") 
    f.close()
    return

if __name__ == "__main__":
    """
    本模块函数说明: 对于输入为testcases的函数，都以load_testcase函数返回的形式为准
    load_testcase(test_file,type:int = 0): 
        记载testcase文件。
        type==0: 文件每行是{tid:testcase}
        type==1: 文件每行是{"task_id":tid,"ios":[{"tin":tin,"tout":tout}]}(uniform format)
        type==2: 文件每行是{"task_id":tid,"testcases":[...]}
        返回值是一个字典testcases = {tid:testcase}
    
    testcase_compare(problems,testcases): 
        testcasets是一个列表包含不同文件读取的testcases，列表的每个元素都是字典testcases = {tid:testcase}
        对于problems中的每个task函数会printtask的一些信息，以及testcasets中的每个字典里该task对应的testcase
        
    testfile_trans(test_file,resfile): 
        将test_file文件中的testcase转化成uniform format也就是{"task_id":tid,"ios":[{"tin":tin,"tout":tout}]}的形式写入resfile
    
    testcase_num(testcases):
        对testcases中testcase的数量进行分析，包括计算平均值，最大最小的等操作
    
    show_gened_testcase(test_file,verbose):
        打印最原始的生成的testcase文件内容
    
    correct_percent_analysis(test_file):
        输入文件中每行是个字典且应该包含“correct_percent”字段，该函数会统计和分析该字段的值的分布情况
    
    count_and_remove_duplicate(testcases):
        对testcases的testcase进行重复数量的统计和去重
    
    draw_hist(data,image_path)：
        data是一个整数列表，函数根据列表中的元素绘制直方图展示列表元素的分布情况，绘制的图像写入image_path文件
        
    write_testcase(test_cases,fpath,type:int=0):
        将testcases写入文件fpath，type==0时以{tid:testcases}每行的形式写入
        type==1时以{"task_id":tid,"ios":[{"tin":tin,"tout":tout}]}的形式写入
        
    mix_testcase(correct_testcases,all_testcases,correct_percent:float=0.5,keep_num:bool=False)
        往正确的testcase中混入错误的testcase使得最终的testcase正确率为correct_percent。keep_num会减少一些正确的testcase使得最终的testcase数量和输入的正确的测试用例一样。
        Add incorrect testcases to correct testcases.The returned mix_testcases have correct percent correct_testcases.
        if keep_num is True, the returned mix_testcases have the number of testcase as same as the input correct_testcases.
    """
    # testcase files
    test_from_check = "../data/test_from_check.jsonl"
    test_from_prompt = "../data/test_from_prompt.jsonl"

    
    correct_testcase_gened = "../try/gen_test_t0.8_topp0.95_sample100_max300_rm_correct.jsonl"
    
    testcase_file = "../try/gen_test_t0.8_topp0.95_sample100_max300_rm_correct.jsonl"
    wrong_testcase_file = "../try/gen_test_t0.8_topp0.95_sample100_max300_rm_wrong.jsonl"
    mix_testcase_file = "../try/gen_test_t0.8_topp0.95_sample100_max300_mix052.jsonl"
    mix_testcase_file_ios = "../try/gen_test_t0.8_topp0.95_sample100_max300_mix053_10ios.jsonl"
    
    check_testcase_file = "../try/gen_test_t0.8_topp0.95_sample100_max300_mix053_10ios.jsonl"
    
    
    # final_gened_testcases_idx = "../try/gen_test_t0.8_topp0.95_sample100_max300_rm_final5_idx.jsonl"
    
    prompt_testcase = load_testcase(test_from_prompt,type=1)
    # correct_testcase = load_testcase(test_file=correct_testcase_gened,type=2)
    # wrong_testcase  = load_testcase(test_file=wrong_testcase_file,type=0)
    check_testcases = load_testcase(test_from_check,type=1)
    
    # testcase_gened = "/home/S/hexiaolong/codex/self-debug/gen_tests_codellama34binst_num100_20240313.jsonl"
    # gened_testcases_file_final = "/home/S/hexiaolong/codex/self-debug/gen_tests_codellama7binst_num100_20240313_cleaned.jsonl"
    # gened_testcases_file_final = "../try/gen_test_t0.8_topp0.95_sample100_max300_rm_final5.jsonl"
    # idx_file = "../try/gen_test_t0.8_topp0.95_sample100_max300_rm_final5_idx.jsonl"
    # gened_testcases = load_testcase(testcase_gened,type=3)
    # final_gened_testcases = testcase_clean(gened_testcases)
    # write_testcase(final_gened_testcases,gened_testcases_file_final,type=0)
    # gened_tesctase = load_testcase(gened_testcases_file_final,type=0)
    # get_correct_and_wrong_idx(gened_tesctase,idx_file)
    problems = read_problems()
    ut_file = "/home/S/hexiaolong/codex/self-debug/data/test_from_prompt.jsonl"
    unit_tests,assertions,assertion_string = get_unit_test(ut_file)
    new_problems = []
    prompt_tests_num = []
    for tid,problem in problems.items():
        problem["prompt_tests"] = assertions[tid]
        prompt_tests_num.append(len(problem["prompt_tests"]))
        new_problems.append(problem)
    # write_testcase(new_problems,"/home/S/hexiaolong/codex/self-debug/data/HumanEval.jsonl",type=0)
    write_jsonl("/home/S/hexiaolong/codex/self-debug/data/humaneval.jsonl",new_problems)
    data_analysis(prompt_tests_num)
    # gened_testcases = load_testcase(gened_testcases_file_final,type=0)
    # idx_data = {}
    # correct_testcase = {}
    # wrong_testcase = {}
    # with open(final_gened_testcases_idx,"r") as f:
    #     for line in f.readlines():
    #         data = json.loads(line)
    #         tid = data["task_id"]
    #         correct_testcase_idx = data["correct_idx"]
    #         wrong_testcase_idx = data["wrong_idx"]
    #         correct_testcase[tid] = [gened_testcases[tid][i] for i in correct_testcase_idx]
    #         wrong_testcase[tid] = [gened_testcases[tid][i] for i in wrong_testcase_idx]
    #         idx_data[tid] = {"correct_idx":correct_testcase,"wrong_idx":wrong_testcase}
    
    # final_gened_testcases = load_testcase(gened_testcases_file_final,type=0)
    # num_list = []
    # cp_list = []
    # # f = open(final_gened_testcases_idx,"w+")
    # for tid in final_gened_testcases.keys():
    #     n = len(final_gened_testcases[tid])
    #     print(f"task {tid} has {len(final_gened_testcases[tid])} testcases.")
        # for t in final_gened_testcases[tid]:
        #     print(t)
        # print("===============================")
        # tsolution = problems[tid]["prompt"]+problems[tid]["canonical_solution"]
        # cnum = 0
        # correct_idx = []
        # wrong_idx = []
        # for i,t in enumerate(final_gened_testcases[tid]):
        #     checkp = tsolution + "\n" + t +" \n"
        #     res = check_test_correctness(checkp,0.1)
        #     if res["passed"]:
        #         cnum += 1
        #         correct_idx.append(i)
        #     else:
        #         wrong_idx.append(i)
        # cp = cnum/n
        # cp_list.append(cp)
        # num_list.append(n)
        # assert len(correct_idx) + len(wrong_idx) == n
        # f.write(json.dumps({"task_id":tid,"correct_idx":correct_idx,"wrong_idx":wrong_idx})+"\n")
    # f.close()
    # data_analysis(num_list)
    # draw_hist(data=num_list,image_path="../image/final_gened_testcase_num.png")
    # data_analysis(cp_list)
    # draw_hist(data=cp_list,image_path="../image/final_gened_testcase_correct_percent.png")
    # testcase_num(check_testcases)
    # testcase_num = []
    # for k,v in check_testcases.items():
    #     testcase_num.append(len(v))
    # draw_hist(data=testcase_num,image_path="../image/check_testcase_num.png")
    
    ##################################################
    # mix_testcase = {}
    # for tid in correct_testcase.keys():
    #     num = 5
        # mix09 = testcase_wait_for_check[tid]
        # cts = correct_testcase[tid]
        # ft = []
        # for t in mix09:
        #     if t in cts:
        #         ft.append(t)
        # mix_testcase[tid] = ft
        # ct = correct_testcase[tid][:num]
        # random.seed(2024)
    #     ctn = len(correct_testcase[tid])
    #     if ctn < num:
    #         num = ctn
    #     # ct = random.sample(correct_testcase[tid],num)
    #     ct = correct_testcase[tid][:num]
    #     cn = len(ct)
    #     if cn == 0:
    #         num = 5
    #         print("+++++++++++++++++++++")
    #         print(tid)
    #         print("+++++++++++++++++++++")
    #         if len(prompt_testcase[tid]) < num:
    #             num = len(prompt_testcase[tid])
    #         ct = prompt_testcase[tid][:num]#random.sample(prompt_testcase[tid],num)#
    #         cn = len(ct)
    #     wt = random.sample(wrong_testcase[tid],cn)
    #     wn = len(wt)
    #     # assert wn == cn
    #     mix_testcase[tid] = ct + wt
    #     print("------------------")
    #     print(tid,cn,wn)
    # data_analysis([len(t) for tid,t in mix_testcase.items()])
    # write_testcase(mix_testcase,mix_testcase_file,type=0)
    # write_testcase(mix_testcase,mix_testcase_file_ios,type=1)
    # testcase_wait_for_check = load_testcase(check_testcase_file,type=1)
    # check_correct_percent(testcases_correct=correct_testcase,test_wait_for_check=testcase_wait_for_check)
    # for tid in correct_testcase.keys():
    #     print("=====================================================")
    #     print(f"task {tid} has {len(correct_testcase[tid])} correct testcases and {len(wrong_testcase[tid])} wrong testcases.")
    #     print(f"the correct percent is {len(correct_testcase[tid])/(len(correct_testcase[tid])+len(wrong_testcase[tid]))}")
    #####################################################
    # mix_testcases = mix_testcase(correct_testcases=testcases_correct,all_testcases=testcases_all,correct_percent=0.5,keep_num=False)
    # write_testcase(mix_testcases,fpath="../try/gen_test_t0.8_topp0.95_sample100_max300_rm_mix0.5.jsonl",type=0)
    # CODET_tescase_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_filter_add.jsonl"
    # add_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test3_add.jsonl"
    # res_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_filter_add.jsonl"
    # testcases_merge(CODET_tescase_file,add_file,res_file)
    # testcase_num2(CODET_tescase_file)
    # res_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_rm_uniform_withlinechange.jsonl"
    # testfile_trans("../try/gen_test_t0.8_topp0.95_sample100_max300_rm.jsonl",res_file)
    # t1 = load_testcase_uniform(test_from_check)
    # t2 = load_testcase_uniform(test_from_prompt)
    # t3 = load_testcase(correct_testcase_gened2,type=2)
    # ts = {"check":t1,"prompt test":t2,"gened":t3}
    # testcase_compare(problems,ts)
    # show_gened_testcase(gened_testcase)
    # print(len(never_passed_task))
    # print(len(pt_not_passed_task))
    # print(len(never_passed_task))
    # correct_percent_analysis(correct_testcase_gened2)
    # gened = load_testcase_gened3(gened_testcase)
    # testcases_rmduplicate = count_and_remove_duplicate(testcases=gened)
    # write_testcase(testcases_rmduplicate,"../try/gen_test_t0.8_topp0.95_sample100_max300_rm.jsonl")
    # idx_data = {}
    # cf = "../try/gen_test_t0.8_topp0.95_sample100_max300_idx.jsonl"
    # for tid,testcases in gened_testcases.items():
    #     cts = correct_testcase[tid]
    #     wts = wrong_testcase[tid]
    #     cts_idx = []
    #     wts_idx = []
    #     for idx,test in enumerate(testcases):
    #         if test in cts:
    #             cts_idx.append(idx)
    #         else:
    #             wts_idx.append(idx)
    #     idx_data[tid] = {"correct_idx": cts_idx,"wrong_idx":wts_idx}
    # with open(cf,"w+") as f:
    #     f.write(json.dumps(idx_data))
            
        
        
    