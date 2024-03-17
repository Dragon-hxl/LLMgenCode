import torch
import json
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import hydra
from omegaconf import DictConfig, OmegaConf

import faulthandler
faulthandler.enable(all_threads=True)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import sys
sys.path.append("/home/S/hexiaolong/codex/self-debug")
sys.path.append("/home/S/hexiaolong/codex/self-debug/humaneval")
from human_eval.data import read_problems
from human_eval.execution import run_code_with_test_result,run_code_with_output2
from myutils import setup_seed,make_printv,print_with_tag,code_clean2,get_unit_test,prompt_for_64,filter_fix_ans,get_CODET_point_v1,get_CODET_point_v3,get_pass_rate,get_UTfeedback_prompt_v1,start_code_extract
from myutils import load_testcase
from model import Model
from generator import PyGenerator


class Node:
    def __init__(self,solution:str,parent=None,prompt:str="",passT_rate:float=-1.0,prob:float=-1.0,depth:int=0,feedbackprompt:str="") -> None:
        self.solution = solution
        self.parent = parent
        self.children = []
        self.prompt = prompt
        self.passT_rate = passT_rate
        self.pass_ut_num = 0
        self.total_ut_num = 0
        self.prob = prob
        self.depth = depth
        self.reflect = ""
        self.feedbackprompt = feedbackprompt #由这个node的solution产生的feedbackprompt
        self.idx = 0
        self.CODET_point = 0.0
        self.CODET_pass_testcase = set()
        self.CODET_pass_rate = 0.0
        self.CODET_total_test_num = 0
        self.already_CODET = False
        self.idx = 0
    def __repr__(self) -> str:
        return f"solution:\n{self.solution}\npassT_rate:{self.passT_rate}\nprob:{self.prob}\n"
    
    def __eq__(self, other: object) -> bool:
        return self.solution == self.solution
    
    def __hash__(self) -> int:
        return hash(self.solution)
    
    def show_parents(self) -> None:
        print("++++++show parents of the node++++++")
        print(self.__repr__())
        print("************************")
        if self.parent:
            self.parent.show_parents()
        return None


# prompt file 
prompt_root = "/home/S/hexiaolong/codex/self-debug/data/prompt/"
prompt_file = prompt_root + "prompt_base2.txt"
UTfeedback_file = prompt_root + "prompt_UTfeedback.txt"
# test file
data_root = "/home/S/hexiaolong/codex/self-debug/data/"
ut_file = data_root + "test_from_prompt.jsonl"# 从问题中提取的unit tests所在的文件
true_tests_file = data_root + "test_from_check.jsonl"# 存放了最终用来check的unit_tests
gened_testcases_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_rm_final5.jsonl"# 用来进行CODET的tests文件

def run_testcase_filter(
    data:dict,
    model_path:str,
    output_file:str,
    sample_num:int=10,
    cir_times:int=10,
    verbose:bool=False,
):
    
    setup_seed(2024)
    print_v = make_printv(verbose)
    model = Model(model_path)
    gen = PyGenerator()
    offline = False
    
    #读取prompt
    with open(prompt_file,"r") as f:
        preflex = f.read()
    with open(UTfeedback_file,"r") as f:
        UTfeedback_promt = f.read()
    
    #读取unit tests
    base_unit_tests,base_assertions,base_assertion_string = get_unit_test(ut_file)
    unit_tests,assertions,assertion_string = get_unit_test(ut_file)

    gened_testcase = load_testcase(gened_testcases_file,type=0)
    
    model = Model(model_path)
    
    #获取solution
    problems = read_problems()
    taskids = list(problems.keys())
    f = open(output_file,"w+",encoding="utf-8")
    if f:
        print_v(f"open file {output_file} success!")
    for tid in taskids:
        num_id = int(tid.split("/")[1])
        if num_id < 3 or num_id > 5:
            continue
        
        task_begin_time = time.time()
        #
        if offline:
            task_gened_testcase = gened_testcase[tid]
        else:
            task_gened_testcase = gen.gen_tests(model,problems[tid],num=100,verbose=verbose)
            print_v(f"gen {len(task_gened_testcase)} tests for task {tid}.")
        print_v(f"get solution for task : {tid} with {len(unit_tests[tid])} tests.")
        
        step_one_st = time.time()
        tprompt = problems[tid]["prompt"]
        if tid == "HumanEval/64":
            tprompt = prompt_for_64
        base_prompt,solution,model_inference_time,input_tokens_len, output_tokens_len = gen.generate_base_complication(model,tprompt,base_assertion_string[tid],record_time=True,verbose=verbose)
        
        # 去除函数头和注释
        entry_point = "def " + problems[tid]["entry_point"]
        solution = code_clean2(code=solution,entry_point=entry_point)
        # 打印生成的初始代码
        print_with_tag(content=solution,tag="solution",verbose=verbose)
        # 去掉problems[tid]["prompt"]中的注释和空行
        start_code = start_code_extract(tprompt,entry_point)
        
        # debug 准备
        run_test = [t["tin"] for t in unit_tests[tid]]
        feedback_prompt = UTfeedback_promt + assertion_string[tid] + "\n\n# Complete the Python funtion:\n" + tprompt+"### result ###\n```python\n" + start_code + "\n"
        fix_input = model.tokenizer(feedback_prompt, return_tensors='pt', return_token_type_ids=False)
        print_v(f"fix input length is {fix_input.input_ids.shape}")
        fix_input_len = fix_input.input_ids.shape[1]
        step_one_total_time = (time.time() - step_one_st)/60
        
        # 开始生成feedback和新代码的循环
        cir = 0
        output_short = {}
        node1 = Node(solution,prompt=base_prompt,depth=cir,feedbackprompt=feedback_prompt)
        nodes,gened_nodes,chosen_nodes = [node1],[node1],[node1]
        left_nodes,time_record,fix_record = [],[],[]
        prompt_test_len = len(base_assertions[tid])
        chosen_num = max(0,8 - prompt_test_len)
        
        chosen_testcase = base_assertions[tid]
        chosen_testcase_id_dict = {}
        pass_testcase_list_dict = {}
        while True:
            st = time.time()
            stop = False
            # 运行所有的solution得到通过的test数量和得分
            run_test = [t.replace("assert ","").split("==")[0].strip() for t in chosen_testcase]
            feedback_prompt = UTfeedback_promt + "\n".join(chosen_testcase) + "\n\n# Complete the Python funtion:\n" + tprompt+"### result ###\n```python\n" + start_code + "\n"
            fix_input_len = model.tokenizer(feedback_prompt, return_tensors='pt', return_token_type_ids=False).input_ids.shape[1]
            get_pass_rate(gened_nodes,base_assertions[tid],tid)
            for i,node in enumerate(gened_nodes):
                solution = node.solution
                # 这里通过一次函数调用同时获得simple和UTfeedback，也就是会判断代码是否正确，同时对于出现AssertError的代码会得到其执行的第一个unit test的值。其他Error因为会返回具体的错误信息就不会得到执行的第一个unit test的值。
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        args = (problems[tid], (start_code+solution), run_test, assertions[tid], 0.1)
                        future = executor.submit(run_code_with_test_result, *args)
                        result = future.result()
                        passed = result["passed"]
                        final_res = result["result"]
                    prompt,passn,pass_tests = get_UTfeedback_prompt_v1(feedback_prompt, solution, passed, final_res, run_test, assertions[tid])
                    node.feedbackprompt = prompt
                    node.passT_rate = passn
                    node.pass_ut_num = pass_tests
                    node.total_ut_num = len(run_test)
                except BaseException as e:
                    print(f"run solution failed with {e}")
                if passn >= 1.0:
                    print("passn:",passn)
                    stop=True
                    print("One node passed! Show it and it's parents.")
                    node.show_parents()
                    break
            
            run_solutions_time = (time.time() - st)/60
            # print(f"Run all solutions spends {run_solutions_time} mins.")
            
            # 对生成的代码进行排序并选取排序靠前的代码,同时进行测试用例筛选
            choose_start = time.time()
            total_nodes = gened_nodes + left_nodes
            total_unique_nodes = list(set(total_nodes))
            
            # try:
            sorted_group,chosen_testcase_id  = get_CODET_point_v3(total_nodes,task_gened_testcase,tid,sort_len=True,chosen_num=chosen_num) #这里是使用去重后的还是不去重的,sort_len=True
            if chosen_testcase_id!=[]:
                chosen_testcase = base_assertions[tid] + [task_gened_testcase[x] for x in chosen_testcase_id]
            else:
                print("chosen_testcase_id is empty!")
            print_v(f"{len(chosen_testcase)} chosen_testcase:")
            print_v("\n".join(chosen_testcase))
            pass_testcase_list = [ list(x[1]) for x in sorted_group]
            # except BaseException as e:
            #     print(f"get CODET point failed with {e}")
            # except:
            #     print("get CODET point failed!")
            sorted_nodes = sorted(total_unique_nodes,key=lambda x: (x.CODET_pass_rate,x.prob),reverse=True)
            
            chosen_nodes = sorted_nodes[:sample_num]
            left_nodes = sorted_nodes[sample_num:]
            choose_solution_time = (time.time()-choose_start)/60
            
            if verbose:
                print(f"task:{tid}, cir:{cir}, gened {len(gened_nodes)} solutions, total nodes:{len(total_nodes)}, total unique nodes:{len(total_unique_nodes)}, chosen nodes:{len(chosen_nodes)}, left nodes:{len(left_nodes)}")
                print(f"chosen nodes idx is {[n.idx for n in chosen_nodes]}")
                print(f"chosen nodes's parent's idx is {[n.parent.idx for n in chosen_nodes if n.parent]}")
                print(f"chosen nodes passT_rates {[n.passT_rate for n in chosen_nodes]}\nprobs are {[n.prob for n in chosen_nodes]}\n")#CODET point are {[n.CODET_point for n in chosen_nodes]}\nprobs are {[n.prob for n in chosen_nodes]}\n
                
            output_short[cir] = [{"solution":n.solution,"passT_rate":n.passT_rate,"prob":n.prob} for n in chosen_nodes]
            chosen_testcase_id_dict[cir] = chosen_testcase_id
            pass_testcase_list_dict[cir] = pass_testcase_list
            time_record.append({"cir":cir,"model_inference_time":model_inference_time,"run_solutions_time":run_solutions_time,"choose_solution_time":choose_solution_time})
            if stop or cir>=cir_times:
                break
            cir += 1
            gened_nodes = []
            # 使用选择的node进行下一步的debug
            st = time.time()
            len_record = []
            k=1
            return_sequences = int(sample_num/k)
            total_output_length = 0
            for i,node in enumerate(chosen_nodes):
                feedback = node.feedbackprompt
                inputs = model.tokenizer(feedback, return_tensors='pt', return_token_type_ids=False)
                input_length = inputs.input_ids.shape[1]
                fix_percent = (fix_input_len*(fix_input_len - 1.0))/(input_length*(input_length - 1.0))
                for _ in range(k):
                    solutions= gen.generate_with_feedback(model,feedback,return_sequences=return_sequences,verbose=True)
                    for s in solutions:
                        ans,true_sc,output_length = s[0],s[1],s[2]
                        # 记录每条solution的长度
                        total_output_length += output_length
                        len_record.append((input_length,fix_input_len,fix_percent,output_length,i))
                        #创建node
                        solution = filter_fix_ans(ans, entry_point, start_code)
                        tmp_node = Node(solution=solution,parent=node,prompt=feedback,prob=true_sc,depth=cir)
                        tmp_node.idx = len(nodes)
                        node.children.append(tmp_node.idx)
                        gened_nodes.append(tmp_node)
                        nodes.append(tmp_node)
            model_inference_time = (time.time()-st)/60
            average_output_length = total_output_length/len(gened_nodes)
            every_token_time= (time.time()-st)/average_output_length
            fix_record.append({"cir":cir,"len_record":len_record,"every_token_time":every_token_time})
            if verbose:
                print(f"run solution time is {run_solutions_time} mins, choose solution time is {choose_solution_time} mins, model inference time is {model_inference_time} mins.")
                print(f"average output length is {average_output_length}, every token time is {every_token_time} s.")
        task_end_time = time.time()
        task_time = (task_end_time - task_begin_time)/60
        if not offline:
            f.write(json.dumps({"task_id": tid,"completion":output_short,"time_record":time_record,"fix_record":fix_record,"step_one_total_time":step_one_total_time,"step_one_tokens_len":(input_tokens_len,output_tokens_len),"task_time":task_time,"chosen_testcase_id_dict":chosen_testcase_id_dict,"pass_testcase_list_dict":pass_testcase_list_dict,"gened_testcase":gened_testcase})+"\n")
            f.flush()
        else:
            f.write(json.dumps({"task_id": tid,"completion":output_short,"time_record":time_record,"fix_record":fix_record,"step_one_total_time":step_one_total_time,"step_one_tokens_len":(input_tokens_len,output_tokens_len),"task_time":task_time,"chosen_testcase_id_dict":chosen_testcase_id_dict,"pass_testcase_list_dict":pass_testcase_list_dict})+"\n")
            f.flush()
    f.close()