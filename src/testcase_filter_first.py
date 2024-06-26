import torch
import json
import time
import random

import faulthandler
faulthandler.enable(all_threads=True)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import sys
sys.path.append("/home/S/hexiaolong/codex/self-debug")
sys.path.append("/home/S/hexiaolong/codex/self-debug/humaneval")
from myutils import setup_seed,make_printv,print_with_tag,code_clean2,filter_fix_ans,get_CODET_point_v3,get_pass_rate,get_UTfeedback_prompt_v1,start_code_extract
from myutils import load_testcase
from model import Model
from generator import PyGenerator
from executor import py_executor


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
UTfeedback_file = prompt_root + "prompt_UTfeedback_short.txt"
simple_feedback_shot2 = prompt_root + "prompt_simfeedback_paper.txt"
simple_feedback_shot1 = prompt_root + "prompt_simfeedback.txt"
expl_feedback_shot1 = prompt_root + "prompt_explfeedback_short.txt"

gened_testcases_file = "/home/S/hexiaolong/codex/self-debug/src/gened_testcase_codellama7bpy_humaneval.jsonl"# 用来进行CODET的tests文件
mbpp_gened_testcase = "/home/S/hexiaolong/codex/self-debug/res/gened_tesatcase_mbpp_codellama7bpy.jsonl"


def run_testcase_filter(
    dataset:dict,
    model_path:str,
    output_file:str,
    sample_num:int=10,
    filter_num:int=1,
    first_num:int=1,
    cir_times:int=10,
    feedback_type:str="UT",
    verbose:bool=False,
):
    
    setup_seed(2024)
    print_v = make_printv(verbose)
    model = Model(model_path)
    gen = PyGenerator()
    exe = py_executor()
    offline = False
    
    print_v(f"model_path : {model_path}")
    print_v(f"sample_num : {sample_num}")
    print_v(f"Run testcase filter self-debug with max cir : {cir_times}.")
    
    #important读取prompt
    if feedback_type=="UT":
        print_v("Use UT feedback.")
        with open(UTfeedback_file,"r") as f:
            prompt_shot = f.read()
    elif feedback_type=="simple":
        print_v("Use simple feedback.")
        with open(simple_feedback_shot1,"r") as f:
            prompt_shot = f.read()
    elif feedback_type=="expl":
        print_v("Use expl feedback.")
        with open(expl_feedback_shot1,"r") as f:
            prompt_shot = f.read()
        
        
    #打开输出文件
    f = open(output_file,"w+",encoding="utf-8")
    full_output_file = output_file.replace(".jsonl","_full.jsonl")
    fullf = open(full_output_file,"w+",encoding="utf-8")
    if f and fullf:
        print_v(f"open {output_file} and {full_output_file} success.")
    #开始生成solution
    for data in dataset:
        tid = data["task_id"]
        has_visibale_tests = False
        
        task_begin_time = time.time()
        #important if no offline testcase offered, gen testcases
        
        if "HumanEval" in tid and "codellama-7bpy" in model_path:
            offline = True
            gened_testcases_file = "/home/S/hexiaolong/codex/self-debug/src/gened_testcase_codellama7bpy_humaneval.jsonl"
        if "MBPP" in tid:
            offline = True
            gened_testcases_file = mbpp_gened_testcase
        if offline:
            print_v(f"Use off line testcases in path {gened_testcases_file}")
            gened_testcase = load_testcase(gened_testcases_file,type=0)
            task_gened_testcase = gened_testcase[tid]
            chosen_testcase = task_gened_testcase[:10]
        else:
            print_v("Gen testcase by model.")
            gen_tests_st = time.time()
            task_gened_testcase = gen.gen_tests_sort_by_prob(model,data,num=110,verbose=verbose)[:110]
            gen_tests_time = (time.time() - gen_tests_st)/60
            print_v(f"Gen {len(task_gened_testcase)} tests for task {tid}, use {gen_tests_time} mins.")
            chosen_testcase = task_gened_testcase[:10]
            random.shuffle(task_gened_testcase)
        
        # first generate the base solution
        step_one_st = time.time()
        tprompt = data["prompt"]
        #important
        entry_point = "def " + data["entry_point"]
        use_prompt_test = True
        base_return_num = first_num
        if data.get('prompt_tests', []) == [] or not use_prompt_test:
            print("No visiable testcase.")
            # base_prompt,solution,model_inference_time,input_tokens_len, output_tokens_len\
            #     = gen.generate_base_complication(model,tprompt,"",record_time=True,verbose=verbose)
            solutions,model_inference_time,input_tokens_len, output_tokens_len = \
            gen.generate_base_complication_multi(model,tprompt,"\n".join(chosen_testcase),
                                                 entry_point=entry_point,return_nums=base_return_num,record_time=True,verbose=verbose)
        
            has_visibale_tests = False
        else :
            print("Use visiable testcase.")
            base_assertion_string = "\n".join(data["prompt_tests"])
            solutions,model_inference_time,input_tokens_len, output_tokens_len = \
            gen.generate_base_complication_multi(model,tprompt,base_assertion_string,
                                                 entry_point=entry_point,return_nums=base_return_num,record_time=True,verbose=verbose)
            
            has_visibale_tests = True
            chosen_testcase = data["prompt_tests"]
        # 去除函数头和注释
        # solution = code_clean2(code=solution,entry_point=entry_point)
        # node1 = Node(solution,prompt="",depth=0)
        # print_with_tag(content=solution,tag="base solution",verbose=verbose)
        # 去掉data["prompt"]中的注释和空行
        start_code = start_code_extract(tprompt,entry_point)
        step_one_total_time = (time.time() - step_one_st)/60
        
        # 开始生成feedback和新代码的循环
        cir = 0
        output_short = {}
        output_full = {} # 存放所有生成的solution
        chosen_testcase_dict = {}
        chosen_testcase_id_dict = {}
        pass_testcase_list_dict = {}
        nodes,gened_nodes,chosen_nodes = [],[],[]
        for s in solutions:
            node = Node(s,prompt="",depth=cir,feedbackprompt="")
            node.idx = len(nodes)
            print_v(f"node {node.idx} solution is:\n{s}")
            nodes.append(node)
            gened_nodes.append(node)
        left_nodes,time_record,fix_record = [],[],[]
        print_v(f"Init node idx:{[n.idx for n in nodes]}")
        while True:
            st = time.time()
            stop = False
            
            # 在筛选的测试用例上执行代码
            print_v("chosen testcases are:")
            print_v("\n".join(chosen_testcase))
            #important
            total_nodes = gened_nodes + left_nodes
            total_unique_nodes = list(set(total_nodes))
            feedback_prompt = prompt_shot + "\n".join(chosen_testcase) + "\n\n# Complete the Python funtion:\n" + tprompt+"\n### result ###\n```python\n" + start_code + "\n"
            #important
            fix_input = model.tokenizer(feedback_prompt, return_tensors='pt', return_token_type_ids=False)
            print_v(f"fix input length is {fix_input.input_ids.shape}")
            fix_input_len = fix_input.input_ids.shape[1]
            
            #important
            if has_visibale_tests:
                    get_pass_rate(data,gened_nodes,data["prompt_tests"])
            
            for i,node in enumerate(total_nodes):
                solution = start_code + node.solution
                if i == 0:  
                    print_v(f"check program : \n{start_code+solution}")
                # 这里通过一次函数调用同时获得simple和UTfeedback，也就是会判断代码是否正确，同时对于出现AssertError的代码会得到其执行的第一个unit test的值。其他Error因为会返回具体的错误信息就不会得到执行的第一个unit test的值。
                #important
                feedback,passn,pass_tests = exe.excute(solution=solution,
                                                       tests=chosen_testcase,
                                                       feedback_prompt=feedback_prompt,
                                                       timeout=0.1,
                                                       feedback_type=feedback_type)
                if cir==1 and i==0:
                    print_v(f"{feedback_type} feedback:\n {feedback}")
                node.feedbackprompt = feedback
                node.passT_rate = passn
                node.pass_ut_num = pass_tests
                node.total_ut_num = len(chosen_testcase)
                #important
                if has_visibale_tests:
                    if node.CODET_pass_rate >=1.0:
                        print("passn:",node.CODET_pass_rate)
                        stop=True
                        print("One node passed! Show it and it's parents.")
                        node.show_parents()
                        break
                else:
                    if passn >= 1.0 and len(total_nodes) > 20:
                        print("passn:",passn)
                        stop=True
                        print("One node passed! Show it and it's parents.")
                        node.show_parents()
                        break
            run_solutions_time = (time.time() - st)/60
            
            # filter testcase
            filter_st = time.time()
            #important
            sorted_group,chosen_testcase_id  = get_CODET_point_v3(total_nodes,
                                                                  task_gened_testcase,
                                                                  data,
                                                                  chosen_num=10,
                                                                  count_solution_num=True,
                                                                  verbose=verbose,
                                                                  sort_len=True) #这里是使用去重后的还是不去重的,sort_len=True
            pass_testcase_list = [ list(x[1]) for x in sorted_group]
            if has_visibale_tests:
                chosen_num = 10
                if len(total_unique_nodes) > 10 or cir > 1:
                    chosen_testcase = data["prompt_tests"] + [task_gened_testcase[x] for x in chosen_testcase_id]
                    chosen_testcase = chosen_testcase[:chosen_num]
            else:
                
                chosen_num = 10
                if len(total_unique_nodes) > 10 or cir > 1:
                    chosen_testcase = [task_gened_testcase[x] for x in chosen_testcase_id][:chosen_num]
            #important
            filter_time = (time.time()-filter_st)/60
            print_v(f"filter testcase time is {filter_time} mins.")
            
            #  对代码进行排序并选取排序靠前的代码
            choose_start = time.time()
            #important
            total_nodes = gened_nodes + left_nodes
            total_unique_nodes = list(set(total_nodes))
            if has_visibale_tests:
                # get_pass_rate(data,total_nodes,data["prompt_tests"])
                print("Sort nodes in has visibale test task.")
                #important
                sorted_nodes = sorted(total_unique_nodes,key=lambda x: (x.CODET_pass_rate,x.prob),reverse=True)#,-len(x.solution)
            else:
                sorted_nodes = sorted(total_unique_nodes,key=lambda x: (x.passT_rate,x.prob),reverse=True)
            #important
            chosen_nodes = sorted_nodes[:filter_num]
            left_nodes = sorted_nodes[filter_num:]
            #important
            choose_solution_time = (time.time()-choose_start)/60
            
            print_v(f"task:{tid}, cir:{cir}, gened {len(gened_nodes)} solutions, total nodes:{len(total_nodes)}, total unique nodes:{len(total_unique_nodes)}, chosen nodes:{len(chosen_nodes)}, left nodes:{len(left_nodes)}")
            print_v(f"chosen nodes idx is {[n.idx for n in chosen_nodes]}")
            print_v(f"chosen nodes's parent's idx is {[n.parent.idx for n in chosen_nodes if n.parent]}")
            print_v(f"chosen nodes's depth is {[n.depth for n in chosen_nodes]}")
            print_v(f"chosen nodes passT_rates {[n.passT_rate for n in chosen_nodes]}\nprobs are {[n.prob for n in chosen_nodes]}\nprompt_pass_rate are {[n.CODET_pass_rate for n in chosen_nodes]}")
            
            output_short[cir] = [{"solution":n.solution,"passT_rate":n.passT_rate,"prob":n.prob} for n in chosen_nodes]
            output_full[cir] = [{"solution":n.solution,"passT_rate":n.passT_rate} for n in gened_nodes]
            chosen_testcase_dict[cir] = chosen_testcase
            chosen_testcase_id_dict[cir] = chosen_testcase_id
            pass_testcase_list_dict[cir] = pass_testcase_list
            time_record.append({"cir":cir,"model_inference_time":model_inference_time,"run_solutions_time":run_solutions_time,"choose_solution_time":choose_solution_time,"filter_time":filter_time})
            
            if stop or cir >= cir_times:
                break
            cir += 1
            # 使用选择的代码进行下一步的self-debug
            st = time.time()
            len_record = []
            gened_nodes = []
            k=1
            return_sequences = int(sample_num/k)
            if cir==1 and filter_num==5 and sample_num==2:
                return_sequences = int(10/len(chosen_nodes))
            total_output_length = 0
            for i,node in enumerate(chosen_nodes):
                feedback = node.feedbackprompt
                
                code = start_code + node.solution
                #important
                if feedback_type=="expl":
                    code_expl = gen.gen_code_explanation(model,code,verbose)
                    idx = feedback.index("Feedback:")
                    feedback = feedback[:idx] + f"Here is a line-by-line explanation of the code:\n{code_expl}\n\n" + feedback[idx:]
                if cir==1 and i==1:
                    print_with_tag(content=feedback,tag="feedback",verbose=verbose)
                
                inputs = model.tokenizer(feedback, return_tensors='pt', return_token_type_ids=False)
                input_length = inputs.input_ids.shape[1]
                fix_percent = (fix_input_len*(fix_input_len - 1.0))/(input_length*(input_length - 1.0))
                for _ in range(k):
                    solutions,inference_time= gen.generate_with_feedback(model,
                                                                         feedback,
                                                                         return_sequences=return_sequences,
                                                                         verbose=True)
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
            print_v(f"run solution time is {run_solutions_time} mins, choose solution time is {choose_solution_time} mins, model inference time is {model_inference_time} mins, filter time is {filter_time} mins.")
            print_v(f"average output length is {average_output_length}, every token time is {every_token_time} s.")
            
        task_end_time = time.time()
        task_time = (task_end_time - task_begin_time)/60
        print_v(f"Task {tid} use total {task_time} mins.")
        if not offline:
            f.write(json.dumps({"task_id": tid,"completion":output_short,"time_record":time_record,"fix_record":fix_record,"step_one_total_time":step_one_total_time,"step_one_tokens_len":(input_tokens_len,output_tokens_len),"task_time":task_time,"chosen_testcase_dict":chosen_testcase_dict,"pass_testcase_list_dict":pass_testcase_list_dict,"gened_testcase":task_gened_testcase,"internal_tests":chosen_testcase,"chosen_testcase_id_dict":chosen_testcase_id_dict,"gen_tests_time":gen_tests_time})+"\n")
            f.flush()
        else:
            f.write(json.dumps({"task_id": tid,"completion":output_short,"time_record":time_record,"fix_record":fix_record,"step_one_total_time":step_one_total_time,"step_one_tokens_len":(input_tokens_len,output_tokens_len),"task_time":task_time,"chosen_testcase_dict":chosen_testcase_dict,"pass_testcase_list_dict":pass_testcase_list_dict})+"\n")
            f.flush()
        fullf.write(json.dumps({"task_id":tid,"completion":output_full})+"\n")
        fullf.flush()
    f.close()
    fullf.close()
    return 0