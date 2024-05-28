import torch
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import faulthandler
faulthandler.enable(all_threads=True)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import sys
sys.path.append("/home/S/hexiaolong/codex/self-debug")
sys.path.append("/home/S/hexiaolong/codex/self-debug/humaneval")
from human_eval.data import read_problems
from human_eval.execution import run_code_with_test_result,run_code_with_output2
from myutils import setup_seed,make_printv,print_with_tag,code_clean2,prompt_for_64,filter_fix_ans,get_CODET_point_v1,get_CODET_point_v3,get_pass_rate,get_UTfeedback_prompt_v1,get_UTfeedback_prompt,start_code_extract
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
expl_feedback_shot1 = prompt_root + "prompt_explfeedback_shot.txt"

def run_not_tree_search(
    dataset:list,
    model_path:str,
    output_file:str,
    sample_num:int=10,
    filter_num:int=1,
    cir_times:int=10,
    feedback_type:str="UT",
    verbose:bool=False,
):
    
    setup_seed(2024)
    print_v = make_printv(verbose)
    model = Model(model_path)
    gen = PyGenerator()
    print_v("Run not tree search.")
    
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
    #important#important
    
    #获取solution
    f = open(output_file,"w+",encoding="utf-8")
    
    for data in dataset:
        task_begin_time = time.time()
        
        tid = data["task_id"]
        internal_tests = []
        if data.get('prompt_tests', []) == []:
            print_v("Use internal_tests.")
            #important
            internal_tests = gen.gen_tests_sort_by_prob(model,data,num=10,verbose=False)
            internal_tests = internal_tests[:6]
            #important
            print_v('\n'.join(internal_tests))
            assertions = internal_tests
            base_assertion_string = "\n".join(assertions)
            assertion_string = "\n".join(assertions)
        else :
            print_v("Use prompt_tests.")
            assertions = data["prompt_tests"]
            base_assertion_string = "\n".join(assertions)
            assertion_string = "\n".join(assertions)
        
        print_v(f"get solution for task : {tid} with {len(assertions)} tests.")

        step_one_st = time.time()
        tprompt = data["prompt"]
        base_prompt,solution,model_inference_time,input_tokens_len, output_tokens_len = gen.generate_base_complication(model,tprompt,base_assertion_string,record_time=True,verbose=verbose)
        
        # 去除函数头和注释
        entry_point = "def " + data["entry_point"]
        solution = code_clean2(code=solution,entry_point=entry_point)
        # 打印生成的初始代码
        print_with_tag(content=solution,tag="solution",verbose=verbose)
        # 去掉data["prompt"]中的注释和空行
        start_code = start_code_extract(tprompt,entry_point)
        
        # debug 准备
        run_test = [t.split("==")[0].replace("assert","").strip() for t in assertions]
        feedback_prompt = prompt_shot + assertion_string + "\n\n# Complete the Python funtion:\n" + tprompt+"\n### result ###\n```python\n" + start_code + "\n"
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
        
        while True:
            st = time.time()
            stop = False
            # 运行所有的solution得到通过的test数量和得分
            for i,node in enumerate(gened_nodes):
                solution = node.solution
                if i == 0:  
                    print_v(f"check program : \n{start_code+solution}")
                # 这里通过一次函数调用同时获得simple和UTfeedback，也就是会判断代码是否正确，同时对于出现AssertError的代码会得到其执行的第一个unit test的值。其他Error因为会返回具体的错误信息就不会得到执行的第一个unit test的值。
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        args = (data, (start_code+solution), run_test, assertions, 0.1)
                        future = executor.submit(run_code_with_test_result, *args)
                        result = future.result()
                        passed = result["passed"]
                        final_res = result["result"]
                    #important
                    prompt,passn,pass_tests = get_UTfeedback_prompt_v1(feedback_prompt, 
                                                                       solution, 
                                                                       passed, 
                                                                       final_res, 
                                                                       run_test, 
                                                                       assertions, 
                                                                       feedback_type)
                    #important
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
            
            # 对生成的代码进行排序并选取排序靠前的代码
            choose_start = time.time()
            #important
            total_nodes = gened_nodes #+ left_nodes
            total_unique_nodes = list(set(total_nodes))
            sorted_nodes = sorted(total_unique_nodes,
                                  key=lambda x: (x.passT_rate,x.prob),#,-len(x.solution)
                                  reverse=True)
            chosen_nodes = sorted_nodes[:filter_num]
            left_nodes = sorted_nodes[filter_num:]
            #important
            choose_solution_time = (time.time()-choose_start)/60
            
            print_v(f"task:{tid}, cir:{cir}, gened {len(gened_nodes)} solutions, total nodes:{len(total_nodes)}, total unique nodes:{len(total_unique_nodes)}, chosen nodes:{len(chosen_nodes)}, left nodes:{len(left_nodes)}")
            print_v(f"chosen nodes idx is {[n.idx for n in chosen_nodes]}")
            print_v(f"chosen nodes's parent's idx is {[n.parent.idx for n in chosen_nodes if n.parent]}")
            print_v(f"chosen nodes passT_rates {[n.passT_rate for n in chosen_nodes]}\nprobs are {[n.prob for n in chosen_nodes]}\n")#CODET point are {[n.CODET_point for n in chosen_nodes]}\nprobs are {[n.prob for n in chosen_nodes]}\n
                
            output_short[cir] = [{"solution":n.solution,"passT_rate":n.passT_rate,"prob":n.prob} for n in chosen_nodes]
            time_record.append({"cir":cir,"model_inference_time":model_inference_time,"run_solutions_time":run_solutions_time,"choose_solution_time":choose_solution_time})
            if stop or cir>=cir_times:
                break
            cir += 1
            gened_nodes = []
            # 使用选择的node进行下一步的debug
            st = time.time()
            len_record = []
            k=1
            #important
            return_sequences = int(sample_num/k)
            if cir==1 and filter_num==5 and sample_num==2:
                return_sequences = 10
            total_output_length = 0
            print_v(f"begin to generate solutions for cir {cir} with {return_sequences} sequences.")
            #feedback
            for i,node in enumerate(chosen_nodes):
                feedback = node.feedbackprompt
                code = start_code + node.solution
                #important
                if feedback_type=="expl":
                    code_expl = gen.gen_code_explanation(model,code,verbose)
                    idx1 = feedback.find(start_code)
                    idx = feedback.index("Feedback:",idx1)
                    feedback = feedback[:idx] + f"Here is a line-by-line explanation of the code:\n{code_expl}\n\n" + feedback[idx:]
                if cir==1 and i<2:
                    print_with_tag(content=feedback,tag="feedback",verbose=verbose)
                #important
                # print_with_tag(content=feedback,tag="feedback",verbose=verbose)
                
                inputs = model.tokenizer(feedback, return_tensors='pt', return_token_type_ids=False)
                input_length = inputs.input_ids.shape[1]
                fix_percent = (fix_input_len*(fix_input_len - 1.0))/(input_length*(input_length - 1.0))
                for _ in range(k):
                    solutions,inference_time= gen.generate_with_feedback(model,feedback,return_sequences=return_sequences,verbose=True)
                    for s in solutions:
                        ans,true_sc,output_length = s[0],s[1],s[2]
                        # 记录每条solution的长度
                        total_output_length += output_length
                        len_record.append((input_length,fix_input_len,fix_percent,output_length,i))
                        #创建node
                        solution = filter_fix_ans(ans, entry_point, start_code)
                        print_with_tag(content=solution,tag="fix solution",verbose=verbose)
                        tmp_node = Node(solution=solution,parent=node,prompt=feedback,prob=true_sc,depth=cir)
                        tmp_node.idx = len(nodes)
                        node.children.append(tmp_node.idx)
                        gened_nodes.append(tmp_node)
                        nodes.append(tmp_node)
            
            #record time and length
            model_inference_time = (time.time()-st)/60
            average_output_length = total_output_length/len(gened_nodes)
            every_token_time= (time.time()-st)/average_output_length
            fix_record.append({"cir":cir,"len_record":len_record,"every_token_time":every_token_time})
            
            print_v(f"run solution time is {run_solutions_time} mins, choose solution time is {choose_solution_time} mins, model inference time is {model_inference_time} mins.")
            print_v(f"average output length is {average_output_length}, every token time is {every_token_time} s.")
        task_end_time = time.time()
        task_time = (task_end_time - task_begin_time)/60
        
        f.write(json.dumps({"task_id": tid,"completion":output_short,"time_record":time_record,"fix_record":fix_record,"step_one_total_time":step_one_total_time,"step_one_tokens_len":(input_tokens_len,output_tokens_len),"task_time":task_time,"internal_tests":internal_tests})+"\n")
        f.flush()
    f.close()
    return