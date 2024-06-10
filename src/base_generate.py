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
from myutils import setup_seed,make_printv,print_with_tag,code_clean2,get_unit_test,prompt_for_64,filter_fix_ans,get_CODET_point_v1,get_CODET_point_v3,get_pass_rate,get_UTfeedback_prompt_v1,get_UTfeedback_prompt,start_code_extract
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

def generate_base_complication_batch(self, model, prompt, unit_tests, entry_point, record_time = False, verbose = False):
        setup_seed(2024)
        #prepare the prompt
        prompt = self.get_one_complication(prompt,unit_tests)
        # print_with_tag("base complication prompt",prompt,verbose=verbose)
        prompts = [prompt,prompt,prompt,prompt,prompt]
        input_len = len(prompt)
        inputs = model.tokenizer(prompts, return_tensors='pt', return_token_type_ids=False)
        input_tokens_len = inputs.input_ids.shape[1]
        print(inputs.input_ids.shape)
        inputs = inputs.to('cuda')
        
        #generate the solution
        st = time.time()
        pred = model.model.generate(**inputs, max_new_tokens=512, temperature=0,pad_token_id=model.tokenizer.eos_token_id)#,temperature=0.4,repetition_penalty=1.1
        model_inference_time = (time.time()-st)/60
        print(pred.shape)
        output_tokens_len = pred.shape[1]
        ans = model.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:].strip("\n")
        # print_with_tag("base complication origin output",ans,verbose=verbose)
        solution = self.code_extract(ans)
        solution = code_clean2(code=solution,entry_point=entry_point)
        solutions = [solution]
        #log the time and length
        if verbose:
            print(f"Model inference time is {model_inference_time} minutes")
            print(f"In generate step, the input tokens shape is {input_tokens_len}, the output tokens shape is {output_tokens_len}")
        
        #return the solution
        if record_time:
            return prompt,solutions, model_inference_time, input_tokens_len, output_tokens_len
        else:
            return prompt,solutions


def run_base_generate(
    dataset:list,
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
    
    
    #获取solution
    
    print_v(f"start to generate solutions for {len(dataset)} tasks.")
    for data in dataset:
        
        tid = data["task_id"]
        if data.get('prompt_tests', []) == []:
            # gened_testcase = gen.gen_tests(model,data,num=6,verbose=True)[:6]
            gened_testcase = []
            base_assertion_string = "\n".join(gened_testcase)
            assertions = gened_testcase
            if base_assertion_string == "":
                print_v(f"task {tid} has no test case, skip it.")
        else :
            assertions = data["prompt_tests"]
            base_assertion_string = "\n".join(assertions)
            assertion_string = "\n".join(assertions)
            
            
        task_begin_time = time.time()
        print_v(f"get solution for task : {tid} with {len(assertions)} tests.")
        step_one_st = time.time()
        tprompt = data["prompt"]
        entry_point = "def " + data["entry_point"]
        base_prompt,solutions,model_inference_time,input_tokens_len, output_tokens_len = \
            gen.generate_base_complication_batch(model,tprompt,base_assertion_string,
                                             entry_point,
                                             record_time=True,verbose=verbose)
        
    