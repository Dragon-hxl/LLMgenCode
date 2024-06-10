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

def testcase_generate(
    dataset:dict,
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
    exe = py_executor()
    offline = False
    
    print_v(f"model_path : {model_path}")
    print_v(f"sample_num : {sample_num}")
    print_v(f"cir_times : {cir_times}")
    print_v("Run testcase generation.")
    

    #打开输出文件
    f = open(output_file,"w+",encoding="utf-8")
    #开始生成solution
    for data in dataset:
        tid = data["task_id"]
        
        
        print_v("Gen testcase by model.")
        gen_tests_st = time.time()
        task_gened_testcase = gen.gen_tests_sort_by_prob(model,data,num=150,verbose=verbose)[:150]
        gen_tests_time = (time.time() - gen_tests_st)/60
        print_v(f"Gen {len(task_gened_testcase)} tests for task {tid}, use {gen_tests_time} mins.")
        
        f.write(json.dumps({tid:task_gened_testcase})+"\n")
        
    return 0