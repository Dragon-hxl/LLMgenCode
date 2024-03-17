import torch
import json
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import hydra
from omegaconf import DictConfig, OmegaConf

import faulthandler
faulthandler.enable()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import sys
sys.path.append("/home/S/hexiaolong/codex/self-debug")
sys.path.append("/home/S/hexiaolong/codex/self-debug/humaneval")
from human_eval.data import read_problems
from human_eval.execution import run_code_with_test_result,run_code_with_output2
from myutils import setup_seed,make_printv,map_gpu_memory,code_clean2,get_unit_test,prompt_for_64,filter_fix_ans,get_pass_rate,get_CODET_point_v1,get_CODET_point_v2,get_UTfeedback_prompt_v1,get_UTfeedback_prompt_v2,get_UTfeedback_prompt,load_testcase
from utils.obj import Node

"""2023.12.16
说明：用于UTfeedback，每次生成100个选前10个的代码。和之前不同的是选择剩下来的代码会加入到下次排序。加入CODET作为评分依据
"""
# 可以被CODET3.py文件覆盖

# prompt file 
prompt_root = "/home/S/hexiaolong/codex/self-debug/data/prompt/"
prompt_file = prompt_root + "prompt_base2.txt"
UTfeedback_file = prompt_root + "prompt_UTfeedback.txt"
# test file
data_root = "/home/S/hexiaolong/codex/self-debug/data/"
ut_file = data_root + "test_from_prompt.jsonl"# 从问题中提取的unit tests所在的文件
true_tests_file = data_root + "test_from_check.jsonl"# 存放了最终用来check的unit_tests
tests_for_CODET_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_rm.jsonl"# 用来进行CODET的tests文件
CODET_test_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_uniform.jsonl"
gened_testcase_for_feedback = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_mix05_10ios.jsonl"
# 控制是否加入CODET分数
with_CODET_Point = True


@hydra.main(version_base=None, config_path="../configs/", config_name="UTfeedback_config.yaml")
def main(cfg: DictConfig):
    #读取配置,获取参数
    print(OmegaConf.to_yaml(cfg))
    
    output_file = cfg.output
    model_path = cfg.model_path
    debug_temp = cfg.multi.debug.temperature
    debug_maxLen = cfg.multi.debug.max_new_tokens
    # debug_top_k = cfg.multi.debug.top_k
    debug_top_p = cfg.multi.debug.top_p
    debug_do_sample = cfg.multi.debug.do_sample
    # debug_num_return_sequences = cfg.multi.debug.num_return_sequences
    sample_num = cfg.multi.sample_num
    
    setup_seed(2024)
    
    #读取prompt
    with open(prompt_file,"r") as f:
        preflex = f.read()

    with open(UTfeedback_file,"r") as af:
        UTfeedback_promt = af.read()
        
    printv = make_printv(True)
    
    #读取unit tests
    base_unit_tests,base_assertions,base_assertion_string = get_unit_test(ut_file)
    unit_tests,assertions,assertion_string = get_unit_test(gened_testcase_for_feedback)#true_tests_file,chosen_num=10
    for tid in list(unit_tests.keys()):
        if len(unit_tests[tid]) == 0:
            unit_tests[tid] = base_unit_tests[tid]
            assertions[tid] = base_assertions[tid]
            assertion_string[tid] = base_assertion_string[tid]
        
    #读取用来进行CODET的tests
    testcases = load_testcase(tests_for_CODET_file,type=0)
    

    # 构成生成初始代码的prompt
    def get_one_complication(tprompt,unit_test):#为模型的输入添加前缀串，看能否提高准确度。目前直接返回原串。
        res = preflex + unit_test + "\n\n# Complete the Python funtion:\n" + tprompt + "### result ###\n"
        return res

    #为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
    max_memory_mapping = map_gpu_memory(used_gpu=[])

    #加载模型
    printv("load model from ",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="balanced", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True)#, use_safetensors=True
    model.eval()
    # model.tie_weights()
    
    #获取solution
    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    # fullf = open(full_output_file,"w+",encoding="utf-8")
    f = open(output_file,"w+",encoding="utf-8")
    if f :#and fullf:
        printv(f"open file {output_file} success!")
    # special_task = [148, 154, 155, 156]# 还不通过的,94,74,87,103,105,106,110,111,112,113,
    # special_task2 = [94]#133,68,126,129,
    # tT CODETPointfirst    19,20
    # mix05 : [9,21,26,33,34,43,44,47,88,94,96,105,115,121,125,126,129,146,148,149,150,153,163]
    special_task = [146,148,153,163]# 67,94,126,130,
    ignore_task = [0, 2, 4, 7, 12, 15, 23, 28, 29, 30, 31, 34, 35, 40, 42, 43, 44, 45, 48, 51, 52, 53, 55, 58, 60, 124, 162,    94,105,115,129]
    for tid in taskids:
        printv(f"get solution for task : {tid} with {len(unit_tests[tid])} tests.")
        num_id = int(tid.split("/")[1])
        if num_id < 117 or num_id > 128 or num_id in ignore_task:
            continue
        # if num_id not in special_task:
        #     continue
        step_one_st = time.time()
        tprompt = problems[tid]["prompt"]
        if tid == "HumanEval/64":
            tprompt = prompt_for_64
        problem = get_one_complication(tprompt,base_assertion_string[tid])
        input_len = len(problem)
        inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
        input_tokens_len = inputs.input_ids.shape[1]
        printv(f"In generate step, the input tokens shape is {inputs.input_ids.shape}")
        inputs = inputs.to('cuda')
        st = time.time()
        pred = model.generate(**inputs, max_new_tokens=512, temperature=0)#,temperature=0.4,repetition_penalty=1.1
        output_tokens_len = pred.shape[1]
        printv(f"output_tokens_len:{output_tokens_len}")
        model_inference_time = (time.time()-st)/60
        ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:]
        solution = ans.strip("\n")
        # 截取程序
        idx2 = solution.find("### Task End ###")
        if idx2 != -1:
            solution = solution[:idx2-1] #这里的减1是为了去掉前面的换行
        if len(solution.split("```"))>1:
            solution = solution.split("```")[1]
        else:
            printv(solution.split("```"))
        if solution.startswith("python"):
            solution = solution[6:]
        solution = solution.strip("\n")
        # 去除函数头和注释
        entry_point = "def " + problems[tid]["entry_point"]
        solution = code_clean2(code=solution,entry_point=entry_point)
        # 生成的初始代码
        printv("+++++++++++filter solution++++++++++++++++++")
        printv(solution)
        printv("++++++++++++++++++++++++++++++++++++++++++++")
        # 去掉problems[tid]["prompt"]中的注释和空行
        start_code = ""
        for line in tprompt.split("\n"):
            if line=="":
                continue
            if entry_point in line:
                start_code += line + "\n"
                break
            start_code += line + "\n"
        printv("=====start code===========")
        printv(start_code)
        # debug 准备
        checkp = assertion_string[tid]
        ut = unit_tests[tid]
        run_test = [t["tin"] for t in ut] # 这个是用来执行的test，会返回代码执行它的结果和test_res比较得到UTfeedback
        test_res = [t["tout"] for t in ut]
        feedback_prompt = UTfeedback_promt + assertion_string[tid] + "\n\n# Complete the Python funtion:\n" + tprompt+"### result ###\n```python\n" + start_code + "\n"
        fix_input = tokenizer(feedback_prompt, return_tensors='pt', return_token_type_ids=False)
        printv(f"fix input length is {fix_input.input_ids.shape}")
        fix_input_len = fix_input.input_ids.shape[1]
        step_one_total_time = (time.time() - step_one_st)/60
        # 开始生成feedback和新代码的循环
        cir = 0
        output_short = {}
        # output_full = {}
        node1 = Node(solution,prompt=problem,depth=cir,feedbackprompt=feedback_prompt)
        nodes = [node1]
        gened_nodes = [node1]
        chosen_nodes = [node1]
        left_nodes = []
        time_record = []
        fix_record = []
        # output_short[0] = chosen_solution
        while True:
            st = time.time()
            stop = False
            get_pass_rate(gened_nodes,base_assertions[tid],tid)
            # 运行所有的solution得到通过的test数量和得分
            for i,node in enumerate(gened_nodes):
                solution = node.solution
                # 这里通过一次函数调用同时获得simple和UTfeedback，也就是会判断代码是否正确，同时对于出现AssertError的代码会得到其执行的第一个unit test的值。其他Error因为会返回具体的错误信息就不会得到执行的第一个unit test的值。
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    args = (problems[tid], (start_code+solution), run_test, assertions[tid], 0.1)
                    future = executor.submit(run_code_with_test_result, *args)
                    result = future.result()
                    passed = result["passed"]
                    final_res = result["result"]
                prompt,passn,pass_tests = get_UTfeedback_prompt_v1(feedback_prompt, solution, passed, final_res, run_test, assertions[tid])
                passn = node.CODET_pass_rate
                node.feedbackprompt = prompt
                node.passT_rate = passn
                node.pass_ut_num = pass_tests
                node.total_ut_num = len(run_test)
                if passn >= 1.0:
                    stop=True
                    printv("One node passed! Show it and it's parents.")
                    node.show_parents()
                    break
            run_solutions_time = (time.time() - st)/60
            printv(f"Run all solutions spends {run_solutions_time} mins.")
            
            # 对生成的代码进行排序并选取排序靠前的代码
            choose_start = time.time()
            total_nodes = gened_nodes + left_nodes
            total_unique_nodes = list(set(total_nodes))
            printv(f"task:{tid}, cir:{cir}, total nodes:{len(total_nodes)}, total unique nodes:{len(total_unique_nodes)}")
            if with_CODET_Point:
                 # pass rate store in CODET_pass_rate
                sorted_nodes = sorted(total_unique_nodes,key=lambda x: (x.CODET_pass_rate,x.prob),reverse=True)
            else:
                sorted_nodes = sorted(total_unique_nodes,key=lambda x: (x.passT_rate,x.prob),reverse=True)
            chosen_nodes = sorted_nodes[:sample_num]
            left_nodes = sorted_nodes[sample_num:]
            printv(f"task {tid} in cir {cir} chooses {len(chosen_nodes)} nodes and left {len(left_nodes)} nodes")
            printv(f"chosen nodes idx is {[n.idx for n in chosen_nodes]}")
            printv(f"chosen nodes's parent's idx is {[n.parent.idx for n in chosen_nodes if n.parent]}")
            printv(f"chosen nodes passT_rates {[n.passT_rate for n in chosen_nodes]}\nprobs are {[n.prob for n in chosen_nodes]}\nCODET point are {[n.CODET_point for n in chosen_nodes]}\nCODET_pass_rate are {[n.CODET_pass_rate for n in chosen_nodes]}")
            choose_solution_time = (time.time()-choose_start)/60
            printv(f"Choose solution spends {choose_solution_time} mins.")
            
            output_short[cir] = [{"solution":n.solution,"passT_rate":n.passT_rate,"prob":n.prob} for n in chosen_nodes]
            time_record.append({"cir":cir,"model_inference_time":model_inference_time,"run_solutions_time":run_solutions_time,"choose_solution_time":choose_solution_time})
            if stop or cir>=10:
                break
            cir += 1
            gened_nodes = []
            # 使用选择的node进行下一步的debug
            st = time.time()
            fix_percents = []
            for i,node in enumerate(chosen_nodes):
                feedback = node.feedbackprompt
                input_len = len(feedback)
                inputs = tokenizer(feedback, return_tensors='pt', return_token_type_ids=False)
                input_length = inputs.input_ids.shape[1]
                printv(f"total input length is {input_length} while fix inuput length is {fix_input_len}")
                inputs = inputs.to('cuda')
                fix_percent = 0 # (fix_input_len*(fix_input_len - 1.0))/(input_length*(input_length - 1.0))
                for _ in range(2):
                    with torch.inference_mode():
                        preds = model.generate(**inputs, max_new_tokens=debug_maxLen, temperature=debug_temp,top_p=debug_top_p, do_sample=debug_do_sample,return_dict_in_generate=True,output_scores=True,num_return_sequences=5)#,temperature=0.4,repetition_penalty=1.1
                        transition_scores = model.compute_transition_scores(
                            preds.sequences, preds.scores, normalize_logits=True
                        ).cpu().numpy()
                    for pred,transition_score in zip(preds["sequences"],transition_scores):
                        # 计算生成概率
                        gen_tokens = pred[input_length:].cpu()
                        valid = np.isfinite(transition_score)
                        tsc = transition_score[valid]
                        output_length = input_length + tsc.shape[-1]
                        sc = np.sum(tsc,axis=-1)/output_length
                        true_sc = np.exp(sc)
                        
                        # 记录每条solution的长度
                        fix_percents.append((input_length,fix_input_len,fix_percent,output_length,i))
                        
                        #创建node
                        ans = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                        solution = filter_fix_ans(ans, entry_point, start_code)
                        tmp_node = Node(solution=solution,parent=node,prompt=feedback,prob=true_sc,depth=cir)
                        tmp_node.idx = len(nodes)
                        node.children.append(tmp_node)
                        gened_nodes.append(tmp_node)
                        nodes.append(tmp_node)
            fix_record.append({"cir":cir,"fix_percents":fix_percents})
            printv(f"cir {cir} gened {len(gened_nodes)} solutions. Total nodes num is {len(nodes)}")
            model_inference_time = (time.time()-st)/60
            printv(f"Total model inference spends {model_inference_time} mins.")
        start_write = time.time()
        f.write(json.dumps({"task_id": tid,"completion":output_short,"time_record":time_record,"fix_record":fix_record,"step_one_total_time":step_one_total_time,"step_one_tokens_len":(input_tokens_len,output_tokens_len)})+"\n")
        f.flush()
        write_time = (time.time()-start_write)/60
        printv(f"Write results spends {write_time} mins.")
    f.close()


if __name__ == "__main__":
    main()

    


