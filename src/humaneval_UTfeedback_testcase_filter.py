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
from human_eval.execution import run_code_with_test_result,run_code_with_output2,check_test_correctness
from myutils import setup_seed,map_gpu_memory,code_clean2,get_unit_test,prompt_for_64,filter_fix_ans,get_CODET_point_v1,get_CODET_point_v3,get_pass_rate,get_UTfeedback_prompt_v1,get_UTfeedback_prompt
from utils.obj import Node

"""2023.3.5
说明：重构了流程，在self-debug的过程中增加了对测试用例的筛选。
"""

# prompt file 
prompt_root = "/home/S/hexiaolong/codex/self-debug/data/prompt/"
prompt_file = prompt_root + "prompt_base2.txt"
UTfeedback_file = prompt_root + "prompt_UTfeedback.txt"
# test file
data_root = "/home/S/hexiaolong/codex/self-debug/data/"
ut_file = data_root + "test_from_prompt.jsonl"# 从问题中提取的unit tests所在的文件
true_tests_file = data_root + "test_from_check.jsonl"# 存放了最终用来check的unit_tests
tests_for_CODET_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_rm_final5.jsonl"# 用来进行CODET的tests文件


@hydra.main(version_base=None, config_path="../configs/", config_name="UTfeedback_config.yaml")
def main(cfg: DictConfig):
    #读取配置,获取参数
    print(OmegaConf.to_yaml(cfg))
    output_file = cfg.output
    model_path = cfg.model_path
    debug_temp = cfg.multi.debug.temperature
    debug_maxLen = cfg.multi.debug.max_new_tokens
    debug_top_p = cfg.multi.debug.top_p
    debug_do_sample = cfg.multi.debug.do_sample
    debug_num_return_sequences = cfg.multi.debug.num_return_sequences
    sample_num = cfg.multi.sample_num
    
    # setup seed 
    setup_seed(2024)
    # 控制是否加入CODET分数
    with_CODET_Point = True
    # 是否保留上轮迭代剩余的solution
    with_solution_before = True

    #读取prompt
    with open(prompt_file,"r") as f:
        preflex = f.read()

    with open(UTfeedback_file,"r") as af:
        UTfeedback_promt = af.read()
    
    #读取unit tests
    base_unit_tests,base_assertions,base_assertion_string = get_unit_test(ut_file)
    unit_tests,assertions,assertion_string = get_unit_test(ut_file)
    
    #读取用来进行CODET的tests
    testcases = {}
    with open(tests_for_CODET_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            for k,v in data.items():
                testcases[k] = v
    
    # 构成生成初始代码的prompt
    def get_one_complication(tprompt,unit_test):#为模型的输入添加前缀串，看能否提高准确度。目前直接返回原串。
        res = preflex + unit_test + "\n\n# Complete the Python funtion:\n" + tprompt + "### result ###\n"
        return res

    #为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
    max_memory_mapping = map_gpu_memory(used_gpu=[])

    #加载模型
    print("load model from ",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True)#, use_safetensors=True
    
    #获取solution
    problems = read_problems()
    taskids = list(problems.keys())
    f = open(output_file,"w+",encoding="utf-8")
    if f:
        print(f"open file {output_file} success!")
    for tid in taskids:
        print(f"get solution for task : {tid} with {len(unit_tests[tid])} tests.")
        num_id = int(tid.split("/")[1])
        if num_id < 0 or num_id > 163:
            continue
        step_one_st = time.time()
        tprompt = problems[tid]["prompt"]
        if tid == "HumanEval/64":
            tprompt = prompt_for_64
        problem = get_one_complication(tprompt,base_assertion_string[tid])
        input_len = len(problem)
        inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
        input_tokens_len = inputs.input_ids.shape[1]
        print(f"In generate step, the input tokens shape is {inputs.input_ids.shape}")
        inputs = inputs.to('cuda')
        st = time.time()
        pred = model.generate(**inputs, max_new_tokens=512, temperature=0)#,temperature=0.4,repetition_penalty=1.1
        output_tokens_len = pred.shape[1]
        print(f"output_tokens_len:{output_tokens_len}")
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
            print(solution.split("```"))
        if solution.startswith("python"):
            solution = solution[6:]
        solution = solution.strip("\n")
        # 去除函数头和注释
        entry_point = "def " + problems[tid]["entry_point"]
        solution = code_clean2(code=solution,entry_point=entry_point)
        # 打印生成的初始代码
        print("+++++++++++filter solution++++++++++++++++++")
        print(solution)
        print("++++++++++++++++++++++++++++++++++++++++++++")
        # 去掉problems[tid]["prompt"]中的注释和空行
        start_code = ""
        for line in tprompt.split("\n"):
            if line=="":
                continue
            if entry_point in line:
                start_code += line + "\n"
                break
            start_code += line + "\n"
        print("=====start code===========")
        print(start_code)
        
        #判断代码是否正确，是否提前结束
        is_solved = False
        try:
            result  = check_test_correctness(start_code+solution+"\n"+assertion_string[tid])
            is_solved = result["passed"]
        except:
            is_solved = False
        if is_solved:
            f.write(json.dumps({"task_id": tid,"completion":{0:[{"solution":solution}]},"time_record":[],"fix_record":[],"step_one_total_time":step_one_total_time,"step_one_tokens_len":(input_tokens_len,output_tokens_len)})+"\n")
            f.flush()
            continue
        
        # debug 准备
        feedback_prompt = UTfeedback_promt + assertion_string[tid] + "\n\n# Complete the Python funtion:\n" + tprompt+"### result ###\n```python\n" + start_code + "\n"
        fix_input = tokenizer(UTfeedback_promt, return_tensors='pt', return_token_type_ids=False)
        print(f"fix input length is {fix_input.input_ids.shape}")
        fix_input_len = fix_input.input_ids.shape[1]
        step_one_total_time = (time.time() - step_one_st)/60
        
        # 开始生成feedback和新代码的循环
        cir = 0
        output_short = {}
        node1 = Node(solution,prompt=problem,depth=cir,feedbackprompt=feedback_prompt)
        nodes,gened_nodes,chosen_nodes = [node1],[node1],[node1]
        left_nodes,time_record,fix_record = [],[],[]
        prompt_test_len = len(base_assertions[tid])
        chosen_testcase_num =  max(0,10 - prompt_test_len)#8 - prompt_test_len#max(prompt_test_len,8 - prompt_test_len)#
        
        chosen_testcase = base_assertions[tid]
        while True:
            # 对生成的代码进行排序并选取排序靠前的代码
            choose_start = time.time()
            if with_solution_before:
                total_nodes = gened_nodes + left_nodes
            else:
                total_nodes = gened_nodes
            total_unique_nodes = list(set(total_nodes))
            print(f"task:{tid}, cir:{cir}, total nodes:{len(total_nodes)}, total unique nodes:{len(total_unique_nodes)}")
            if with_CODET_Point:
                sorted_group,chosen_testcase_id  = get_CODET_point_v3(total_nodes,testcases[tid],tid,chosen_num=chosen_testcase_num,sort_len=True) #这里是使用去重后的还是不去重的,sort_len=True
                if chosen_testcase_id!=[]:
                    next_chosen_testcase = base_assertions[tid] + [testcases[tid][x] for x in chosen_testcase_id]
                else:
                    print("chosen_testcase_id is empty!")
                print(f"{len(next_chosen_testcase)} chosen_testcase:")
                print("\n".join(next_chosen_testcase))
                pass_testcase_list = [ list(x[1]) for x in sorted_group]
                get_pass_rate(total_nodes,base_assertions[tid],tid) # pass rate store in CODET_pass_rate
                sorted_nodes = sorted(total_unique_nodes,key=lambda x: (x.CODET_pass_rate,x.prob),reverse=True)
            else:
                sorted_nodes = sorted(total_unique_nodes,key=lambda x: (x.passT_rate,x.prob),reverse=True)
            chosen_nodes = sorted_nodes[:sample_num]
            left_nodes = sorted_nodes[sample_num:]
            choose_solution_time = (time.time()-choose_start)/60
            
            # 对选择的代码进行拓展
            st = time.time()
            fix_percents = []
            k=1
            return_sequences = 10
            for i,node in enumerate(chosen_nodes):
                feedback = node.feedbackprompt
                input_len = len(feedback)
                inputs = tokenizer(feedback, return_tensors='pt', return_token_type_ids=False)
                input_length = inputs.input_ids.shape[1]
                print(f"total input length is {input_length} while fix inuput length is {fix_input_len}")
                inputs = inputs.to('cuda')
                fix_percent = 0 # (fix_input_len*(fix_input_len - 1.0))/(input_length*(input_length - 1.0))
                for _ in range(k):
                    try:
                        with torch.inference_mode():
                            preds = model.generate(**inputs, max_new_tokens=debug_maxLen, temperature=debug_temp,top_p=debug_top_p, do_sample=debug_do_sample,return_dict_in_generate=True,output_scores=True,num_return_sequences=return_sequences)#,temperature=0.4,repetition_penalty=1.1
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
                            node.children.append(tmp_node.idx)
                            gened_nodes.append(tmp_node)
                            nodes.append(tmp_node)
                    except BaseException as e:
                        print(f"model inference failed with {e}")
                    except:
                        print("model inference failed!")
            model_inference_time = (time.time()-st)/60
            
            st = time.time()
            stop = False
            # 运行所有的solution得到通过的test数量和得分
            run_test = [t.replace("assert ","").split("==")[0].strip() for t in chosen_testcase]
            print(f"run_test:{run_test}")
            feedback_prompt = UTfeedback_promt + "\n".join(chosen_testcase) + "\n\n# Complete the Python funtion:\n" + tprompt+"### result ###\n```python\n" + start_code + "\n"
            get_pass_rate(gened_nodes,base_assertions[tid],tid)
            for i,node in enumerate(gened_nodes):
                solution = node.solution
                # 这里通过一次函数调用同时获得simple和UTfeedback，也就是会判断代码是否正确，同时对于出现AssertError的代码会得到其执行的第一个unit test的值。其他Error因为会返回具体的错误信息就不会得到执行的第一个unit test的值。
                with ThreadPoolExecutor(max_workers=1) as executor:
                    args = (problems[tid], (start_code+solution), run_test, chosen_testcase, 0.1)
                    future = executor.submit(run_code_with_test_result, *args)
                    result = future.result()
                    passed = result["passed"]
                    final_res = result["result"]
                prompt,passn,pass_tests = get_UTfeedback_prompt_v1(feedback_prompt, solution, passed, final_res, run_test, chosen_testcase)
                node.feedbackprompt = prompt
                node.passT_rate = node.CODET_pass_rate
                node.pass_ut_num = pass_tests
                node.total_ut_num = len(run_test)
                assert len(run_test) == len(chosen_testcase)
                if node.CODET_pass_rate >= 1.0:
                    print("passn:",node.CODET_pass_rate)
                    stop=True
                    print("One node passed! Show it and it's parents.")
                    node.show_parents()
                    break