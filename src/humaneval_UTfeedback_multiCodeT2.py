"""2023.11.6
说明：用于UTfeedback，每次生成100个选前10个的代码。和之前不同的是选择剩下来的代码会加入到下次排序。
"""

import transformers, torch
import json
import time
import numpy as np
import sys
sys.path.append("/home/S/hexiaolong/codex/human-eval")
sys.path.append("/home/S/hexiaolong/codex/self-debug")
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import read_problems
from human_eval.execution import run_code_with_output2, check_correctness
from concurrent.futures import ThreadPoolExecutor
from myutils import map_gpu_memory,get_args,code_clean,code_clean2,get_unit_test,prompt_for_64,get_UTfeedback_prompt,filter_fix_ans,get_CODET_point,get_CODET_point2,get_CODET_point3
import os
from collections import defaultdict
from utils.obj import Node
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# prompt file 
prompt_root = "/home/S/hexiaolong/codex/self-debug/data/prompt/"
prompt_file = prompt_root + "prompt_base2.txt"
UTfeedback_file = prompt_root + "prompt_UTfeedback.txt"
# 从问题中提取的unit tests所在的文件
data_root = "/home/S/hexiaolong/codex/self-debug/data/"
ut_file = data_root + "test_from_prompt.jsonl"
# codeT生成的unitTest文件
codeT_test_file = data_root + "test_from_codeT_7b16k_t300_s100.jsonl"
# 存放了最终用来check的unit_tests
true_tests_file = data_root + "test_from_check.jsonl"
# random.seed(1024)
# 用来进行CODET的tests文件
# tests_for_CODET_file = "/home/S/hexiaolong/codex/self-debug/gen_tests_7b16k_num1002.jsonl"
tests_for_CODET_file = "/home/S/hexiaolong/codex/self-debug/try/gen_test_t0.8_topp0.95_sample100_max300_filter.jsonl"

#config file
config_file = "../configs/UTfeedback_config.yaml"

@hydra.main(version_base=None, config_path="../configs/", config_name="UTfeedback_config.yaml")
def main(cfg: DictConfig):
    #读取配置,获取参数
    print(OmegaConf.to_yaml(cfg))
    
    output_file = cfg.output
    model_path = cfg.model_path
    debug_temp = cfg.multi.debug.temperature
    debug_maxLen = cfg.multi.debug.max_new_tokens
    debug_top_k = cfg.multi.debug.top_k
    debug_top_p = cfg.multi.debug.top_p
    debug_do_sample = cfg.multi.debug.do_sample
    debug_num_return_sequences = cfg.multi.debug.num_return_sequences
    sample_num = cfg.multi.sample_num
    full_output_file = output_file.replace(".jsonl","_full.jsonl")
    
    #读取prompt
    with open(prompt_file,"r") as f:
        preflex = f.read()

    with open(UTfeedback_file,"r") as af:
        UTfeedback_promt = af.read()
    #读取unit tests，保存在unit_tests，用来判断程序对错。one_test里保存了每个task的第一个unit test，这个test会用在prompt里。
    base_unit_tests,base_assertions,base_assertion_string = get_unit_test(ut_file)
    unit_tests,assertions,assertion_string = get_unit_test(ut_file)#true_tests_file,chosen_num=10
    
    #读取用来进行CODET的tests
    testcases = {}
    # limit = 5
    with open(tests_for_CODET_file,"r") as f:
        for line in f.readlines():
            testcases = json.loads(line)
            for k,v in testcases.items():
                print(f"task {k} gen {len(v)} testcases")
            # for k,v in data.items():
            #     limited_task_test_cases = [cases_per_sample[:limit] for cases_per_sample in v]
            #     limited_task_test_cases = sum(limited_task_test_cases, [])
            #     print(f"task {k} gen {len(limited_task_test_cases)} testcases")
            #     testcases[k] = limited_task_test_cases
    

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
    model.tie_weights()
    
    #获取solution
    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    # f = open(output_file,"w+",encoding='utf-8')
    fullf = open(full_output_file,"w+",encoding="utf-8")
    f = open(output_file,"w+",encoding="utf-8")
    if f and fullf:
        print(f"open file {output_file} success!")
    for tid in taskids:
        print("get solution for task :",tid)
        num_id = int(tid.split("/")[1])
        if num_id < 136 or num_id > 140:
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
        output_tokens_len = pred.input_ids.shape[1]
        model_inference_time = (time.time()-st)/60
        ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:]
        solution = ans.strip("\n")#.split("```")[0]#.replace("->>","")
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
        # 生成的初始代码
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
        checkp = assertion_string[tid]
        # for io in unit_tests[tid]:
        #     checkp += "assert " + io["tin"] + " == " + io["tout"] + "\n" # checkp把所有unit tests集合到一起用来判断程序对错
        ut = unit_tests[tid]
        run_test = [t["tin"] for t in ut] # 这个是用来执行的test，会返回代码执行它的结果和test_res比较得到UTfeedback
        # test_res = [literal_eval(t["tout"]) for t in ut]
        test_res = [t["tout"] for t in ut]
        feedback_prompt = UTfeedback_promt + assertion_string[tid] + "\n\n# Complete the Python funtion:\n" + tprompt+"### result ###\n```python\n" + start_code + "\n"
        fix_input = tokenizer(feedback_prompt, return_tensors='pt', return_token_type_ids=False)
        print(f"fix input length is {fix_input.input_ids.shape}")
        fix_input_len = fix_input.input_ids.shape[1]
        step_one_total_time = (time.time() - step_one_st)/60
        # 开始生成feedback和新代码的循环
        cir = 0
        output_short = {}
        output_full = {}
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
            # 运行所有的solution得到通过的test数量和得分
            for i,node in enumerate(gened_nodes):
                solution = node.solution
                # 这里通过一次函数调用同时获得simple和UTfeedback，也就是会判断代码是否正确，同时对于出现AssertError的代码会得到其执行的第一个unit test的值。其他Error因为会返回具体的错误信息就不会得到执行的第一个unit test的值。
                with ThreadPoolExecutor(max_workers=1) as executor:
                    args = (problems[tid], (start_code+solution), run_test, checkp, 0.1)
                    future = executor.submit(run_code_with_output2, *args)
                    result = future.result()
                    passed = result["passed"]
                    code_res = result["result"]
                # print(f"task:{tid},cir:{cir},solution:{i},passed:{passed}")
                if passed:
                    stop=True #一个通过，终结循环
                    node.passT_rate = 1.0
                    print("One node passed! Show it and it's parents.")
                    node.show_parents()
                    break
                else:
                    prompt,passn = get_UTfeedback_prompt(feedback_prompt, solution, code_res, run_test, test_res, assertions[tid])
                    node.feedbackprompt = prompt
                    node.passT_rate = passn
            run_solutions_time = (time.time() - st)/60
            print(f"Run all solutions spends {run_solutions_time} mins.")
            
            # 对生成的代码进行排序并选取排序靠前的代码
            choose_start = time.time()
            total_nodes = gened_nodes + left_nodes
            total_unique_nodes = list(set(total_nodes))
            print(f"task:{tid}, cir:{cir}, total nodes:{len(total_nodes)}, total unique nodes:{len(total_unique_nodes)}")
            get_CODET_point2(total_unique_nodes,testcases[tid],tid) #这里是使用去重后的还是不去重的
            sorted_nodes = sorted(total_unique_nodes,key=lambda x: (x.passT_rate,x.CODET_point,x.prob),reverse=True)
            chosen_nodes = sorted_nodes[:sample_num]
            # if len(sorted_nodes) > sample_num:
            left_nodes = sorted_nodes[sample_num:]
            # chosen_nodes = get_CODET_point3(total_unique_nodes,testcases[tid],tid)
            # left_nodes = []
            # for node in total_nodes:
            #     if node in chosen_nodes:
            #         continue
            #     left_nodes.append(node)
            print(f"task {tid} in cir {cir} chooses {len(chosen_nodes)} nodes and left {len(left_nodes)} nodes")
            print(f"chosen nodes idx is {[n.idx for n in chosen_nodes]}")
            print(f"chosen nodes's parent's idx is {[n.parent.idx for n in chosen_nodes if n.parent]}")
            print(f"chosen nodes passT_rates {[n.passT_rate for n in chosen_nodes]}\nprobs are {[n.prob for n in chosen_nodes]}\nCODET point are {[n.CODET_point for n in chosen_nodes]}")
            choose_solution_time = (time.time()-choose_start)/60
            print(f"Choose solution spends {choose_solution_time} mins.")
            
            output_short[cir] = [{"solution":n.solution,"passT_rate":n.passT_rate,"prob":n.prob,"CODET_point":node.CODET_point} for n in chosen_nodes]
            output_full[cir] = [{"solution":n.solution,"passT_rate":n.passT_rate,"prob":n.prob,"CODET_point":node.CODET_point} for n in total_nodes]
            time_record.append({"cir":cir,"model_inference_time":model_inference_time,"run_solutions_time":run_solutions_time,"choose_solution_time":choose_solution_time})
            if stop or cir==10:
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
                # print("feedback prompt's token nums is :",inputs["input_ids"].size())
                inputs = inputs.to('cuda')
                input_length = inputs.input_ids.shape[1]
                print(f"total input length is {input_length} while fix inuput length is {fix_input_len}")
                fix_percent = (fix_input_len*(fix_input_len - 1.0))/(input_length*(input_length - 1.0))
                print(f"fix percent is {fix_percent*100.0}%")
                # fix_percents.append((input_length,fix_input_len,fix_percent,cir,i))
                # for _ in range(sample_num):
                with torch.no_grad():
                    # preds = model.generate(**inputs, max_new_tokens=512, temperature=1.0,top_p=0.95,num_beams=sample_num, do_sample=True,num_return_sequences=sample_num)
                    preds = model.generate(**inputs, max_new_tokens=debug_maxLen, temperature=debug_temp,top_p=debug_top_p, do_sample=debug_do_sample,num_return_sequences=sample_num,return_dict_in_generate=True,output_scores=True)#,temperature=0.4,repetition_penalty=1.1
                    transition_scores = model.compute_transition_scores(
                        preds.sequences, preds.scores, normalize_logits=True
                    ).cpu().numpy()
                for pred,transition_score in zip(preds["sequences"],transition_scores):
                    gen_tokens = pred[input_length:].cpu()
                    valid = np.isfinite(transition_score)
                    tsc = transition_score[valid]
                    output_length = input_length + tsc.shape[-1]
                    sc = np.sum(tsc,axis=-1)/output_length
                    true_sc = np.exp(sc)
                    #print(f"gen_length: {tsc.shape[-1]},output_length: {output_length}, tsc: {sc}, true_sc: {true_sc}")
                    fix_percents.append((input_length,fix_input_len,fix_percent,output_length,i))
                    ans = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    solution = filter_fix_ans(ans, entry_point, start_code)
                    tmp_node = Node(solution=solution,parent=node,prompt=feedback,prob=true_sc,depth=cir)
                    tmp_node.idx = len(nodes)
                    node.children.append(tmp_node)
                    gened_nodes.append(tmp_node)
                    nodes.append(tmp_node)
            fix_record.append({"cir":cir,"fix_percents":fix_percents})
            print(f"fix record len: {len(fix_record)}")
            print(f"cir {cir} gened {len(gened_nodes)} solutions. Total nodes num is {len(nodes)}")
            model_inference_time = (time.time()-st)/60
            print(f"Total model inference spends {model_inference_time} mins.")
        start_write = time.time()
        print(f"time_record:{time_record}\nfix_record:{fix_record}")
        f.write(json.dumps({"task_id": tid,"completion":output_short,"time_record":time_record,"fix_record":fix_record,"step_one_total_time":step_one_total_time,"step_one_tokens_len":(input_tokens_len,output_tokens_len)})+"\n")
        f.flush()
        fullf.write(json.dumps({"task_id": tid,"completion":output_full})+"\n")
        fullf.flush()
        write_time = (time.time()-start_write)/60
        print(f"Write results spends {write_time} mins.")
    f.close()
    fullf.close()


if __name__ == "__main__":
    main()

    


