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
from myutils import map_gpu_memory,get_args,code_clean,code_clean2,get_unit_test,prompt_for_64,get_UTfeedback_prompt,filter_fix_ans
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
    unit_tests,assertions,assertion_string = get_unit_test(ut_file)#,chosen_num=10

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
        if num_id < 147 or num_id > 166:
            continue
        tprompt = problems[tid]["prompt"]
        if tid == "HumanEval/64":
            tprompt = prompt_for_64
        problem = get_one_complication(tprompt,base_assertion_string[tid])
        input_len = len(problem)
        inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
        inputs = inputs.to('cuda')
        pred = model.generate(**inputs, max_new_tokens=512, temperature=0)#,temperature=0.4,repetition_penalty=1.1
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
        # 开始生成feedback和新代码的循环
        cir = 0
        output_short = {}
        output_full = {}
        node1 = Node(solution,prompt=problem,depth=cir,feedbackprompt=feedback_prompt)
        nodes = [node1]
        gened_nodes = [node1]
        chosen_nodes = [node1]
        left_nodes = []
        # output_short[0] = chosen_solution
        while True:
            stop = False
            st = time.time()
            # 运行所有的solution得到通过的test数量
            for i,node in enumerate(gened_nodes):
                solution = node.solution
                # 这里通过一次函数调用同时获得simple和UTfeedback，也就是会判断代码是否正确，同时对于出现AssertError的代码会得到其执行的第一个unit test的值。其他Error因为会返回具体的错误信息就不会得到执行的第一个unit test的值。
                with ThreadPoolExecutor(max_workers=1) as executor:
                    args = (problems[tid], (start_code+solution), run_test, checkp, 3.0)
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
                    # print("---------------------feeedback prompt---------------------------")
                    # print(prompt)#[len(UTfeedback_promt):]
                    # print("----------------------------------------------------------------")
                    # print("pass rate: ",passn)
                    node.feedbackprompt = prompt
                    node.passT_rate = passn
            print(f"Run all solutions spends {(time.time()-st)/60} mins.")
            total_nodes = gened_nodes + left_nodes
            total_unique_nodes = list(set(total_nodes))
            print(f"task:{tid}, cir:{cir}, total nodes:{len(total_nodes)}, total unique nodes:{len(total_unique_nodes)}")
            sorted_nodes = sorted(total_unique_nodes,key=lambda x: (x.passT_rate,x.prob),reverse=True)
            chosen_nodes = sorted_nodes[:sample_num]
            # if len(sorted_nodes) > sample_num:
            left_nodes = sorted_nodes[sample_num:]
            print(f"task {tid} in cir {cir} chooses {len(chosen_nodes)} nodes and left {len(left_nodes)} nodes")
            print(f"chosen nodes passT_rates {[n.passT_rate for n in chosen_nodes]}\nprobs are {[n.prob for n in chosen_nodes]}")
            output_short[cir] = [{"solution":n.solution,"passT_rate":n.passT_rate,"prob":n.prob} for n in chosen_nodes]
            output_full[cir] = [{"solution":n.solution,"passT_rate":n.passT_rate,"prob":n.prob,"prompt":n.prompt} for n in chosen_nodes]
            if stop or cir==10:
                break
            cir += 1
            gened_nodes = []
            st = time.time()
            for i,node in enumerate(chosen_nodes):
                feedbackprompt = node.feedbackprompt
                input_len = len(feedbackprompt)
                inputs = tokenizer(feedbackprompt, return_tensors='pt', return_token_type_ids=False)
                # print("feedback prompt's token nums is :",inputs["input_ids"].size())
                inputs = inputs.to('cuda')
                input_length = inputs.input_ids.shape[1]
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
                    # print(f"gen_length: {tsc.shape[-1]},output_length: {output_length}, tsc: {sc}, true_sc: {true_sc}")
                    ans = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    solution = filter_fix_ans(ans, entry_point, start_code)
                    tmp_node = Node(solution=solution,parent=node,prompt=feedbackprompt,prob=true_sc,depth=cir)
                    node.children.append(tmp_node)
                    gened_nodes.append(tmp_node)
                    nodes.append(tmp_node)
            print(f"Total model inference spends {(time.time()-st)/60} mins.")
            print(f"cir {cir} gened {len(gened_nodes)} solutions.Total nodes num is {len(nodes)}")
        f.write(json.dumps({"task_id": tid,"completion":output_short})+"\n")
        f.flush()
        fullf.write(json.dumps({"task_id": tid,"completion":output_full})+"\n")
        fullf.flush()
    f.close()
    fullf.close()


if __name__ == "__main__":
    main()

    


