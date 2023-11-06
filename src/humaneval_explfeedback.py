import transformers, torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import read_problems
from human_eval.execution import check_test_correctness, run_code_with_output2
from concurrent.futures import ThreadPoolExecutor
from myutils import map_gpu_memory,code_clean2,get_args,code_clean
import os
from collections import defaultdict
os.environ["TOKENIZERS_PARALLELISM"] = "true"

prompt_file = "prompt_base2.txt"
explfeedback_file = "prompt_explfeedback2.txt"
gen_expl_file = "prompt_code_expl2.txt"
ut_file = "tests_from_prompt.jsonl"

if __name__ == "__main__":
    args = get_args()
    output_file = args.output
    model_path = args.model_path
    verbose = True#args.verbose
    unit_tests = {}
    assertions = defaultdict(list)
    assertion_strings = {}

    with open(prompt_file,"r") as f:
        preflex = f.read()

    with open(explfeedback_file,"r") as af:
        explfeedback_prompt = af.read()

    with open(gen_expl_file,"r") as af:
        genexpl_prompt = af.read()

    with open(ut_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["tid"]
            ios = data["ios"]
            unit_tests[tid]=ios
            for io in ios:
                assertions[tid].append("assert "+io["tin"] + " == " + io["tout"])
            assertion_strings[tid] = "\n".join(assertions[tid])


    def get_one_complication(problem,unit_test):#为模型的输入添加前缀串，看能否提高准确度。目前直接返回原串。
        res = preflex + unit_test + "\n\n# Complete the Python funtion:\n" + problem["prompt"] + "### result ###\n"
        # print("=================unit test in prompt==================")
        # print(unit_test)
        # print("=================end==================")
        return res

    #为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
    max_memory_mapping = map_gpu_memory(used_gpu=[])

    #加载vicuna模型
    print("load model from ",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)#

    #获取solution
    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    f = open(output_file,"w+",encoding='utf-8')
    for tid in taskids:
        print("get solution for task :",tid)
        # tid_int = int(tid.split("/")[1])
        # if tid_int<156:
        #     continue
        problem = get_one_complication(problems[tid],assertion_strings[tid])
        input_len = len(problem)
        inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
        inputs = inputs.to('cuda')
        pred = model.generate(**inputs, max_new_tokens=512, temperature=0)#,temperature=0.4,repetition_penalty=1.1
        ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:]
        solution = ans.strip("\n")#.split("```")[0]#.replace("->>","")
        # print("=========origin solution====================")
        # print(solution)
        # print("============================================")
        # 截取程序
        idx2 = solution.find("### Task End ###")
        if idx2 != -1:
            solution = solution[:idx2-1]
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
        print("+++++++++++filter solution++++++++++++++++++")
        print(solution)
        print("++++++++++++++++++++++++++++++++++++++++++++")
        #self-debug step
        # entry_point = problems[id]["entry_point"]
        start_code = ""
        for line in problems[tid]["prompt"].split("\n"):
            if line=="":
                continue
            if entry_point in line:
                start_code += line + "\n"
                break
            start_code += line + "\n"
        print(f"entry_point: {entry_point}")
        print("=====start code===========")
        print(start_code)
        cir = 0 
        unchanged = 0
        # 构建测试的代码，本应该放在下面的ThreadPoolExecutor下的，因为在循环中保持不变，所以放在这
        checkp = ""
        for io in unit_tests[tid]:
            checkp += "assert " + io["tin"] + " == " + io["tout"] + "\n"
        ut = unit_tests[tid]
        run_test = [t["tin"] for t in ut] # 这个是用来执行的test，会返回代码执行它的结果和test_res比较得到UTfeedback
        # test_res = [literal_eval(t["tout"]) for t in ut]
        test_res = [t["tout"] for t in ut]
        # 这部分代码主要是为了加快执行速度把循环中的不变部分提取出来了,feedback prompt是针对该task的feedback的prompt的前缀，gen_expl_prefix则是用来生成代码解释的前缀
        feedback_prompt = explfeedback_prompt + assertion_strings[tid]+ "\n\n# Complete the Python funtion:\n" + problems[tid]["prompt"]+"### result ###\n```python\n" + start_code + "\n"
        gen_expl_prefix = genexpl_prompt + "\n```python\n" +start_code + "\n"
        while cir < 10:
            completion = solution
            # 这里通过一次函数调用同时获得simple和UTfeedback
            with ThreadPoolExecutor(max_workers=1) as executor:
                # print("%%%%%%%%%%%%%%%%%%% check program %%%%%%%%%%%%%%%%%%%%%%%%%%")
                # print(checkp)
                # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                args = (problems[tid], (start_code+solution), run_test, checkp, 3.0)
                future = executor.submit(run_code_with_output2, *args)
                result = future.result()
                passed = result["passed"]
                code_res = result["result"]
            print(f"task:{tid},cir:{cir},passed:{passed},result:{result}")
            if passed:
                cir = 1024
            else:
                promt_genexpl = gen_expl_prefix + solution + "\n```\n# Line-by-line explanation of the code:\n"
                input_len = len(promt_genexpl)
                inputs = tokenizer(promt_genexpl, return_tensors='pt', return_token_type_ids=False)
                inputs = inputs.to('cuda')
                pred = model.generate(**inputs, max_new_tokens=1024, temperature=0,repetition_penalty=1.1)#,temperature=0.4
                ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:].strip()
                # print(promt_genexpl)
                # print("====================origin expl=================")
                # print(ans)
                idx = ans.find("### Task End ###")
                if idx!=-1:
                    ans = ans[:idx]
                else:
                    expls=""
                    for line in ans.split("\n"):
                        if line=="":
                            continue
                        if line!="# Line-by-line explanation of the code:" and line[0]!="\"" and line[0]!="\'":
                            break
                        expls+=line+"\n"
                    ans=expls
                code_linen = len([x for x in (start_code + "\n" + solution).split("\n") if x!="" and x!="\t" and x!="    "])
                ans = "\n".join(ans.split("\n")[:code_linen])
                ans = ans.replace("\\","")
                # ans.replace("# Line-by-line explanation of the code:","Here is a line-by-line explanation of the code:")
                print("--------------------code expl-------------------")
                print(ans)
                print("------------------------------------------------")
                if type(code_res) is str:
                    prompt = feedback_prompt +solution +"\n```\n# Line-by-line explanation of the code:\n" + ans + "\n\nFeedback: With the above function, " + unit_tests[tid][0]["tin"] +" returns the following error:\n\"\"\"\n"+code_res+ "\n\"\"\"\nSo the code does not pass the assertion. Please fix it.\n### fixed result ###\n"
                else:
                    utFeedback = "\nFeedback: With the above function,"
                    for i,cres in enumerate(code_res):
                        cres = str(cres)
                        print(f"code return res is {cres}. real res is {test_res[i]}")
                        if cres == test_res[i]:
                            utFeedback += f" {run_test[i]} == {cres} while the assertion is \"{assertions[tid][i]}\".The code pass this aasertion."
                        else:
                            utFeedback += f" {run_test[i]} == {cres} while the assertion is \"{assertions[tid][i]}\".The code does not pass this aasertion."
                    utFeedback += "\nSo the code is wrong. Please fix it.\n\n### fix result ###\n"
                    prompt = feedback_prompt +solution +"\n```\n# Line-by-line explanation of the code:\n" + ans + "\n" + utFeedback
                print("---------------------feeedback prompt---------------------------")
                print(prompt[len(explfeedback_prompt):])
                print("----------------------------------------------------------------")
                prompt_len = len(prompt)
                input_prompt = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
                print("feedback prompt's token nums is :",input_prompt["input_ids"].size())
                input_prompt = input_prompt.to('cuda')
                with torch.no_grad():
                    pred = model.generate(**input_prompt, max_new_tokens=512, temperature=0,repetition_penalty=1.1)#,temperature=0.4,repetition_penalty=1.1,top_p=0.95
                    print("feedback output's token nums is :",pred.size())
                ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[prompt_len:].strip()
                # print("=================fix ans=====================")
                # print(ans)
                if "```" in ans:
                    tmp = ans.split("```")
                    if len(tmp) > 1:
                        solution = tmp[1].strip()
                    else:
                        solution = ans
                else:
                    solution = ans
                if solution.startswith("python"):
                    solution = solution[6:]
                # solution = solution.strip("\n")
                # 去除函数头和注释
                solution = code_clean2(code=solution,entry_point=entry_point,start_code=start_code).strip("\n")
                if len(solution) > 720:
                    print("the solution generated is too long!")
                    solution = ""
                print("-------------filter fix ans----------------")
                print(solution)
                print("============fix end===============")
                if completion == solution:
                    unchanged += 1
                    print(f"unchanged solution in cir[{cir}] with task {tid}")
            cir += 1

        print("tid:",tid)
        output = {"task_id": tid,"completion":solution}
        f.write(json.dumps(output)+"\n")
    f.close()


