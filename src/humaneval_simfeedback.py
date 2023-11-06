import transformers, torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import read_problems
from human_eval.execution import check_test_correctness
from concurrent.futures import ThreadPoolExecutor
from myutils import map_gpu_memory,code_clean2,get_args
import os
from collections import defaultdict
os.environ["TOKENIZERS_PARALLELISM"] = "true"

prompt_file = "prompt_base2.txt"
simfeedback_file = "prompt_simfeedback.txt"
ut_file = "tests_from_prompt.jsonl"

if __name__ == "__main__":
    # 获取参数
    args = get_args()
    output_file = args.output
    model_path = args.model_path
    verbose = args.verbose
    unit_tests = {}
    assertions = defaultdict(list)
    assertion_strings = {}

    with open(prompt_file,"r") as f:
        preflex = f.read()
        # print(preflex)

    with open(simfeedback_file,"r") as af:
        simde_promt = af.read()

    def get_unit_test():
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
        if verbose:
            print("=============prompt===============")
            print(res)
        return res

    #为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
    max_memory_mapping = map_gpu_memory(used_gpu=[])

    #加载模型
    print("load model from ",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True)#, use_safetensors=True

    #获取solution
    problems = read_problems()
    get_unit_test()
    taskids = list(problems.keys())
    num_task = len(taskids)
    f = open(output_file,"w+",encoding='utf-8')
    for tid in taskids:
        print("get solution for task :",tid)
        problem = get_one_complication(problems[tid],assertion_strings[tid])
        input_len = len(problem)
        inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
        inputs = inputs.to('cuda')
        pred = model.generate(**inputs, max_new_tokens=512, temperature=0)#,temperature=0.4,repetition_penalty=1.1
        ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:]
        solution = ans.strip("\n")
        # print("=========origin solution====================")
        # print(solution)
        # print("============================================")
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
        print("+++++++++++filter solution++++++++++++++++++")
        print(solution)
        print("++++++++++++++++++++++++++++++++++++++++++++")
        #self-debug step
        # entry_point = problems[tid]["entry_point"]
        start_code = ""
        for line in problems[tid]["prompt"].split("\n"):
            if line=="":
                continue
            if entry_point in line:
                start_code += line + "\n"
                break
            start_code += line + "\n"
        print("=====start code===========")
        print(start_code)
        cir = 0
        unchanged = 0
        print("###### start self-debug process ######")
        # 获取测试的代码
        checkp = ""
        for io in unit_tests[tid]:
            checkp += "assert " + io["tin"] + " == " + io["tout"] + "\n"
        # simfeedback 的prompt前缀
        simfeedback_prefix = simde_promt + assertion_strings[tid] + "\n\n# Complete the Python funtion:\n" + problems[tid]["prompt"]+"### result ###\n```python\n" + problems[tid]["prompt"]
        # solutions = []
        while cir < 10:
            pre_solution = solution
            result = {}
            with ThreadPoolExecutor(max_workers=1) as executor:
                test_program = start_code + solution + "\n" + checkp
                # print("%%%%%%%%%%%%%%%%%%% check program %%%%%%%%%%%%%%%%%%%%%%%%%%")
                # print(test_program)
                # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                args = (test_program, 3.0)
                future = executor.submit(check_test_correctness, *args)
                result = future.result()
                passed = result["passed"]
            print(f"task:{tid},cir:{cir},passed:{passed},result:{result}")
            if passed:
                cir = 1024
            else:
                prompt =  simfeedback_prefix +  solution + "\n```\nFeedback: The code above is wrong. Please fix it.\n\n### fix result ###\n"
                # print("---------------------simple feeedback prompt---------------------------")
                # print(prompt)
                # print("-----------------------------------------------------------------------")
                input_len = len(prompt)
                inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
                inputs = inputs.to('cuda')
                with torch.no_grad():
                    pred = model.generate(**inputs, max_new_tokens=512, temperature=0.4,top_p=0.95, do_sample=True)#,temperature=0.4,repetition_penalty=1.1,.8,top_p=0.95, do_sample=True
                ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:].strip("\n")
                print("------------fix ans----------------")
                print(ans)
                print("-----------------------------------")
                sol = ""
                if "```" in ans:
                    tmp = ans.split("```")
                    if len(tmp) > 1:
                        sol = tmp[1].strip()
                    else:
                        sol = ans
                else:
                    sol = ans
                if sol.startswith("python"):
                    sol = sol[6:]
                sol = sol.strip("\n")
                # 去除函数头和注释
                solution = code_clean2(code=sol,entry_point=entry_point,start_code=start_code).strip("\n")
                print(f"--------task {tid}, cir {cir}, final fix ans---------")
                print(solution)
                print("------------------------------------------------------")
                # solutions.append(solution)
                if pre_solution==solution:
                    print(f"unchanged solution in cir[{cir}] with task {tid}")
                    unchanged += 1
            cir += 1
        print("tid:",tid)
        output = {"task_id": tid,"completion":solution}
        f.write(json.dumps(output)+"\n")
    f.close()


# if unchanged > 4:
#                         print(f"unchaged solution for 5 times, stop debug! task id is {tid},cir {cir}")
#                         cir = 1024