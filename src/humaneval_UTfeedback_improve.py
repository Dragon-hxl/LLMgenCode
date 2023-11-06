import transformers, torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import read_problems
from human_eval.execution import run_code_with_output2
from concurrent.futures import ThreadPoolExecutor
from myutils import map_gpu_memory,get_args,code_clean,code_clean2
import os
from ast import literal_eval
from collections import defaultdict
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# prompt file 
prompt_file = "prompt_base2.txt"
UTfeedback_file = "prompt_UTfeedback_improve.txt"
# 从问题中提取的unit tests所在的文件
ut_file = "tests_from_prompt.jsonl"

def main():
    # 获取参数
    args = get_args()
    output_file = args.output
    model_path = args.model_path
    verbose = args.verbose
    unit_tests = {}
    assertions = defaultdict(list)
    assertion_strings = {}

    #读取prompt
    with open(prompt_file,"r") as f:
        preflex = f.read()

    with open(UTfeedback_file,"r") as af:
        UTfeedback_promt = af.read()
    #读取unit tests，保存在unit_tests，用来判断程序对错。one_test里保存了每个task的第一个unit test，这个test会用在prompt里。
    with open(ut_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["tid"]
            ios = data["ios"]
            unit_tests[tid]=ios
            for io in ios:
                assertions[tid].append("assert "+io["tin"] + " == " + io["tout"])
            assertion_strings[tid] = "\n".join(assertions[tid])
                
    # 构成生成初始代码的prompt
    def get_one_complication(tprompt,unit_test):#为模型的输入添加前缀串，看能否提高准确度。目前直接返回原串。
        res = preflex + unit_test + "\n\n# Complete the Python funtion:\n" + tprompt + "### result ###\n"
        # print("=============prompt===============")
        # print(res)
        return res

    #为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
    max_memory_mapping = map_gpu_memory(used_gpu=[])

    #加载模型
    print("load model from ",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True)#, use_safetensors=True

    #获取solution
    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    f = open(output_file,"w+",encoding='utf-8')
    for tid in taskids:
        print("get solution for task :",tid)
        # num_id = int(tid.split("/")[1])
        # if num_id < 155:
        #     continue
        tprompt = problems[tid]["prompt"]
        if tid == "HumanEval/64":
            tprompt = '''def vowels_count(s):
    """Write a function vowels_count which takes a string representing
    a word as input and returns the number of vowels in the string.
    Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a
    vowel, but only when it is at the end of the given word.

    Example:
    >>> vowels_count("abcde")
    2
    >>> vowels_count("ACEDY")
    3
    """ '''
        problem = get_one_complication(tprompt,assertion_strings[tid])
        input_len = len(problem)
        inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
        inputs = inputs.to('cuda')
        pred = model.generate(**inputs, max_new_tokens=512, temperature=0)#,temperature=0.4,repetition_penalty=1.1
        ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:]
        solution = ans.strip("\n")#.split("```")[0]#.replace("->>","")
        # if verbose:
        #     print("=========origin solution====================")
        #     print(solution)
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
        # print(f"entry_point: {entry_point}")
        print("=====start code===========")
        print(start_code)
        #
        checkp = ""
        for io in unit_tests[tid]:
            checkp += "assert " + io["tin"] + " == " + io["tout"] + "\n" # checkp把所有unit tests集合到一起用来判断程序对错
        ut = unit_tests[tid]
        run_test = [t["tin"] for t in ut] # 这个是用来执行的test，会返回代码执行它的结果和test_res比较得到UTfeedback
        # test_res = [literal_eval(t["tout"]) for t in ut]
        test_res = [t["tout"] for t in ut]
        #
        feedback_prompt = UTfeedback_promt + assertion_strings[tid] + "\n\n# Complete the Python funtion:\n" + tprompt+"### Incorrect Python Code ###\n```python\n" + start_code + "\n"
        # 开始生成feedback和新代码的循环
        cir = 0 
        unchanged = 0
        while cir < 1:
            completion = solution # 保存上一阶段的代码用来比较看代码是否变化
            # 这里通过一次函数调用同时获得simple和UTfeedback，也就是会判断代码是否正确，同时对于出现AssertError的代码会得到其执行的第一个unit test的值。其他Error因为会返回具体的错误信息就不会得到执行的第一个unit test的值。
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
                cir = 1024 #只是终结循环，并无特别意义
            else:
                if type(code_res) is str:
                    prompt =  feedback_prompt +solution + "\n```\nFeedback: With the above function, " + unit_tests[tid][0]["tin"] +" returns the following error:\n\"\"\"\n"+code_res+ "\n\"\"\"\nSo the code does not pass the assertion. The code is wrong.\n\n### a concise explanation of the issue ###\n"
                else:
                    utFeedback = "\n```\nFeedback: With the above function,"
                    for i,cres in enumerate(code_res):
                        cres = str(cres)
                        print(f"code return res is {cres}. real res is {test_res[i]}")
                        if cres == test_res[i]:
                            utFeedback += f" {run_test[i]} == {cres} while the assertion is \"{assertions[tid][i]}\".The code pass this aasertion."
                        else:
                            utFeedback += f" {run_test[i]} == {cres} while the assertion is \"{assertions[tid][i]}\".The code does not pass this aasertion."
                    utFeedback += "\nSo the code is wrong.\n\n### a concise explanation of the issue ###\n" # \nThe following is a concise explanation of the issue:\n
                    prompt = feedback_prompt +solution + utFeedback
                print("---------------------feeedback prompt---------------------------")
                print(prompt)#[len(UTfeedback_promt):]
                print("----------------------------------------------------------------")
                input_len = len(prompt)
                inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
                print("feedback prompt's token nums is :",inputs["input_ids"].size())
                inputs = inputs.to('cuda')
                with torch.no_grad():
                    pred = model.generate(**inputs, max_new_tokens=512, temperature=0)#,temperature=0.4,repetition_penalty=1.1  .8,top_p=0.95, do_sample=True
                ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:].strip("\n")
                print("=================fix ans=====================")
                print(ans)
                # if "```" in ans: # 过滤掉生成的无用信息
                #     tmp = ans.split("```")
                #     if len(tmp) > 1:
                #         sol = tmp[1].strip()
                #     else:
                #         sol = ans
                # else:
                #     sol = ans
                # if sol.startswith("python"):
                #     sol = sol[6:]
                # sol = sol.strip("\n")
                # # 去除函数头和注释
                # sol = code_clean(code=sol,entry_point=entry_point,start_code=start_code)
                # print("-------------filter fix ans----------------")
                # print(sol)
                print("============fix end===============")
                solution = ans
                # if completion == solution:
                #     unchanged += 1
                #     print(f"unchanged solution in cir[{cir}] with task {tid}")
                    # if unchanged > 4:
                    #     cir =1024
                    #     print(f"unchaged solution for 5 times, stop debug! task id is {tid},cir {cir}")
            cir += 1
        output = {"task_id": tid,"completion":solution}
        f.write(json.dumps(output)+"\n")
    f.close()

if __name__ == "__main__":
    main()

    


