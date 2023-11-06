import transformers, torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import read_problems
from human_eval.execution import check_test_correctness, run_code_with_output
from concurrent.futures import ThreadPoolExecutor
from myutils import map_gpu_memory,code_clean,get_args
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

prompt_file = "prompt_base.txt"
UTfeedback_file = "prompt_UTfeedback3.txt"
ut_file = "tests_from_prompt.jsonl"

def main():
    args = get_args()
    output_file = args.output
    model_path = args.model_path
    verbose = args.verbose
    unit_tests = {}
    one_test = {}

    with open(prompt_file,"r") as f:
        preflex = f.read()
        print(preflex)

    with open(UTfeedback_file,"r") as af:
        UTfeedback_promt = af.read()

    with open(ut_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["tid"]
            ios = data["ios"]
            unit_tests[tid]=ios
            one_test[tid]="assert "+ios[0]["tin"] + " == " + ios[0]["tout"]

    def get_one_complication(problem,unit_test):#为模型的输入添加前缀串，看能否提高准确度。目前直接返回原串。
        res = preflex + unit_test + "\n\n#Complete the Python funtion:\n" + problem["prompt"] + "### result ###\n"
        if verbose:
            print("=============prompt===============")
            print(res)
        return res

    #为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
    max_memory_mapping = map_gpu_memory(used_gpu=[])

    #加载模型
    print("load model from ",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True)#, use_safetensors=True

    #获取solution
    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    f = open(output_file,"w+",encoding='utf-8')
    for id in taskids:
        print("get solution for task :",id)
        problem = get_one_complication(problems[id],one_test[id])
        input_len = len(problem)
        inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
        inputs = inputs.to('cuda')
        pred = model.generate(**inputs, max_new_tokens=512, temperature=0,repetition_penalty=1.1)#,temperature=0.4
        ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len-1:]
        solution = ans.strip("\n")#.split("```")[0]#.replace("->>","")
        if verbose:
            print("=========origin solution====================")
            print(solution)
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
        solution = code_clean(code=solution,entry_point=entry_point)
        print("+++++++++++filter solution++++++++++++++++++")
        print(solution)
        print("++++++++++++++++++++++++++++++++++++++++++++")
        #self-debug step
        # entry_point = problems[id]["entry_point"]
        start_code = ""
        for line in problems[id]["prompt"].split('\"\"\"')[0].split("\n"):
            if entry_point in line and "def" in line:
                start_code += line + "\n"
                break
            start_code += line + "\n"
        print("=====start code===========")
        print(start_code)
        cir = 0 
        while cir < 10:
            completion = solution
            result = {}
            unit_result = {}
            with ThreadPoolExecutor(max_workers=1) as executor:
                checkp = ""
                for io in unit_tests[tid]:
                    checkp += "assert " + io["tin"] + " == " + io["tin"] + "\n"
                test_program = start_code + completion + "\n" + checkp
                print("%%%%%%%%%%%%%%%%%%% check program %%%%%%%%%%%%%%%%%%%%%%%%%%")
                print(test_program)
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                args = (test_program, 3.0)
                future = executor.submit(check_test_correctness, *args)
                result = future.result()
            print(f"task:{tid},cir{cir},result:{result}")
            if result["passed"]:
                cir = 1024
            else:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    ut = unit_tests[id]
                    print("check unit_test:",ut[0])
                    run_test = [ut[0]["tin"]]
                    # test_res = json.loads([ut[0]["out"]])
                    test_res = ut[0]["tout"]
                    args = (problems[id], (start_code+completion), run_test, 3.0)
                    future = executor.submit(run_code_with_output, *args)
                    result2 = future.result()#check_one_correctness(problems[id],completion,3.0)
                    code_res = result2["result"]
                    print("code_res: ",code_res)
                if type(code_res) is str and ("failed" in code_res or "time out" in code_res):
                    prompt = UTfeedback_promt + one_test[id]+ "\n# Complete the Python funtion:\n" + problems[id]["prompt"]+"### result ###\n```\n" + start_code + "\n" \
                      +solution + "\n```\nFeedback: With the above function, " + unit_tests[id][0]["tin"] +" returns the following error:\n\"\"\"\n"+code_res+ "\n\"\"\"\nSo the code does not pass the assertion. Please fix it.\n\n### fix result ###"
                else:
                    # real_res = json.loads(code_res)
                    real_res = str(code_res)
                    if real_res == test_res:
                        prompt = UTfeedback_promt + one_test[id]+ "\n# Complete the Python funtion:\n" + problems[id]["prompt"]+"### result ###\n" + start_code + "\n" \
                      +solution + "\nFeedback: With the above function, " + unit_tests[id][0]["tin"] +" == "+ str(code_res) +". The assertion is '"+one_test[id] +"'.\nSo the code pass the assertion. The code above is wrong. Please fix it.\n\n### fix result ###"
                    else:
                        print(f"real_res: #{real_res}# does not equal to test_res:#{test_res}#")
                        prompt = UTfeedback_promt + one_test[id]+ "\n# Complete the Python funtion:\n" + problems[id]["prompt"]+"### result ###\n" + start_code + "\n" \
                      +solution + "\nFeedback: With the above function, " + unit_tests[id][0]["tin"] +" == "+ str(code_res) +". The assertion is \""+one_test[id] +"\".\nSo the code does not pass the assertion. The code above is wrong. Please fix it.\n\n### fix result ###"
                # print("---------------------feeedback prompt---------------------------")
                # print(prompt)
                # print("----------------------------------------------------------------")
                input_len = len(prompt)
                inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
                inputs = inputs.to('cuda')
                pred = model.generate(**inputs, max_new_tokens=512, temperature=0,repetition_penalty=1.1)#,temperature=0.4
                ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:].strip()
                if "```" in ans:
                    tmp = ans.split("```")
                    if len(tmp) > 1:
                        solution = tmp[1].strip()
                    else:
                        solution = ans
                if solution.startswith("python"):
                    solution = solution[6:]
                solution = solution.strip("\n")
                # 去除函数头和注释
                solution = code_clean(code=solution,entry_point=entry_point)
                print("=================fix ans=====================")
                print(ans)
                print("-------------filter fix ans----------------")
                print(solution)
                print("============fix end===============")
            cir += 1

        print("id:",id)
        output = {"task_id": id,"completion":solution}
        f.write(json.dumps(output)+"\n")
        if "5" in id:
            print("Will finish")
            break
    f.close()



if __name__ == "__main__":
    main()

    


